
import os
import sys
import time
import argparse
import functools
import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    Decorator source:
    https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


# function to print the tensor shape.  useful for debugging
def print_tensor_shape(tensor, string):
    ''''
    input: tensor and string to describe it
    '''

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


class Model(object):

    def __init__(self, input_size, label_size, learning_rate,
                 enqueue_threads, val_enqueue_threads, data_dir,
                 train_file, validation_file):

        # Internalize instantiation parameters
        self.label_size = label_size
        self.input_size = input_size
        self.enqueue_threads = enqueue_threads
        self.val_enqueue_threads = val_enqueue_threads
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.train_file = train_file
        self.validation_file = validation_file
		
        # Build placeholders values which change during execution.
        self.stimulus_placeholder = tf.placeholder(tf.float32,
                                                   [None, input_size])
        self.target_placeholder = tf.placeholder(tf.int32,
                                                 [None, label_size])
        self.keep_prob = tf.placeholder(tf.float32)

        # Register instance methods, building the computational graph.
        self.inference
        self.loss
        self.optimize
        self.error

        #PJT: Instantiate parameter server
        with tf.device('/job:ps/task:0'):
            #PJT: Initialize parameters as empty variables. Values to be set
            #in relevant part of neural network
            self.W_conv1 = None
            self.b_conv1 = None
            self.h_pool1 = None
            self.W_conv2 = None
            self.b_conv2 = None
            self.h_pool2 = None
            self.W_fc1 = None
            self.b_fc1 = None
            self.W_fc2 = None
            self.b_fc2 = None

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor
        (for TensorBoard visualization)."""

        with tf.name_scope('summaries'):

            mean = tf.reduce_mean(var)

            tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):

                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)

            tf.summary.scalar('max', tf.reduce_max(var))

            tf.summary.scalar('min', tf.reduce_min(var))

            tf.summary.histogram('histogram', var)

        return()

    def weight_variable(self, shape):

        initial = tf.truncated_normal(shape, stddev=0.1)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def bias_variable(self, shape):

        initial = tf.constant(0.1, shape=shape)
        self.variable_summaries(initial)
        return tf.Variable(initial)

    def conv2d(self, x, W):

        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def read_and_decode_mnist(self, filename_queue):

        # Instantiate a TFRecord reader.
        reader = tf.TFRecordReader()

        # Read a single example from the input queue.
        _, serialized_example = reader.read(filename_queue)

        # Parse that example into features.
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.input_size])

        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label_batch = features['label']

        label = tf.one_hot(label_batch,
                           self.label_size,
                           on_value=1.0,
                           off_value=0.0)

        return image, label

    def get_train_batch_ops(self, batch_size):

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.train_file)

        # Create an input scope for the graph.
        with tf.name_scope('input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename],
                                                            capacity=1)

            # Even when reading in multiple threads, share the filename queue.
            image, label = self.read_and_decode_mnist(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=10000.0,
                num_threads=self.enqueue_threads,
                min_after_dequeue=10)

        return images, sparse_labels

    def get_val_batch_ops(self, batch_size):

        # Set the filename pointing to the data file.
        filename = os.path.join(self.data_dir, self.validation_file)

        # Create an input scope for the graph.
        with tf.name_scope('val_input'):

            # Produce a queue of files to read from.
            filename_queue = tf.train.string_input_producer([filename],
                                                            capacity=1)

            # Even when reading in multiple threads, share the filename queue.
            image, label = self.read_and_decode_mnist(filename_queue)

            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images, sparse_labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                capacity=10000.0,
                num_threads=self.val_enqueue_threads,
                min_after_dequeue=10)

        return images, sparse_labels

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def inference(self, input=None):
        '''
        input: tensor of input image. if none, uses instantiation input
        output: tensor of computed logits
        '''
        # ###############################
        # print_tensor_shape(self.stimulus_placeholder, 'images shape')
        # print_tensor_shape(self.target_placeholder, 'label shape')

        # Fully-connected layer.
        # with tf.name_scope('fully_connected1'):

        #     W_fc1 = self.weight_variable([self.input_size, 16])
        #     b_fc1 = self.bias_variable([16])

        #     h_fc1 = tf.nn.relu(tf.matmul(self.stimulus_placeholder, W_fc1) + b_fc1)
        #     print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # # Dropout layer.
        # with tf.name_scope('dropout'):

        #     h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # # Output layer (will be transformed via stable softmax)
        # with tf.name_scope('readout'):

        #     W_fc2 = self.weight_variable([16, self.label_size])
        #     b_fc2 = self.bias_variable([self.label_size])

        #     readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #     print_tensor_shape(readout, 'readout shape')

        # return readout
        ###############################

        print_tensor_shape(self.stimulus_placeholder, 'images shape')
        print_tensor_shape(self.target_placeholder, 'label shape')

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        images_re = tf.reshape(self.stimulus_placeholder, [-1, 28, 28, 1])
        print_tensor_shape(images_re, 'reshaped images shape')
		
			
        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            self. W_conv1 = self.weight_variable([5, 5, 1, 32])
            self. b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            self.h_pool1 = self.max_pool_2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            self.W_conv2 = self.weight_variable([5, 5, 32, 64])
            self.b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            self.h_pool2 = self.max_pool_2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            self.b_fc1 = self.bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            self.W_fc2 = self.weight_variable([1024, 10])
            self.b_fc2 = self.bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout
        ###############################

    @define_scope
    def loss(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_placeholder, logits=self.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        return(loss)

    @define_scope
    def optimize(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.target_placeholder, logits=self.inference,
            name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        # Minimize the loss by incrementally changing trainable variables.
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.target_placeholder, 1),
                                tf.argmax(self.inference, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # tf.summary.scalar('error', error)
        return(error)


def example_usage(_):

    # Clear the log directory, if it exists.
    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Reset the default graph.
    tf.reset_default_graph()

    # Declare experimental measurement vars.
    steps = []
    val_losses = []
    train_losses = []

    # Instantiate a model.
    model = Model(FLAGS.input_size, FLAGS.label_size, FLAGS.learning_rate,
                  FLAGS.enqueue_threads, FLAGS.val_enqueue_threads,
                  FLAGS.data_dir, FLAGS.train_file, FLAGS.validation_file)

    # Get input data.
    image_batch, label_batch = model.get_train_batch_ops(batch_size=FLAGS.batch_size)

    (val_image_batch,
     val_label_batch) = model.get_val_batch_ops(batch_size=FLAGS.val_batch_size)

    tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=10.0)

    with sv.managed_session() as sess:

        # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             # sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Declare timekeeping vars.
        total_time = 0
        i_delta = 0

        # Print a line for debug.
        print('step | train_loss | train_error | val_loss | \
               val_error | t | total_time')

        # Load the validation set batch into memory.
        val_images, val_labels = sess.run([val_image_batch, val_label_batch])

        # Iterate until max steps.
        for i in range(FLAGS.max_steps):

            # Check for break.
            if sv.should_stop():
                break

            # If we have reached a testing interval, test.
            if i % FLAGS.test_interval == 0:

                # Update the batch, so as to not underestimate the train error.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

                # Make a dict to load the batch onto the placeholders.
                train_dict = {model.stimulus_placeholder: train_images,
                              model.target_placeholder: train_labels,
                              model.keep_prob: 1.0}

                # Compute error over the training set.
                train_error = sess.run(model.error, train_dict)

                # Compute loss over the training set.
                train_loss = sess.run(model.loss, train_dict)

                # Make a dict to load the val batch onto the placeholders.
                val_dict = {model.stimulus_placeholder: val_images,
                            model.target_placeholder: val_labels,
                            model.keep_prob: 1.0}

                # Compute error over the validation set.
                val_error = sess.run(model.error, val_dict)

                # Compute loss over the validation set.
                val_loss = sess.run(model.loss, val_dict)

                # Store the data we wish to manually report.
                steps.append(i)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print_tuple = (i, train_loss, train_error, val_loss,
                               val_error, i_delta, total_time)

                print('%d | %.6f | %.2f | %.6f | %.2f | %.6f | %.2f' %
                      print_tuple)

            # Hack the start time.
            i_start = time.time()

            # If it is a batch refresh interval, refresh the batch.
            if((i % FLAGS.batch_interval == 0) or (i == 0)):

                # Update the batch.
                train_images, train_labels = sess.run([image_batch,
                                                       label_batch])

            # Make a dict to load the batch onto the placeholders.
            train_dict = {model.stimulus_placeholder: train_images,
                          model.target_placeholder: train_labels,
                          model.keep_prob: FLAGS.keep_prob}

            # Run a single step of the model.
            sess.run(model.optimize, feed_dict=train_dict)

            # train_writer.add_summary(summary, i)

            i_stop = time.time()
            i_delta = i_stop - i_start
            total_time = total_time + i_delta

        # Close the summary writers.
        # test_writer.close()
        # train_writer.close()
        sv.stop()
        sess.close()

    return()

# Instrumentation: Loss function stability by batch size.


if __name__ == '__main__':

    # Instantiate an arg parser.
    parser = argparse.ArgumentParser()

    # Establish default arguements.
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data/mnist',
                        help='Directory from which to pull data.')

    parser.add_argument('--log_dir', type=str,
                        default='../log/baseline_model/',
                        help='Summaries log directory.')

    parser.add_argument('--batch_size', type=int,
                        default=128,
                        help='Training set batch size.')

    parser.add_argument('--val_batch_size', type=int,
                        default=10000,
                        help='Validation set batch size.')

    parser.add_argument('--batch_interval', type=int,
                        default=1,
                        help='Interval of steps at which a new training ' +
                             'batch is drawn.')

    parser.add_argument('--keep_prob', type=float,
                        default=1.0,
                        help='Keep probability for output layer dropout.')

    parser.add_argument('--input_size', type=int,
                        default=28 * 28,
                        help='Dimensionality of the input space.')

    parser.add_argument('--label_size', type=int,
                        default=10,
                        help='Dimensionality of the output space.')

    parser.add_argument('--train_file', type=str,
                        default='train.tfrecords',
                        help='Training dataset filename.')

    parser.add_argument('--validation_file', type=str,
                        default='validation.tfrecords',
                        help='Validation dataset filename.')

    parser.add_argument('--enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue training examples.')

    parser.add_argument('--val_enqueue_threads', type=int,
                        default=32,
                        help='Number of threads to enqueue val examples.')

    # Parse known arguements.
    FLAGS, unparsed = parser.parse_known_args()

    # Run the example usage function as TF app.
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)
