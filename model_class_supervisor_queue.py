import os
import sys
import time
import argparse
import functools
import tensorflow as tf


def segmentation_layer(prev_layer):
    # Conv layer to generate the 2 score classes
    with tf.name_scope('Score_classes'):

        W_score_classes = tf.Variable(tf.truncated_normal([1, 1, 300, 2],
                                                          stddev=0.1,
                                                          dtype=tf.float32),
                                      name='W_score_classes')

        print_tensor_shape(W_score_classes, 'W_score_classes_shape')

        score_classes_conv_op = tf.nn.conv2d(prev_layer, W_score_classes,
                                             strides=[1, 1, 1, 1],
                                             padding='SAME',
                                             name='score_classes_conv_op')

        print_tensor_shape(score_classes_conv_op, 'score_conv_op shape')

    # Upscore the results to 256x256x2 image
    with tf.name_scope('Upscore'):

        W_upscore = tf.Variable(tf.truncated_normal([31, 31, 2, 2],
                                                    stddev=0.1,
                                                    dtype=tf.float32),
                                name='W_upscore')
        print_tensor_shape(W_upscore, 'W_upscore shape')

        upscore_conv_op = tf.nn.conv2d_transpose(score_classes_conv_op,
                                                 W_upscore,
                                                 output_shape=[1, 256, 256, 2],
                                                 strides=[1, 16, 16, 1],
                                                 padding='SAME',
                                                 name='upscore_conv_op')

        print_tensor_shape(upscore_conv_op, 'upscore_conv_op shape')

    return upscore_conv_op


# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'
IMAGE_PIXELS = 28 * 28


def read_and_decode(filename_queue):

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
    image.set_shape([IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
                  train forever.
    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
      * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs:
        num_epochs = None

    # Set the filename pointing to the data file.
    filename = os.path.join(FLAGS.train_dir,
                            TRAIN_FILE if train else VALIDATION_FILE)

    # Create an input scope for the graph.
    with tf.name_scope('input'):

        # Produce a queue of files to read from.
        filename_queue = tf.train.string_input_producer([filename],
                                                        capacity=1000)

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=1000000.0 * batch_size,
            num_threads=15,
            min_after_dequeue=1)

    return images, sparse_labels


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


class Model:

    def __init__(self, stimulus_placeholder, target_placeholder):

        self.stimulus_placeholder = stimulus_placeholder
        self.target_placeholder = target_placeholder
        self.learning_rate = FLAGS.learning_rate
        self.inference
        self.loss
        self.optimize
        self.error

        # tf.summary.scalar('error', self.error)

        # # Merge all the summaries and instantiate the writers
        # # merged_summaries = tf.summary.merge_all()

        # self.summaries = merged_summaries

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

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def inference(self, input=None):
        '''
        input: tensor of input image. if none, uses instantiation input
        output: tensor of computed logits
        '''

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
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            h_pool1 = self.max_pool_2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            h_pool2 = self.max_pool_2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, FLAGS.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout

    @define_scope
    def loss(self):

        # Compute the cross entropy.
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int64(self.target_placeholder), logits=self.inference,
            name='xentropy1')

        # Take the mean of the cross entropy.
        loss_val = tf.reduce_mean(xe, name='xentropy_mean1')

        # Add a scalar summary for the snapshot loss.
        tf.summary.scalar('cross_entropy1', loss_val)

        return loss_val

    @define_scope
    def optimize(self):

        # Create a variable to track the global step.
        # global_step = tf.Variable(0, name='global_step', trainable=False)

        # # Minimize the loss by incrementally changing trainable variables.
        # return tf.train.AdamOptimizer(self.learning_rate).minimize(
        #     self.loss, global_step=global_step)

        # Compute the cross entropy.

        # self.target_placeholder = tf.to_int64(self.target_placeholder)

        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int64(self.target_placeholder), logits=self.inference,
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

        return error


def train():

    # Get input data.
    images, labels = inputs(train=True,
                            batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    model = Model(images, labels)

    init_local = tf.local_variables_initializer()
    init_global = tf.global_variables_initializer()

    tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=10.0)

    # sess = tf.Session()

    # with tf.Session() as sess:
    with sv.managed_session() as sess:

        # sess.run(init_local)
        # sess.run(init_global)

        # sv.start_standard_services(sess=sess)

        # sv.start_queue_runners

        # # Start input enqueue threads.
        # threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        # init = tf.global_variables_initializer()
        sess.run(init_local)
        # init = tf.local_variables_initializer()
        sess.run(init_global)

        # coordinator
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=sv.coord)

        print(threads)

        for t in range(20):
            time.sleep(1)
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):

                print(sess.run(qr.queue.size()))

        total_time = 0
        i_delta = 0

        # Iterate, training the model.
        for i in range(FLAGS.max_steps):

            i_start = time.time()

            sess.run(model.optimize)

            # if sv.should_stop():
            #     break

            i_stop = time.time()
            i_delta = i_stop - i_start
            total_time = total_time + i_delta

            # If we have reached a testing interval, test.
            if i % FLAGS.test_interval == 0:

                # for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):

                #     print(sess.run(qr.queue.size()))

                # Compute loss over the test set.
                loss = sess.run(model.loss)
                # loss = 1.0
                # error = sess.run(model.error)
                print('Step %d:  loss = %.2f, t = %.6f, total_t = %.2f, ' % (i, loss, i_delta, total_time))
                # test_writer.add_summary(summary, i)

            # Iterate, training the network.
            # else:

                # Grab a batch
                # images, labels = mnisse:t.train.next_batch(128)

                # Train the model on the batch.
                # train_writer.add_summary(summary, i)

        coord.request_stop()
        coord.join(threads)


    # sv.request_stop()
    # sv.coord.join()
    # sv.stop()



    # test_writer.close()
    # train_writer.close()
    # sess.close()

def main(_):

    if tf.gfile.Exists(FLAGS.log_dir):

        tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)

    train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')

    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=100,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data',
                        help='Directory for storing input data')

    parser.add_argument('--log_dir', type=str,
                        default='../tensorboard',
                        help='Summaries log directory')

    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='Batch size.')

    parser.add_argument('--num_epochs', type=int,
                        default=0,
                        help='Number of epochs.')

    parser.add_argument('--train_dir', type=str,
                        default='../data',
                        help='Directory with the training data.')

    parser.add_argument('--keep_prob', type=float,
                        default=0.5,
                        help='Keep probability for output layer dropout.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
