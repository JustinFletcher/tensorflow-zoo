import os
import sys
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
TRAIN_FILE = 'cso-train.tfrecords'
VALIDATION_FILE = 'cso-validation.tfrecords'


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
            'label': tf.FixedLenFeature([2], tf.int64)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    # TODO: Require specification, rather than mnist dependence.
    # image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.decode_raw(features['image_raw'], tf.int16)
    # image.set_shape([mnist.IMAGE_PIXELS])
    # image.set_shape([262144])
    # print(262144)
    image.set_shape([512 * 512 * 5])
    # print(1310720)
    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.cast(image, tf.float32) * (1. / 65535) - 0.5

    # Convert label from int64 tensor to an int32 tensor.
    label = tf.cast(features['label'], tf.float32)

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
                                                        num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

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

    def conv3d(self, x, W):

        return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):

        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def max_pool_2x2x2(self, x):

        return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                                strides=[1, 2, 2, 2, 1], padding='SAME')

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
        # images_re = tf.reshape(self.stimulus_placeholder, [-1, 28, 28, 1])
        # images_re = tf.reshape(self.stimulus_placeholder, [-1, 512, 512, 5])
        images_re = tf.reshape(self.stimulus_placeholder, [-1, 512, 512, 5, 1])
        print_tensor_shape(images_re, 'reshaped images shape')

        # Convolution layer.
        with tf.name_scope('Conv1'):

            # weight variable 4d tensor, first two dims are patch (kernel) size
            # 3rd dim is number of input channels, 4th dim is output channels
            # W_conv1 = self.weight_variable([5, 5, 1, 32])
            # W_conv1 = self.weight_variable([5, 5, 5, 32])

            # weight variable 5d tensor, first 3 dims are patch (kernel) size
            # 4th dim is number of input channels, 5th dim is output channels
            # W_conv1 = self.weight_variable([5, 5, 3, 1, 32])
            W_conv1 = self.weight_variable([5, 5, 3, 1, 8])
            # b_conv1 = self.bias_variable([32])
            b_conv1 = self.bias_variable([8])
            # h_conv1 = tf.nn.relu(self.conv2d(images_re, W_conv1) + b_conv1)
            h_conv1 = tf.nn.relu(self.conv3d(images_re, W_conv1) + b_conv1)
            print_tensor_shape(h_conv1, 'Conv1 shape')

        # Pooling layer.
        with tf.name_scope('Pool1'):

            # h_pool1 = self.max_pool_2x2(h_conv1)
            h_pool1 = self.max_pool_2x2x2(h_conv1)
            print_tensor_shape(h_pool1, 'MaxPool1 shape')

        # Conv layer.
        with tf.name_scope('Conv2'):

            # W_conv2 = self.weight_variable([5, 5, 32, 64])
            # W_conv2 = self.weight_variable([5, 5, 3, 32, 64])
            W_conv2 = self.weight_variable([5, 5, 3, 8, 16])
            # b_conv2 = self.bias_variable([64])
            b_conv2 = self.bias_variable([16])
            h_conv2 = tf.nn.relu(self.conv3d(h_pool1, W_conv2) + b_conv2)
            print_tensor_shape(h_conv2, 'Conv2 shape')

        # Pooling layer.
        with tf.name_scope('Pool2'):

            # h_pool2 = self.max_pool_2x2(h_conv2)
            h_pool2 = self.max_pool_2x2x2(h_conv2)
            print_tensor_shape(h_pool2, 'MaxPool2 shape')

        # Fully-connected layer.
        with tf.name_scope('fully_connected1'):

            # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # h_pool2_flat = tf.reshape(h_pool2, [-1, 128 * 128 * 64])
            # h_pool2_flat = tf.reshape(h_pool2, [-1, 128 * 128 * 2 * 64])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 128 * 128 * 2 * 16])
            print_tensor_shape(h_pool2_flat, 'MaxPool2_flat shape')

            # W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            # W_fc1 = self.weight_variable([128 * 128 * 64, 1024])
            # W_fc1 = self.weight_variable([128 * 128 * 2 * 64, 1024])
            W_fc1 = self.weight_variable([128 * 128 * 2 * 16, 1024])
            b_fc1 = self.bias_variable([1024])

            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            print_tensor_shape(h_fc1, 'FullyConnected1 shape')

        # Dropout layer.
        with tf.name_scope('dropout'):

            h_fc1_drop = tf.nn.dropout(h_fc1, FLAGS.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            # W_fc2 = self.weight_variable([1024, 10])
            # b_fc2 = self.bias_variable([10])
            W_fc2 = self.weight_variable([1024, 2])
            b_fc2 = self.bias_variable([2])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout

    @define_scope
    def loss(self):

        self.target_placeholder = tf.to_int64(self.target_placeholder)

        # Compute the cross entropy.
        # xe = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.target_placeholder, logits=self.inference,
        #     name='xentropy')

        # Take the mean of the cross entropy.
        # loss_val = tf.reduce_mean(xe, name='xentropy_mean')
        # loss_val = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.target_placeholder, self.inference))))
        loss_val = tf.losses.mean_squared_error(
            labels=self.target_placeholder, predictions=self.inference)

        # Add a scalar summary for the snapshot loss.
        # tf.summary.scalar('cross_entropy', loss_val)
        tf.summary.scalar('regression_loss', loss_val)

        return loss_val

    @define_scope
    def optimize(self):

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Minimize the loss by incrementally changing trainable variables.
        return tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, global_step=global_step)

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.target_placeholder, 1),
                                tf.argmax(self.inference, 1))

        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        # tf.summary.scalar('error', error)

        return error

# TODO: Convert to QueueRunners


def train():

    # Get input data.
    images, labels = inputs(train=True,
                            batch_size=FLAGS.batch_size,
                            num_epochs=FLAGS.num_epochs)

    model = Model(images, labels)

    merged = tf.summary.merge_all()

    # Instantiate a session and initialize it.
    sv = tf.train.Supervisor(logdir=FLAGS.log_dir, save_summaries_secs=5)

    with sv.managed_session() as sess:

        sv.start_standard_services(sess=sess)

        # Start input enqueue threads.
        sv.start_queue_runners(sess=sess)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        # Iterate, training the model.
        for i in range(FLAGS.max_steps):

            if sv.should_stop():
                break

            # If we have reached a testing interval, test.
            if i % FLAGS.test_interval == 0:

                # Compute loss over the test set.
                summary, loss = sess.run([merged, model.loss])
                print(sess.run(labels))
                print(sess.run(model.inference))
                print('Step %d: loss = %.2f' % (i, loss))
                test_writer.add_summary(summary, i)

            # Iterate, training the network.
            else:

                # Grab a batch
                # images, labels = mnist.train.next_batch(128)

                # Train the model on the batch.
                summary, _ = sess.run([merged, model.optimize])
                train_writer.add_summary(summary, i)

    sv.request_stop()
    sv.coord.join()
    test_writer.close()
    train_writer.close()
    sv.stop()
    sess.close()


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

    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')

    # parser.add_argument('--test_interval', type=int, default=10,
    parser.add_argument('--test_interval', type=int, default=10,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='../data',
                        help='Directory for storing input data')

    parser.add_argument('--log_dir', type=str,
                        default='../data/tensorboard',
                        help='Summaries log directory')

    parser.add_argument('--batch_size', type=int,
                        default=8,
                        help='Batch size.')

    parser.add_argument('--num_epochs', type=int,
                        default=10,
                        help='Number of epochs.')

    parser.add_argument('--train_dir', type=str,
                        default='../data',
                        help='Directory with the training data.')

    parser.add_argument('--keep_prob', type=float,
                        default=0.5,
                        help='Keep probability for output layer dropout.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
