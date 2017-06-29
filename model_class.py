
import sys
import argparse
import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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

    def __init__(self, image, label, keep_prob):
        self.image = image
        self.label = label
        self.keep_prob = keep_prob
        self.learning_rate = FLAGS.learning_rate
        self.inference
        self.optimize
        self.error

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

        print_tensor_shape(self.image, 'images shape')
        print_tensor_shape(self.label, 'label shape')

        # resize the image tensors to add channels, 1 in this case
        # required to pass the images to various layers upcoming in the graph
        images_re = tf.reshape(self.image, [-1, 28, 28, 1])
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

            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Output layer (will be transformed via stable softmax)
        with tf.name_scope('readout'):

            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])

            readout = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            print_tensor_shape(readout, 'readout shape')

        return readout

    @define_scope
    def optimize(self):

        # Compute the cross entropy.
        xe = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.inference, name='xentropy')

        # Take the mean of the cross entropy.
        loss = tf.reduce_mean(xe, name='xentropy_mean')

        # Minimize the loss by incrementally changing trainable variables.
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    @define_scope
    def error(self):

        mistakes = tf.not_equal(tf.argmax(self.label, 1),
                                tf.argmax(self.inference, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))


def train():

    # Get input data.
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)

    # Build placeholders for the input and desired response.
    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    # Instantiate a model.
    model = Model(image, label, keep_prob)

    # Instantiate a session and initialize it.
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Notional contraction loops would occur here.
    tf.summary.scalar('error', model.error)

    # Merge all the summaries and instantiate the writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    # Iterate, training the model. Outer loop iterataions evaluate test loss.
    for i in range(FLAGS.max_steps):

        # If we have reached a testing interval, test.
        if i % FLAGS.test_interval == 0:

            # Load the full dataset.
            images, labels = mnist.test.images, mnist.test.labels

            # Compute error over the test set.
            summary, error = sess.run([merged, model.error],
                                      {image: images,
                                       label: labels,
                                       keep_prob: 1.0})

            print('Test error: {:6.2f}%'.format(100 * error))

            test_writer.add_summary(summary, i)

        # Iterate, training the network.
        else:

            # Grabe a batch
            images, labels = mnist.train.next_batch(128)

            # Train the model on the batch.
            summary, _ = sess.run([merged, model.optimize],
                                  {image: images,
                                   label: labels,
                                   keep_prob: 0.8})

            train_writer.add_summary(summary, i)

    # Close the summary writers.
    test_writer.close()
    train_writer.close()


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

    parser.add_argument('--max_steps', type=int, default=100,
                        help='Number of steps to run trainer.')

    parser.add_argument('--test_interval', type=int, default=10,
                        help='Number of steps between test set evaluations.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')

    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')

    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/mnist/logs/standard_model',
                        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
