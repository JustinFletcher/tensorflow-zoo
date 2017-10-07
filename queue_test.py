# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:41:43 2017
@author: tomhope
"""

import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

#READ
NUM_EPOCHS = 10

filename = os.path.join("C:\\mnist","train.tfrecords")

filename_queue = tf.train.string_input_producer(
    [filename], num_epochs=NUM_EPOCHS)


reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])  

image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

label = tf.cast(features['label'], tf.int32)



# Shuffle the examples + batch
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000,
    num_threads=8)


W = tf.get_variable("W", [28*28, 10])
y_pred = tf.matmul(images_batch, W)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
init = tf.local_variables_initializer()
sess.run(init)

## coordinator 
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print(threads)

try:
  step = 0
  while not coord.should_stop():  
      step += 1
      print(step)
      sess.run([train_op])
      if step%500==0:
          loss_mean_val = sess.run([loss_mean])
          print(step)
          print(loss_mean_val)
except tf.errors.OutOfRangeError:  
    print('Done training for %d epochs, %d steps.' % (NUM_EPOCHS, step))
finally:
    # When done, ask the threads to stop.
    coord.request_stop()


#example -- get image,label
img1, lbl1 = sess.run([image,label])

#example - get random batch
labels, images= sess.run([labels_batch, images_batch])