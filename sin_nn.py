"""

Author: Dante Tam

In this project, we approximate a linear function with a neural network.

The input is a single number x, the intermediate architecture is a substantial hidden layer,
and the final output is ideally trained to be the output lin. reg., without knowledge of the function beforehand.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

def generateTrainingSet():
  training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))

  validation_dataset = tf.data.Dataset.range(100)
  
  return training_dataset, validation_dataset


def main(_):
  # Import data
  X_train, Y_train = generateTrainingSet()
  
  print(X_train)
  print(Y_train)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 1])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))
  """
                                      
tf.app.run(main=main)