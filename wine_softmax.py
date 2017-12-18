# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""

TensorFlow application meant to show the convergence of a logistic regression 
in a properly defined experiment (train, valid, test)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import scipy.io

FLAGS = None

validProp = 0.1

# Shuffle two arrays that are paired together
def shuffle(a, b):
  assert len(a) == len(b)
  p = np.random.permutation(len(a))
  return a[p], b[p]

def main(_):
  # Import data
  wine_data = scipy.io.loadmat("./wine_data.mat")
  X_data, Y_data = wine_data["X"], wine_data["y"]
  # print(X_data.shape)
  
  X_data_shuf, Y_data_shuf = shuffle(X_data, Y_data)
  X_train = X_data_shuf[int(validProp * X_data.shape[0]):]
  Y_train = Y_data_shuf[int(validProp * Y_data.shape[0]):]
  X_valid = X_data_shuf[0 : int(validProp * X_data.shape[0])]
  Y_valid = Y_data_shuf[0 : int(validProp * Y_data.shape[0])]
  X_test = wine_data["X_test"]
  
  # print(X_valid, Y_valid)
  
  # Create the model
  x = tf.placeholder(tf.float32, [None, 12])
  W = tf.Variable(tf.zeros([12, 1]))
  b = tf.Variable(tf.zeros([1]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 1])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for iterIndex in range(25):
    X_train_shuf, Y_train_shuf = shuffle(X_train, Y_train)
    batch_xs = X_train_shuf[0:int(X_train_shuf.shape[0]*0.1)]
    batch_ys = Y_train_shuf[0:int(Y_train_shuf.shape[0]*0.1)]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: X_valid, y_: Y_valid}))
 
  testResults = sess.run(y, feed_dict={x: X_test})

  print("--------")
  print(testResults)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  #parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      #help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)