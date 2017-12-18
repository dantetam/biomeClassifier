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
import numpy as np
import math

def generateTrainingSet():
  training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))

  validation_dataset = tf.data.Dataset.range(100)
  
  return training_dataset, validation_dataset
                                      
tf.app.run(main=main)