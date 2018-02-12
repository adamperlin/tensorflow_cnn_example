from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# make sure logging verbosity is at the appropriate level
tf.logging.set_verbosity(tf.logging.INFO)

# This CNN will contain the following:

'''
Convolutional layer 1 - 32 5x5 filters, ReLU activation function

Pooling layer 1 - max pooling 2x2 filter, stride 2

Convolutional #2 - 64 5x5 filters, ReLU activation

Pooling Layer 2 - max pooling w/ 2x2 filter and stride 2

Dense layer #1: 1024 neurons, 0.4 dropout regularization rate (40% element will be dropped in training)

Dense layer 2: 10 neurons for each discrete class (digits 0-9)
'''
def main():
    print("Beginning setup...")
# entry point
if __name__ == "__main__":
    main()
