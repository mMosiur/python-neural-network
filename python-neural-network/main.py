"""
Main project file
"""

#!/usr/bin/env python3
from neuralnetwork import NeuralNetwork
from importdata import import_data
from time import time

TRAIN_DATA = import_data("resources\\train-images.idx3-ubyte",
                         "resources\\train-labels.idx1-ubyte")
TEST_DATA = import_data("resources\\t10k-images.idx3-ubyte",
                        "resources\\t10k-labels.idx1-ubyte")

NETWORK = NeuralNetwork([784,16,16,10])

def test(testing_set, nof_items=None):
    print("Testing the network...")
    result = (NETWORK.test(testing_set), len(testing_set))
    print("Network got {} of {} right - {:5.2f}% accuracy.\n".format(result[0], result[1], result[0] / result[1] * 100))

def train(training_set, epochs, batch_size, lr, testing_set=None):
    print("Training the network...")
    NETWORK.train(training_set, epochs, batch_size, lr, testing_set)

test(TEST_DATA)
train(TRAIN_DATA, 30, 10, 1.0, TEST_DATA)



print("End of program.\n")