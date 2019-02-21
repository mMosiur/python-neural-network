"""
Main project file
"""

#!/usr/bin/env python3
from neuralnetwork import NeuralNetwork
from importdata import import_data

TRAIN_DATA = import_data("resources\\train-images.idx3-ubyte",
                         "resources\\train-labels.idx1-ubyte")
TEST_DATA = import_data("resources\\t10k-images.idx3-ubyte",
                        "resources\\t10k-labels.idx1-ubyte")

NETWORK = NeuralNetwork([28*28,16,16,10])

def test(data, nof_items=None):
    print("Testing the network...")
    result = NETWORK.test(data[:nof_items])
    print("Network got {0} questions right out of {1} - {2:5.2f}% correct.\n".format(result[1], result[0], result[1] / result[0] * 100))
def train(data, nof_items=None):
    print("Training the network...")
    NETWORK.train(data[:nof_items])
    print("Finished.\n")

test(TEST_DATA)
train(TRAIN_DATA)
test(TEST_DATA)

print("End of program.\n")