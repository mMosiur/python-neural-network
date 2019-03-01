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

data = TRAIN_DATA[:1000]

NETWORK = NeuralNetwork([784,30,10])

NETWORK.train(data, epochs=200, batch_size=10, lr=2, testing_set=data)

print("End of program.\n")