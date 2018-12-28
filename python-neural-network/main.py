"""
Main project file
"""

#!/usr/bin/env python3
from neuralnetwork import NeuralNetwork
from importdata import import_data


def convert_to_binary(x):
    result = [0,0,0,0]
    for i in range(4):
        result[i] = x&1
        x=x>>1
    return tuple(reversed(result))
def convert_to_input(x, y):
    return convert_to_binary(x)+convert_to_binary(y)
def convert_from_binary(x):
    result = 0
    for i in range(0,4):
        result += x[-i-1]*2**i
    return result


IMAGES_SOURCE = "resources/train-images.idx3-ubyte"
LABELS_SOURCE = "resources/train-labels.idx1-ubyte"
DATA = import_data(IMAGES_SOURCE, LABELS_SOURCE)

NETWORK = NeuralNetwork([8,6,6,4])

input_data = convert_to_input(12,10)
output = NETWORK.think(input_data)
print(output)


print("End of program")