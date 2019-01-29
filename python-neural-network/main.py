"""
Main project file
"""

#!/usr/bin/env python3
from neuralnetwork import NeuralNetwork
from importdata import import_data
from numpy.random import randint


def convert_to_binary(x):
    x = int(x)
    if(x<0): x=0
    elif(x>15): x=15
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
        result += x[-i - 1] * 2 ** i
    return result


#TRAIN_DATA = import_data("resources\\train-images.idx3-ubyte",
#                         "resources\\train-labels.idx1-ubyte")
#TEST_DATA = import_data("resources\\t10k-images.idx3-ubyte",
#                        "resources\\t10k-labels.idx1-ubyte")

NETWORK = NeuralNetwork([8,6,6,4])

input_data = []
correct_answers = []
for _ in range(16):
    for _ in range(16):
        i = randint(16)
        j = randint(16)
        input_data.append(convert_to_input(i,j))
        correct_answers.append(i^j)

result = NETWORK.test(input_data, correct_answers)
print("Network got {} questions right out of {} - {}% correct.".format(result[1], result[0], round(result[1]/result[0]*100, 2)))


print("> End of program <")