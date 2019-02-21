"""
Module designed to construct a neural network.
"""

import numpy as np

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def relu(x, derivative=False):
    x[x < 0] = 0
    if(derivative): x[x > 0] = 1
    return x

class Neuron:
    def __init__(self, nof_weights=0):
        if(nof_weights > 0):
            self.__bias = np.random.uniform(-1,1)
            self.__weights = np.zeros(nof_weights, 'float')
            for i in range(nof_weights):
                self.__weights[i] = np.random.uniform(-1,1)
    @property
    def activation(self):
        return self.__activation
    @activation.setter
    def activation(self, value):
        self.__activation = value
    @property
    def bias(self):
        return self.__bias
    @bias.setter
    def bias(self, value):
        self.__bias = value
    @property
    def weights(self):
        return self.__weights
    @weights.setter
    def weights(self, values):
        l = len(self.__weights)
        if(l == len(values)):
            for i in range(l):
                self.__weights[i] = values[i]

class Layer:
    def __init__(self, nof_neurons, nof_neurons_prev):
        self.activations = np.zeros(nof_neurons)
        if(nof_neurons_prev > 0):
            self.biases = np.zeros(nof_neurons)
            self.weights = np.random.randn(nof_neurons, nof_neurons_prev) * np.sqrt(2 / nof_neurons_prev)

class NeuralNetwork:
    def __init__(self, structure):
        l = len(structure)
        self.layers = np.empty(l, Layer)
        nof_neurons = 0
        for i in range(l):
            nof_neurons_prev = nof_neurons
            nof_neurons = structure[i]
            self.layers[i] = Layer(nof_neurons, nof_neurons_prev)

    def think(self, input_data):
        input_data = np.array(input_data).astype(float).ravel()
        self.layers[0].activations = input_data
        for i in range(1, len(self.layers)):
            prev_a = self.layers[i - 1].activations
            w = self.layers[i].weights
            b = self.layers[i].biases
            z = w.dot(prev_a) + b
            a = relu(z)
            self.layers[i].activations = a
        return self.layers[-1].activations

    def test(self, data):
        result = [0,0] # [all_answers, right_answers]
        for pair in data:
            guess = self.think(pair.image).argmax()
            if(guess == pair.label): result[1] += 1
            result[0] += 1
        return result

    def train(self, training_set):
        nof_batches = int(np.ceil(len(training_set) / 100))
        print("There will be {} batches of 100.".format(nof_batches))
        for i in range(nof_batches):
            #training on a single batch
            pass
