"""
Module designed to construct a neural network.
"""

import numpy as np
from math import ceil

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class Neuron:
    def __init__(self, nof_weights=0):
        self.__activation = None
        self.__bias = None
        self.__weights = None
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
        self.neurons = np.zeros(nof_neurons, 'O')
        for i in range(nof_neurons):
            self.neurons[i] = Neuron(nof_neurons_prev)
        return
    def get_layer_activations(self):
        a = np.array([])
        for n in self.neurons:
            a = np.append(a, n.activation)
        return a.reshape(-1,1)
    def get_layer_biases(self):
        b = np.array([])
        for n in self.neurons:
            b = np.append(b, n.bias)
        return b.reshape(-1,1)
    def get_layer_weights(self):
        w = np.array([])
        for n in self.neurons:
            w = np.append(w, n.weights)
        return w.reshape(len(self.neurons), -1)

class NeuralNetwork:
    def __init__(self, structure):
        self.layers = np.array([])
        nof_neurons = 0
        for i in structure:
            nof_neurons_prev = nof_neurons
            nof_neurons = i
            self.layers = np.append(self.layers, Layer(nof_neurons, nof_neurons_prev))
        return
    
    def set_layer_activation(self, index, values):
        layer = self.layers[index]
        l = len(layer.neurons)
        values = (np.array(values)).ravel()
        if(l - len(values)):
            return True
            #raise IndexError("Nof values and nof neurons don't match.")
        for i in range(l):
            layer.neurons[i].activation = values[i]
        return False

    def think(self, input_data):
        input_data = np.array(input_data).astype(float)
        if(self.set_layer_activation(0, input_data)):
            print("Incorrect input data")
            return
        for i in range(1, len(self.layers)):
            prev_a = self.layers[i-1].get_layer_activations()
            w = self.layers[i].get_layer_weights()
            b = self.layers[i].get_layer_biases()
            z = w.dot(prev_a) + b
            a = sigmoid(z)
            self.set_layer_activation(i, a)
        return self.layers[-1].get_layer_activations()
    def answer(self, input_data):
        def convert_from_binary(x):
            x = x.ravel()
            result = 0
            for i in range(0,4):
                result += x[-i - 1] * 2**i
            return result
        guess = self.think(input_data).round()
        return int(convert_from_binary(guess))
        #return self.think(input_data).argmax()
    def test(self, input_data, answer_sheet):
        result = [0,0] # [all_answers, right_answers]
        l = len(input_data)
        assert(l == len(answer_sheet)), "input_data and answer_sheet lengths dont match up"
        l = ceil(l/10)
        print("Testing the network...")
        for i in range(10):
            print("{}0%".format(i))
            for input, answer in zip(input_data[i*l:(i+1)*l], answer_sheet[i*l:(i+1)*l]):
                guess = self.answer(input)
                if(guess == answer): result[1] += 1
                result[0] += 1
        print("100%")
        return result
    def train(self, training_set, answers):
        return