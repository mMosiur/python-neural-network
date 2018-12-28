"""
Module designed to construct a neural network.
"""

import numpy as np
from random import uniform

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

class Neuron:
    def __init__(self, nof_weights=0):
        self.activation = None
        if(nof_weights>0):
            self.bias = uniform(-1,1)
            self.weights = np.zeros(nof_weights, 'float')
            with np.nditer(self.weights, op_flags=['writeonly']) as iterable:
                for i in iterable:
                    i[...] = uniform(-1,1)
        return
    def get_activation(self):
        return self.activation
    def set_activation(self, value):
        self.activation = value
        return
    def get_bias(self):
        return self.bias
    def set_bias(self, value):
        self.bias = value
        return
    def get_weights(self):
        return self.weights
    def set_weights(self, values):
        l = len(self.weights)
        if(l == len(values)):
            for i in range(l):
                self.weights[i] = values[i]
        return

class Layer:
    def __init__(self, nof_neurons, nof_neurons_prev):
        self.neurons = np.zeros(nof_neurons, 'O')
        with np.nditer(self.neurons, flags=['refs_ok'], op_flags=['writeonly']) as iterable:
            for i in iterable:
                i[...] = Neuron(nof_neurons_prev)
        return
    def get_activations(self):
        a = np.array([])
        for n in self.neurons:
            a = np.append(a, n.get_activation())
        return a.reshape(-1,1)
    def get_biases(self):
        b = np.array([])
        for n in self.neurons:
            b = np.append(b, n.get_bias())
        return b.reshape(-1,1)
    def get_weights(self):
        w = np.array([])
        for n in self.neurons:
            w = np.append(w, n.get_weights())
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
    def get_layer_activations(self, index):
        return self.layers[index].get_activations()
    def get_layer_biases(self, index):
        return self.layers[index].get_biases()
    def get_layer_weights(self, index):
        return self.layers[index].get_weights() 
    def set_layer_activation(self, index, values):
        layer = self.layers[index]
        l = len(layer.neurons)
        values = (np.array(values)).ravel()
        if(l-len(values)):
            return True
            #raise IndexError("Nof values and nof neurons don't match.")
        for i in range(l):
            layer.neurons[i].set_activation(values[i])
        return False

    def think(self, input_data):
        if(self.set_layer_activation(0, input_data)):
            print("Incorrect input data")
            return
        for i in range(1, len(self.layers)):
            prev_a = self.get_layer_activations(i-1)
            w = self.get_layer_weights(i)
            b= self.get_layer_biases(i)
            a = w.dot(prev_a)+b
            self.set_layer_activation(i, sigmoid(a))
        return self.get_layer_activations(-1)
