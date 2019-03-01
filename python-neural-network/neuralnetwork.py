"""
Module designed to construct a neural network.
"""

import numpy as np

def sigmoid(x, derivative=False):
    result = 1. / (1. + np.exp(-x))
    if(derivative): return result * (1. - result)
    return result

def relu(x, derivative=False):
    if(derivative): return 1 * (x > 0)
    return x * (x > 0)

act_fun = sigmoid

class Layer:
    def __init__(self, nof_neurons, nof_neurons_prev):
        self.activations = np.zeros([nof_neurons, 1])
        if(nof_neurons_prev > 0):
            self.biases = np.random.randn(nof_neurons, 1)
            self.weights = np.random.randn(nof_neurons, nof_neurons_prev)

class NeuralNetwork:
    def __init__(self, structure):
        self.layers = np.empty(len(structure), Layer)
        nof_neurons = 0
        for i in range(len(structure)):
            nof_neurons_prev = nof_neurons
            nof_neurons = structure[i]
            self.layers[i] = Layer(nof_neurons, nof_neurons_prev)

    def think(self, input_data, backprop=False):
        a = np.array(input_data).astype(float).reshape(-1,1)
        for layer in self.layers[1:]:
            prev_a = a
            w = layer.weights
            b = layer.biases
            z = w.dot(prev_a) + b
            a = act_fun(z)
            layer.activations = a
            if(backprop): layer.weighted_sums = z
        return a

    def test(self, testing_set):
        np.random.shuffle(testing_set)
        correct = 0
        for pair in testing_set:
            guess = self.think(pair.image).argmax()
            if(guess == pair.label): correct += 1
        return correct

    def gradient_step(self, batch, lr):
        for layer in self.layers[1:]: # Initializing jacobian matrices for each layer
                layer.gradient_b = np.zeros_like(layer.biases)
                layer.gradient_w = np.zeros_like(layer.weights)
        for pair in batch:
            y = np.zeros([10, 1])
            y[pair.label] = 1
            a = self.think(pair.image, backprop=True)
            delta = 2 * (a - y) * act_fun(self.layers[-1].weighted_sums, derivative=True)
            self.layers[-1].gradient_b = delta
            self.layers[-1].gradient_w = delta.dot(self.layers[-2].activations.transpose())
            for i in range(len(self.layers) - 2, 0, -1):
                delta = self.layers[i + 1].weights.transpose().dot(delta) * act_fun(self.layers[i].weighted_sums, derivative=True)
                self.layers[i].gradient_b += delta
                self.layers[i].gradient_w += delta.dot(self.layers[i - 1].activations.transpose())
        for layer in self.layers[1:]: # Applying gradient step
            layer.biases -= layer.gradient_b / len(batch) * lr
            layer.weights -= layer.gradient_w / len(batch) * lr

    def train(self, training_set, epochs=1, batch_size=None, lr=1.0, testing_set=None):
        print("Training the network...")
        for e in range(epochs):
            np.random.shuffle(training_set)
            if(batch_size == None): self.gradient_step(training_set, lr)
            else:
                for i in range(0, len(training_set), batch_size):
                    self.gradient_step(training_set[i:i + batch_size], lr)
            print("Epoch {} finished.".format(e + 1), end="")
            if testing_set:
                result = (self.test(testing_set), len(testing_set))
                print(" Network got {} of {} right - {:5.2f}% accuracy.".format(result[0], result[1], result[0] / result[1] * 100))
            else:
                print("\n")
        for layer in self.layers[1:]: # Deleting jacobian matrices of each layer
            del layer.gradient_b
            del layer.gradient_w
            del layer.weighted_sums
        print("Learning finished.")

