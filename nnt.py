#! /usr/bin/python

''' A simple multilyaer Perceptron

Mehdi Mirza [memirzamo at gmail]
August 2011
'''

import numpy as np


class Hidden_layer:

    def __init__(self, num_in. num_out):

        self.in_data = in_data
        self.weights = np.random.uniform(-2.0, 2.0,
                            (num_in, num_out))
        self.bias = 1.0

    def activation(self, input_data):

        return self._sigmoid(self.in_data * self.weights + self.bias)


class NeuralNet:

   
    def __init__(self, num_visible, num_hidden, num_output):
        
        # Number of units
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.nun_output = num_output

        # unit activations
        self.a_visible = np.ones(self.num_visible)
        self.a_hidden = np.ones(self.num_hidden)
        self.a_output = np.ones(seld.num_output)

        # weights
        self.weights_visible = np.random.unifrom(-2.0, 2.0, 
                            (self.num_visible, self.num_hidden))
        self.weights_hidden = np.random.unifrom(-2.0, 2.0, 
                            (self.num_hidden, self.num_output))

        # bias
        self.bias_visible = 1.0
        self.bias_hidden = 1.0

    
    def forward(data):
        '''Updates the network in forward pass'''\

        # visible layer
        self.a_visible = data

        # hidden layer
        self.a_hidden = self._sigmoid(self.a_visible * self.weights_visible \
                                        + self.bias_visible)

        # output layer
        self.a_output = self._sigmoid(self.a_hidden
        

    def backpropagte():
        '''computes the garident of cost function with back propagation'''

    def train():

        for epoch in range(num_epochs):
            


    def test():


    def _sigmoid(self, x):
        '''Sigmoid function'''

        return 1.0 / (1 + np.exp(-x)
