from cnn_architecture.BaseLayer import BaseLayer

import copy as cp
import math
import numpy as np

class FullyConnectedLayer(BaseLayer):
    """A class that defines all the methods and specifications for the Dense/Fully Connected Layer. 
    
    Attributes
    ----------
    number_of_units: Any
        the number of neurons/units in the layer.
    inp_size: Any | None
        the size of the input image pixels. for eg. (100,100)
        
    Methods
    -------
    initialize_value(enhancer):
        takes in the specified enhancer and initializes values for weights and more.
    
    get_total_parameters():
        returns the total number of parameters for the particular dense layer.
    
    forward_feed(X, train=True):
        defines the forward flow of input values. it applies the weights to the input layer and returns the output of that.
    
    backward_flow(total_grad):
        defines the backward flow from the output layer. returns the total gradient.
        
    return_out():
        returns the number of output units.
    """
    def __init__(self, total_units, size_of_input=None):
        self.input_to_layer = None
        self.weight = None
        self.weight0 = None
        self.total_units = total_units
        self.size_of_input = size_of_input

    def get_total_parameters(self):
        return np.prod(self.weight.shape) + np.prod(self.weight0.shape)

    def initialize_value(self, enhancer):
        to_calculate_weight = 1 / math.sqrt(self.size_of_input[0])
        self.weight = np.random.uniform(-to_calculate_weight, to_calculate_weight, (self.size_of_input[0], self.total_units))
        self.weight0 = np.zeros((1, self.total_units))
        self.weight_opt = cp.copy(enhancer)
        self.weight0_opt = cp.copy(enhancer)

    def forward_feed(self, X, t=True):
        self.input_to_layer = X
        return self.weight0 + X.dot(self.weight)

    def backward_feed(self, gradient_sum):
        W = self.weight
        weight_of_gradient = self.input_to_layer.T.dot(gradient_sum)
        weight_of_gradient0 = np.sum(gradient_sum, axis=0, keepdims=True)
        self.weight = self.weight_opt.change_weight(self.weight, weight_of_gradient)
        self.weight0 = self.weight0_opt.change_weight(self.weight0, weight_of_gradient0)
        gradient_sum = gradient_sum.dot(W.T)
        return gradient_sum

    def return_out(self):
        return (self.total_units,)
