from cnn_architecture.BaseLayer import BaseLayer
import numpy as np

class TransferFunction(BaseLayer):
    """Defines the Activation Functions that we're using to build our model.
    
    Attributes
    ----------
    func: Any
        object of a specific Activation Function
        
    Methods
    -------
    return_out():
        returns the input size taken in as input to apply the activation function on

    name():
        returns the name of the activation function we're using
        
    forward_feed(val, training=True):
        returns the output after applying the activation function on the output of the previous layer
    
    backward_feed(total_gradient):
        returns the gradient of the loss function 
    """
    def __init__(self, function):
        self.arch_layer = None
        self.function = function()

    def return_out(self):
        return self.size_of_input

    def name(self):
        activation_func_name = "Activation/Transfer Function: " + self.function.__class__.__name__ + " "
        return activation_func_name

    def forward_feed(self, layer_name, t=True):
        self.arch_layer = layer_name
        return self.function(layer_name)

    def backward_feed(self, gradient_sum):
        return gradient_sum * self.function.grad(self.arch_layer)

class ReLuActivationFunc:
    """A class to define specific methods for ReLu Activation Function.
    
    Methods
    -------
    grad(val):
        returns the gradient of the output of the layer, to the next layer
    """
    def grad(self, ip):
        return np.where(ip < 0, 0, 1)

    def __call__(self, input):
        return np.where(input < 0, 0, input)

class SoftMaxActivationFunc:
    """A class to define specific methods for SoftMax activation Function. 
    
    Methods
    -------
    grad(data):
        returns the output after application of SoftMax Activation Function on the output of the last layer
    """
    def grad(self, data):
        probs = self.__call__(data)
        return probs * (1 - probs)

    def __call__(self, input):
        res = np.exp(input - np.max(input, keepdims=True, axis=-1))
        return res / np.sum(res,keepdims=True, axis=-1,)



