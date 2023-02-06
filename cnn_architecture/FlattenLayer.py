from cnn_architecture.BaseLayer import BaseLayer
import numpy as np

class FlattenLayer(BaseLayer):
    """A class to flatten the pooled feature map into a column.
    
    Methods
    -------
    get_output():
        returns the product of array elements based on the input size
        
    forward_feed(self, X, training=True):
        returns the output after flattening and reshaping the feature map obtained from the previous layer

    backward_feed(self, total_gradient):
        returns the gradient of the loss function 

    """

    def __init__(self, size_of_input=None):
        self.prev_input_shape = None
        self.size_of_input = size_of_input

    def return_out(self):
        return (np.prod(self.size_of_input),)
    
    def backward_feed(self, gradient_sum):
        return gradient_sum.reshape(self.prev_input_shape)

    def forward_feed(self, X, t=True):
        self.prev_input_shape = X.shape
        return X.reshape((X.shape[0], -1))

    
