from cnn_architecture.BaseLayer import BaseLayer
from cnn_architecture.utils import apply_padding, image_convert_2_column, column_convert_2_image
import numpy as np
import math
import copy as cp

class Convolutional2DLayer(BaseLayer):
    """A class that defines all the methods and specifications for the convolutional layer. 
    
    Attributes
    ----------
    number_of_filter: Any
        the number of filters passed to the layer
    f_size: Any
        this is the size for the filter/kernel that will slide over the input image pixels. for eg. (3,3)
    size_of_input: Any | None
        the size of the input image pixels. for eg. (100,100)
    stride: int
        this is a parameter of the filter that decides the amount of movement over the image
    type_of_padding: str
        the padding value. this can be either "same" or "valid"
        
    Methods
    -------
    initialize_values(optimizer):
        takes in the specified optimizer and initializes values for channels, weights and so on.
    
    params():
        returns the total number of parameters for the particular convolutional layer.
    
    forward_feed(X, train=True):
        defines the forward flow of input values. it makes use of utility methods to apply the filter to the input values.
        this returns the output after the filter has been applied to the input values.
    
    backward_feed(total_grad):
        defines the backward flow from the output layer. returns the total gradient.
        
    return_out():
        returns the output in the form of the filter count, and the adjusted weights.
    """
    def __init__(self, filter_amount, size_of_kernel, size_of_input=None, stride=1, type_of_padding='same'):
        self.size_of_input = size_of_input
        self.filter_amount = filter_amount
        self.type_of_padding = type_of_padding
        self.size_of_kernel = size_of_kernel
        self.stride = stride

    def initialize_value(self, enhancer):
        filter_height, filter_width = self.size_of_kernel
        to_calculate_weight = 1 / math.sqrt(np.prod(self.size_of_kernel))
        channel = self.size_of_input[0]
        self.weight = np.random.uniform(-to_calculate_weight, to_calculate_weight, size=(self.filter_amount, channel, filter_height, filter_width))
        self.weight0 = np.zeros((self.filter_amount, 1))
        self.weight_opt = cp.copy(enhancer)
        self.weight0_opt = cp.copy(enhancer)

    def get_total_parameters(self):
        weight0_shape = self.weight0.shape
        weight_shape = self.weight.shape
        # TODO: remove print statement
        print(str(weight0_shape) + " " + str(weight_shape))
        return np.prod(weight0_shape) + np.prod(weight_shape)

    def forward_feed(self, X, train=True):
        size_of_batch, channel, train_height, train_width = X.shape
        self.in_layer = X
        self.weight_cols = self.weight.reshape((self.filter_amount, -1))
        self.input_cols = image_convert_2_column(X, self.size_of_kernel, p=self.type_of_padding, stride=self.stride)
        result = self.weight_cols.dot(self.input_cols) + self.weight0
        result = result.reshape(self.return_out() + (size_of_batch,))
        return result.transpose(3, 0, 1, 2)

    def return_out(self):
        c, train_height, train_width = self.size_of_input
        padding_ht, padding_wt = apply_padding(self.size_of_kernel, type_of_padding=self.type_of_padding)
        height_output = (train_height + np.sum(padding_ht) - self.size_of_kernel[0]) / self.stride + 1
        width_output = (train_width + np.sum(padding_wt) - self.size_of_kernel[1]) / self.stride + 1
        return self.filter_amount, int(height_output), int(width_output)

    def backward_feed(self, gradient_sum):
        gradient_sum = gradient_sum.transpose(1, 2, 3, 0)
        gradient_sum = gradient_sum.reshape(self.filter_amount, -1)
        grad_weight = gradient_sum.dot(self.input_cols.T).reshape(self.weight.shape)
        grad_weight0 = np.sum(gradient_sum, keepdims=True, axis=1, )
        self.weight = self.weight_opt.change_weight(self.weight, grad_weight)
        self.weight0 = self.weight0_opt.change_weight(self.weight0, grad_weight0)
        gradient_sum = self.weight_cols.T.dot(gradient_sum)
        gradient_sum = column_convert_2_image(gradient_sum,
                                 self.in_layer.shape,
                                 self.size_of_kernel,
                                 p=self.type_of_padding,
                                 stride=self.stride,
                                )
    
        return gradient_sum
