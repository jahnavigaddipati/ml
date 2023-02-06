from datetime import datetime
import math
import numpy as np

PADDING_SAME = "same"
PADDING_VALID = "valid"

"""
def create_wt_matrix_dgl(x):
    
    This function is used to generate diagonal linear product by initializing 
    a purely block diagonal weight matrix.
    
    dm = np.zeros((len(x), len(x)))
    for k in range(len(dm[0])):
        dm[k, k] = x[k]
    return dm
    """

def batch_iteration(X, y = None, size_of_batch = 64):
    """
    In this method it processes a batch of images at once for forward and backward.
    Here one iteration is described as one epoch.
    """
    no_of_samples = X.shape[0]
    for index in np.arange(0, no_of_samples, size_of_batch):
        begin, stop = index, min(index + size_of_batch, no_of_samples)
        if y is not None:
            yield X[begin:stop], y[begin:stop]
        else:
            yield X[begin:stop]
"""
def standardize(X, ord=2, ax=-1):
    
    This function is used for reducing the variable to standard or norm (here, L2 norm).

    l2_norm_val = np.atleast_1d(np.linalg.norm(X, ord, ax))
    l2_norm_val[l2_norm_val == 0] = 1
    return X / np.expand_dims(l2_norm_val, ax)
"""

def apply_padding(fil, type_of_padding = PADDING_SAME):
    """
    This function is used to add padding (pixels) to images when they are being processed,
    especially in convolutional or pooling layer. 
    It returns altered height and width of input.
    """
    if type_of_padding == "valid":
        return (0, 0), (0, 0)
    elif type_of_padding == PADDING_SAME:
        height_of_fil, width_of_fil = fil
        
        height_1 = int(math.floor((height_of_fil - 1) / 2))
        width_1 = int(math.floor((width_of_fil - 1) / 2))
        
        height_2 = int(math.ceil((height_of_fil - 1) / 2))
        width_2 = int(math.ceil((width_of_fil - 1) / 2))
        return (height_1, height_2), (width_1, width_2)

def cal_col_vals(img, fil, type_of_padding, stride=1):
    """
    This method returns the altered attributes of image by taking image and matrix filter as input. 
    """
    size_of_batch, chn, height, width = img
    fil_h, fil_w = fil
    height_padding, width_padding = type_of_padding
    hout = int((height + np.sum(height_padding) - fil_h) / stride + 1)
    wout = int((width + np.sum(width_padding) - fil_w) / stride + 1)

    p0 = np.repeat(np.arange(fil_h), fil_w)
    p0 = np.tile(p0, chn)
    
    p1 = stride * np.repeat(np.arange(hout), wout)
    
    q0 = np.tile(np.arange(fil_w), fil_h * chn)
    q1 = stride * np.tile(np.arange(wout), hout)
    
    p = p0.reshape(-1, 1) + p1.reshape(1, -1)
    q = q0.reshape(-1, 1) + q1.reshape(1, -1)

    l = np.repeat(np.arange(chn), fil_h * fil_w).reshape(-1, 1)
    
    return (l, p, q)

def image_convert_2_column(images, f, stride, p=PADDING_SAME):
    """
    This function returns the columns after converting images taken as input.
    """
    padding_h, padding_w = apply_padding(f, p)
    img_pad = np.pad(images, ((0, 0), (0, 0), padding_h, padding_w), mode='constant')
    i, q, r = cal_col_vals(images.shape, f, (padding_h, padding_w), stride)
    clmns = img_pad[:, i, q, r]
    chn = images.shape[1]
    height_of_f, width_of_f = f
    clmns = clmns.transpose(1, 2, 0).reshape(height_of_f * width_of_f * chn, -1)
    return clmns

def column_convert_2_image(clmns, shape_of_img, f, stride, p=PADDING_SAME):
    """
    This function converts the columns to images by taking columns as input.
    """
    size_of_batch, chn, height, width = shape_of_img
    padding_h, padding_w = apply_padding(f, p)
    after_p_h = height + np.sum(padding_h)
    after_p_w = width + np.sum(padding_w)
    dummy_vector = np.zeros((size_of_batch, chn, after_p_h, after_p_w))
    i, q, r = cal_col_vals(shape_of_img, f, (padding_h, padding_w), stride)
    clmns = clmns.reshape(chn * np.prod(f), -1, size_of_batch)
    clmns = clmns.transpose(2, 0, 1)
    np.add.at(dummy_vector, (slice(None), i, q, r), clmns)
    return dummy_vector[:, :, padding_h[0]:height + padding_h[0], padding_w[0]:width + padding_w[0]]


class AdaptMomentumEst:
    """A class that defines the methods that define the technique for optimizing gradient descent (Adam Optimizer). 

    Attributes
    ----------
    
    rate:
        learning rate at given time
    decay_rate1:
        aggregate of gradients at time t
    decay_rate2:
        aggregate of gradients at time t-1
    
    Methods
    -------
    def change_weight(self, original_weight, weight_grad):
        This method is used for initializing the variables required to calculate momentum
    """
    def __init__(self, learning_r=0.001, dr_1=0.9, dr_2=0.999):
        self.change = None
        self.EPSILON = 1e-8
        self.learning_r = learning_r
        self.mom = None
        self.dr_1 = dr_1
        self.dr_2 = dr_2
        self.vel = None

    def change_weight(self, og_wt, gradient_wt):
        if self.mom is None:
            self.mom = np.zeros(np.shape(gradient_wt))
            self.vel = np.zeros(np.shape(gradient_wt))

        self.mom = self.dr_1 * self.mom + (1 - self.dr_1) * gradient_wt
        self.vel = self.dr_2 * self.vel + (1 - self.dr_2) * np.power(gradient_wt, 2)

        new_vel = self.vel / (1 - self.dr_2)
        new_mom = self.mom / (1 - self.dr_1)

        self.change = self.learning_r * new_mom / (np.sqrt(new_vel) + self.EPSILON)
        return og_wt - self.change


def acc_score(y_true, y_pred):
    """
    This function is used for calculate the accurac
    """
    return np.sum(y_pred == y_true, axis=0) / len(y_true)


class Loss(object):
    """
    A class that defines the methods to calculate loss or error of model
    """
    def loss(self, real, estimated):
        pass

    def calculate_g(self, real, estimated):
        pass

    def get_acc(self, real, estimated):
        return 0


class QuadraticError(Loss):
    """
    A class that defines the methods to calculate the square loss of the model
    This class inherits super class 'Loss.'
    """
    def __init__(self): pass

    def loss(self, real, estimated):
        return 0.5 * np.power((real - estimated), 2)

    def calculate_g(self, real, estimated):
        return -(real - estimated)


class CategoricalCroEntrError(Loss):
    """
    A class that defines the methods to calculate the categorical cross entropy of the model
    This class inherits super class 'Loss.'
    """
    def __init__(self): pass

    def loss(self, real, estimated):
        estimated = np.clip(estimated, 1e-15, 1 - 1e-15)
        return - real * np.log(estimated) - (1 - real) * np.log(1 - estimated)

    def get_acc(self, real, estimated):
        return acc_score(np.argmax(real, axis=1), np.argmax(estimated, axis=1))

    def calculate_g(self, real, estimated):
        estimated = np.clip(estimated, 1e-15, 1 - 1e-15)
        return - (real / estimated) + (1 - real) / (1 - estimated)

def calculate_diff_of_time(begin_time):
    """
    This function is used for calculating the time difference.
    """
    return str((datetime.now() - begin_time)).split(".")[0]
