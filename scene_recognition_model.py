import os

import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical

from cnn_architecture.TransferFunction import TransferFunction, ReLuActivationFunc, SoftMaxActivationFunc
from cnn_architecture.Convolutional2D import Convolutional2DLayer
from cnn_architecture.CNN import NN
from cnn_architecture.FullyConnected import FullyConnectedLayer
from cnn_architecture.FlattenLayer import FlattenLayer
from cnn_architecture.MaxPooling import MaximumPoolingLayer
from cnn_architecture.utils import AdaptMomentumEst, CategoricalCroEntrError, QuadraticError

def get_dataset():
    """Download image classification dataset from kaggle.
    
    This does the following:
    1. use the kaggle API key to download the dataset from kaggle 
    2. unzip the data 
    3. renaming the downloaded data folders and rearranging it for ease of use
    """
    # 1. download dataset from kaggle
    os.system("pip install kaggle terminaltables")
    os.system("mkdir ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("kaggle datasets download -d puneet6060/intel-image-classification")

    # 2. unzipping downloaded data
    os.system("rm -rf seg_test seg_train seg_val")
    os.system("unzip -o intel-image-classification.zip &> /dev/null")
    
    # 3. rearranging data folders for ease of use
    os.system("mkdir seg_val && mv seg_test/seg_test/** seg_val/ && rm -rf seg_test**")
    os.system("mkdir seg_test && mv seg_pred/seg_pred/** seg_test/ && rm -rf seg_pred**")
    os.system("mv seg_train/seg_train/** seg_train/ && rm -rf seg_train/seg_train")

class CNNModel:
    """
    A class to build a CNN model. 
    
    Attributes:
    n_inputs: Any
        number of inputs to the model (image shape)
    n_outputs: Any
        number of outputs (output neurons)
    val_datas: Any
        validation data

    Methods:
    get_model():
        returns the NeuralNetwork class object.
    """
    
    def __init__(self, total_inputs, total_outputs, validation_data):
        """Constructs all the necessary attributes for the model object."""
        
        cnn = NN(type_of_enhancer=AdaptMomentumEst(), loss_function=CategoricalCroEntrError, validation_data=validation_data)
        # added one more layer 
        cnn.build(Convolutional2DLayer(size_of_input=total_inputs, filter_amount=8, size_of_kernel=(2, 2), stride=1, type_of_padding='same'))
        cnn.build(TransferFunction(ReLuActivationFunc))
        cnn.build(MaximumPoolingLayer(shape_of_pool=(2, 2), stride=2, type_of_padding='same'))

        cnn.build(Convolutional2DLayer(filter_amount=16, size_of_kernel=(2, 2), stride=1, type_of_padding='same'))
        cnn.build(TransferFunction(ReLuActivationFunc))
        cnn.build(MaximumPoolingLayer(shape_of_pool=(2, 2)))  # by default, we're choosing valid padding here as it makes more sense

        cnn.build(Convolutional2DLayer(filter_amount=32, size_of_kernel=(2, 2), stride=1, type_of_padding='same'))
        cnn.build(TransferFunction(ReLuActivationFunc))
        cnn.build(MaximumPoolingLayer(shape_of_pool=(2, 2)))  

        cnn.build(Convolutional2DLayer(filter_amount=64, size_of_kernel=(2, 2), stride=1, type_of_padding='same'))
        cnn.build(TransferFunction(ReLuActivationFunc))
        cnn.build(MaximumPoolingLayer(shape_of_pool=(2, 2)))  
        
        cnn.build(Convolutional2DLayer(filter_amount=128, size_of_kernel=(2, 2), stride=1, type_of_padding='same'))
        cnn.build(TransferFunction(ReLuActivationFunc))
        cnn.build(MaximumPoolingLayer(shape_of_pool=(2, 2)))  
    
        cnn.build(FlattenLayer())
        cnn.build(FullyConnectedLayer(256))
        cnn.build(TransferFunction(ReLuActivationFunc))

        cnn.build(FullyConnectedLayer(256))
        cnn.build(TransferFunction(ReLuActivationFunc))

        cnn.build(FullyConnectedLayer(total_outputs))
        cnn.build(TransferFunction(SoftMaxActivationFunc))

        self.model = cnn

    def get_model(self):
        """ Returns the model object. """
        
        return self.model

print("\n******************************** GETTING DATASET ********************************\n")
get_dataset()

print(CNNModel.__doc__)
# train and validation directories
train_directory = './seg_train'
val_directory = './seg_val'


# defining batch size and image size 
SIZE_OF_BATCH = 32
SIZE_OF_IMAGE = (154, 154)

# creating a tensorflow dataset using the training images
dataset_for_training = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    seed=123,
    image_size=SIZE_OF_IMAGE,
    batch_size=SIZE_OF_BATCH
    )

# print(dataset_for_training.__attributes__)

# creating a tensorflow dataset using the validation images
dataset_for_validation = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    seed=123,
    image_size=SIZE_OF_IMAGE,
    batch_size=SIZE_OF_BATCH
    )

# TODO: is this required? or can it be removed?
categories = dataset_for_training.class_names
print(categories)

# added by soha
# TODO: remove when running the final time
dataset_for_training = dataset_for_training.take(10)
dataset_for_validation = dataset_for_validation.take(10)

print("\n******************************** RESCALING AND PREPROCESSING ********************************\n")

adjusting_range = tf.keras.layers.Rescaling(1. / 255)
dataset_for_training = dataset_for_training.map(lambda x, y: (adjusting_range(x), y))
dataset_for_validation = dataset_for_validation.map(lambda x, y: (adjusting_range(x), y))

batch_of_img_features, categories_batch = next(iter(dataset_for_training))
image_1st = batch_of_img_features[0]
print("After Rescaling (Min, Max): ", np.min(image_1st), np.max(image_1st))

def preprocess_ip(dataset):
    """Returns preprocessed input. 
    
    Parameters: 
    dataset: Any
        takes in tensorflow Dataset type dataset (train and validation)
        and returns the split after separating labels from the features.
    """
    X_matrix = [] # matrix [ features (image pixels) ]
    y_vector = [] # vector [ labels (the 6 categories) ]
    for batch_of_img_features, categories_batch in dataset:
        for batch_size_num in range(SIZE_OF_BATCH):
            if batch_size_num < batch_of_img_features.shape[0]:
                X_matrix.append(batch_of_img_features[batch_size_num].numpy())
                y_vector.append(categories_batch[batch_size_num].numpy())
    X_matrix = np.array(X_matrix)
    y_vector = np.array(y_vector)

    X_matrix = np.moveaxis(X_matrix, -1, 1)
    y_vector = to_categorical(y_vector.astype("int"))

    return X_matrix, y_vector

# separating features from the labels using preprocess_input
X_train, y_train = preprocess_ip(dataset_for_training)
X_val, y_val = preprocess_ip(dataset_for_validation)
print("\nShape (X train, y train): ", X_train.shape, y_train.shape)

total_epochs = 10
SHAPE_OF_IMAGE = (3,) + SIZE_OF_IMAGE
total_outputs = 6
cnn = CNNModel(total_inputs=SHAPE_OF_IMAGE, total_outputs=total_outputs, validation_data=(X_val, y_val)).get_model()

print("\n******************************** SUMMARY OF CNN MODEL ********************************\n")
cnn.model_summary("Summary of Model")

# Model training
print("\n******************************** MODEL TRAINING ********************************\n")
training_error, validation_error, training_accuracy, validation_accuracy = cnn.fit(total_epochs=total_epochs, X=X_train, y=y_train, size_of_batch=SIZE_OF_BATCH)

print("\n******************************** MODEL PERFORMANCE AND EVALUATION ********************************\n")
print("\nAccuracy of Training Data: {:.5f}".format(100 * training_accuracy[-1]))
print("Loss of Training Data: {:.5f}".format(training_error[-1]))

print("\nAccuracy of Validation Data: {:.5f}".format(100 * validation_accuracy[-1]))
print("Loss of Validation Data: {:.5f}".format(validation_error[-1]))
print("\n")
