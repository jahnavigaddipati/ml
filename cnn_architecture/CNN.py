from datetime import datetime

from terminaltables import AsciiTable
import numpy as np

from cnn_architecture.utils import batch_iteration, calculate_diff_of_time


class NN:
    """A class that builds the CNN model and fits it over training and test data. 
    It also defines the methods for performance evaluation of the model. 
    
    Attributes
    ----------
    
    type_of _enhancer:
        It contains the type of optimizer we choose
    loss_function:
        It is a loss function that we choose for model
    validation_data:
        It is a data that is required for validation
        
    Methods 
    -------
    build(self, layer):
        It append layers one after another
    batch_eval_test(self, X, y):
        This function calculates error and accuracy for given test dataset
    batch_eval_train(self, X, y):
        This function calculates error and accuracy for given train dataset
    fit(self, X, y, nepochs, batch_size):
        This function trains the model on training and validation data and calculates evaluation matrix and time taken for training model.
    _frnt_propogation(self, X, t=True):
        Defines the forward flow of input values and returns layer output
    _bwd_propogation(self, gradient_loss):
        Defines the backward flow from the output layer and returns the gradient of loss function
    model_summary(self, name):
        It describes entire model in form of a table.
    estimate(self, X):
        It predicts layer  output  for given input values.
    """

    def __init__(self, type_of_enhancer, loss_function, validation_data=None):
        self.layers_list = []
        self.type_of_enhancer = type_of_enhancer
        self.loss_function = loss_function()
        self.error_map = {"val": [], "train": []}
        self.validation_set = None
        if validation_data:
            X, y = validation_data
            self.validation_set = {"X": X, "y": y}

    def build(self, arch_layer):
        if self.layers_list:
            arch_layer.set_input(self.layers_list[-1].return_out())
        if hasattr(arch_layer, 'initialize_value'):
            arch_layer.initialize_value(enhancer=self.type_of_enhancer)
        self.layers_list.append(arch_layer)

    def batch_eval_test(self, X, y):
        predicted_y = self._frnt_propogation(X, t=False)
        error_value = np.mean(self.loss_function.loss(y, predicted_y))
        acc_value = self.loss_function.get_acc(y, predicted_y)
        return error_value, acc_value

    def batch_eval_train(self, X, y):
        predicted_y = self._frnt_propogation(X)
        error_value = np.mean(self.loss_function.loss(y, predicted_y))
        acc_value = self.loss_function.get_acc(y, predicted_y)
        gradient_error = self.loss_function.calculate_g(y, predicted_y)
        self._bwd_propogation(gradient_loss=gradient_error)
        return error_value, acc_value

    def fit(self, X, y, total_epochs, size_of_batch):
        accuracy_train = []
        accuracy_validation = []
        begin_time = datetime.now()
        for epoch in range(total_epochs):
            loss_of_batch = []
            batch_accuracy_train = []
            val_acc_value = 0
            batch = 1
            epoch_begin_time = datetime.now()
            for X_batch, y_batch in batch_iteration(X, y, size_of_batch=size_of_batch):
                err, train_acc = self.batch_eval_train(X_batch, y_batch)
                loss_of_batch.append(err)
                batch_accuracy_train.append(train_acc)
                print("Epoch No.: {} === Training model... [ batch:{}, time taken:{} ] -> accuracy={:.2f}, loss={:.2f}"
                      .format(epoch, batch, calculate_diff_of_time(epoch_begin_time), train_acc, err), end='\r')
                batch += 1
            print("")

            if self.validation_set is not None:
                val_err, val_acc_value = self.batch_eval_test(self.validation_set["X"], self.validation_set["y"])
                self.error_map["val"].append(val_err)

            avg_train_loss = np.mean(loss_of_batch)
            avg_train_acc = np.mean(batch_accuracy_train)
            accuracy_train.append(avg_train_acc)
            accuracy_validation.append(val_acc_value)

            self.error_map["train"].append(avg_train_loss)
            print(">>> Training model (loop) complete [ epoch no.:{}, time taken: {} ] -> train accuracy:{:.2f}, train loss:{:.2f} | val accuracy:{:.2f}, val loss:{:.2f}"
                    .format(epoch, calculate_diff_of_time(epoch_begin_time), avg_train_acc, avg_train_loss, val_acc_value, val_err)
                 )
            print("\n")

        print(">>> FINAL ACCURACY:{:.2f} -> Time taken:{}".format(accuracy_validation[-1], calculate_diff_of_time(begin_time)))
        return self.error_map["train"], self.error_map["val"], accuracy_train, accuracy_validation

    def _frnt_propogation(self, X, t=True):
        layer_output = X
        for layer in self.layers_list:
            layer_output = layer.forward_feed(layer_output, t)
        return layer_output
    
    def _bwd_propogation(self, gradient_loss):
        for l in reversed(self.layers_list):
            gradient_loss = l.backward_feed(gradient_loss)

    def model_summary(self, name):
        name = "Summary of Model"
        print(AsciiTable([[name]]).table)
        print("Shape of Input: %s" % str(self.layers_list[0].size_of_input))
        print("\n")
        table_row = [["Layer Name", "Total Parameters", "Output Shape"]]
        sum_of_params = 0
        for layer in self.layers_list:
            layer_name = layer.name()
            params = layer.get_total_parameters()
            shape_of_op = layer.return_out()
            table_row.append([layer_name, str(params), str(shape_of_op)])
            # new
            table_row.append([])
            sum_of_params += params
        print(AsciiTable(table_row).table)
        print("The sum of all parameters of all layers are: %d\n" % sum_of_params)

    def estimate(self, X):
        return self._frnt_propogation(X, t=False)
