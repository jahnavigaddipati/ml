class BaseLayer(object):
    """A class that defines the basic structure of all the methods used in all the other layers. 
        
    Methods
    -------
    set_input(self, shape):
        defines the input size

    def backward_feed(self, total_grad):
      takes care of the backward flow of the layer

    def forward_feed(self, X, training):
        takes care of the front flow of the layer
 
    def return_out(self):
        returns the output

    """
    def __init__(self):
        self.size_of_input = None

    def set_input(self, shape):
        self.size_of_input = shape

    def get_total_parameters(self):
        return 0

    def name(self):
        return self.__class__.__name__
        
    def backward_feed(self, total_grad):
        print("Raising error...")
        raise NotImplementedError()

    def forward_feed(self, X, training):
        print("Raised error...")
        raise NotImplementedError()

    def return_out(self):
        print("Raising error...")
        raise NotImplementedError()
