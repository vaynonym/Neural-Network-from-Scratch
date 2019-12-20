import torch # Tensor Package (for use on GPU)
import torch.nn as nn ## Neural Network package



class LengthMismatchError(Exception):
    """Exception raised when two lists whose lengths depend on one another do not match"""
    def __init__(self, message):
        self.message = message
class InvalidNumberOfLayersError(Exception):
    """Exception raised when two lists whose lengths depend on one another do not match"""
    def __init__(self, message):
        self.message = message

class FeedforwardNet(nn.Module):
    def __init__(self, sizes_of_layers, activation_functions):
        super(FeedforwardNet, self).__init__()
        if(len(sizes_of_layers) != len(activation_functions) + 1):
            raise LengthMismatchError("Length of sizes_of_layers must be length of activation_function_of_layers - 1")
        self.number_layers = len(activation_functions)
        if(self.number_layers < 1):
            raise InvalidNumberOfLayersError("A NN must have at least one layer")
        self.activations = activation_functions
        modulelist = []
        for i in range(self.number_layers):
            modulelist.append(nn.Linear(sizes_of_layers[i], sizes_of_layers[i+1]))
        self.module_list  = nn.ModuleList(modulelist)
    
    def forward(self, x):
        i = 0
        
        for module in self.module_list:
            x = module(x)
            x = self.activations[i](x)
            i = i + 1
        return x