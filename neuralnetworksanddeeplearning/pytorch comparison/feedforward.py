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
    def __init__(self, sizes_of_layers, activation_functions, USE_DROPOUT = False, dropout_probability = 0):
        super(FeedforwardNet, self).__init__()
        if(len(sizes_of_layers) != len(activation_functions) + 1):
            raise LengthMismatchError("Length of sizes_of_layers must be length of activation_function_of_layers - 1")
        self.number_layers = len(activation_functions)
        if(self.number_layers < 1):
            raise InvalidNumberOfLayersError("A NN must have at least one layer")
        self.USE_DROPOUT = USE_DROPOUT
        self.activations = activation_functions
        modulelist = []
        dropout_modulelist = []
        for i in range(self.number_layers):
        
            modulelist.append(nn.Linear(sizes_of_layers[i], sizes_of_layers[i+1]))
            
            
        self.module_list  = nn.ModuleList(modulelist)
        
        self.dropout = nn.Dropout(dropout_probability)
        self.dropout_input = nn.Dropout(0.2)
        self.dropout_probability = dropout_probability

    
    def forward(self, x):
        i = 0
        if(self.USE_DROPOUT):
            x = self.dropout_input(x)
        
        for module in self.module_list:

            x = module(x)
            # apparently the last layer of activations breaks the cost function, so we ignore them
            
            if(i != (len(self.module_list) - 1) ):
                x = self.activations[i](x)
            
            if(self.USE_DROPOUT):
                if(i != (len(self.module_list) - 2) ):
                    x = self.dropout(x)
                    if(not self.training): # to compensate for training with not all nodes active and the ensuing higher weights
                        x = x * (1-self.dropout_probability)
                
            
            i = i + 1
        return x

    def save_NN_state(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load_NN_state(self, PATH):
        self.load_state_dict(torch.load(PATH))
        self.eval()
 

    """This function attempts to load all layers, excluding the last one, into the network 
    based on a file provided by PATH. This even works when the model has more layers than
    the one of the loaded state. However, the sizes of the layers must still fit.
    """
    def load_partial_NN_state(self, PATH):
        loaded_state = torch.load(PATH)
        own_state = self.state_dict()
        for name, param in loaded_state.items():
            if((name, param) != list(loaded_state.items())[-1]
                and (name, param) != list(loaded_state.items())[-2]):
                
                own_state[name].copy_(param)

