"""The class laTeX_log will automatically log the parameters and training results to a textfile.
The text is formatted such that it looks decent in LaTeX."""

class laTeX_log:
    def __init__(self, RANDOM_SEED, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
                 sizes_of_layers, activation_functions,
                 validationset_size, testset_size,
                 loss_function, optimizer):
        self.RANDOM_SEED = RANDOM_SEED
        self.NUMBER_OF_EPOCHS = NUMBER_OF_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.MOMENTUM = MOMENTUM
        self.sizes_of_layers = sizes_of_layers
        self.activation_functions = activation_functions
        self.epoch_results = []
        self.testset_result = ""
        self.testset_size = testset_size
        self.validationset_size = validationset_size
        self.loss_function = loss_function
        self.optimizer = optimizer


    def add_epoch_result(self, result):
        self.epoch_results.append(result)


    def add_testset_result(self, result):
        self.testset_result = result
    

    def write_to_file(self, filename):

        outputfile = open(filename, 'w')
        outputfile.write(
            r"\begin{center}" + "\n" +
            r"\begin{tabular}{ |c|cc| }"+ "\n" +
            r"\hline" + "\n" +
            r"\textbf{Hyperparameters, Algorithms, Net Topology} & \multicolumn{2}{|c|}{\textbf{Training (Validationset)}}  \\" + "\n" +
            r"\hline" + "\n" +
            r"\multirow{10}{8cm}{" + r"torch.cuda.manual\_seed(" + "{})".format(self.RANDOM_SEED) + r"\\" + "\n" +
            r"torch.manual\_seed(" + "{})".format(self.RANDOM_SEED) + r"\\" + "\n" + 
            r"NUMBER\_OF\_EPOCHS =" + "{}".format(self.NUMBER_OF_EPOCHS) + r"\\"  + "\n" +
            r"LEARNING\_RATE =" +  "{}".format(self.LEARNING_RATE) +  r"\\"  + "\n" +
            r"MOMENTUM = " + "{}".format(self.MOMENTUM) + r"\\" + "\n" +
            r"sizes\_of\_layers = "  + "{}".format(self.sizes_of_layers) + r"\\" + "\n" +
            r"activation\_functions = " + "{}".format(self.activation_functions) + r"\\"  + "\n" +
            r"loss\_function = " + str(self.loss_function) + r"\\" + "\n"
            r"optimizer = " + str(self.optimizer) + r"\\" + 
            r"} & \textbf{Epoch} & \textbf{Accuracy} \\" + "\n")
        
        highest_index = 0
        highest_value = 0
        for i in range(len(self.epoch_results)):
            if(self.epoch_results[i] >= highest_value):
                highest_index = i
                highest_value = self.epoch_results[i]

        for i in range(len(self.epoch_results)):
            if (i == highest_index):
                outputfile.write(
                    " & {} & ".format(i + 1) + r"\textbf{" +  str(self.epoch_results[i])  + 
                    " out of {}".format(self.validationset_size) + r"}\\" + "\n")
            else:
                outputfile.write(
                    " & {} & ".format(i + 1) + str(self.epoch_results[i]) + 
                    " out of {}".format(self.validationset_size) + r"\\" + "\n")
        
        outputfile.write(
            r"\hline" + "\n" + 
            r"\multicolumn{3}{|c|}{"+ "\n" + 
            r"\textbf{Accuracy on Testset after Training:}} \\" + "\n" + 
            r"\multicolumn{3}{|c|}{" + "\n" + 
            r"\textbf{" + str(self.testset_result) + " out of {}".format(self.testset_size) + r"}" + "\n" + 
            r"} \\" + "\n" + 
            r"\hline" + "\n" + 
            r"\end{tabular}" + "\n" + 
            r"\end{center}" + "\n"
        )
        outputfile.close()
        