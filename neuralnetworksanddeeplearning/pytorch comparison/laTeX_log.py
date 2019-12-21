"""The class laTeX_log will automatically log the parameters and training results to a textfile.
The text is formatted such that it looks decent in LaTeX."""

class LOG_TRAINING_SET_ERROR(Exception):
    # Exception raised when LOG_TRAINING_SET is set to false but add_trainingset_result is called
    def __init__(self, message):
        self.message = message

class laTeX_log:
    def __init__(self, RANDOM_SEED, NUMBER_OF_EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM,
                 sizes_of_layers, activation_functions,
                 trainingset_size, validationset_size, testset_size, 
                 loss_function, optimizer, LOG_TRAINING_SET = False):
        self.RANDOM_SEED = RANDOM_SEED
        self.NUMBER_OF_EPOCHS = NUMBER_OF_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.MOMENTUM = MOMENTUM
        self.sizes_of_layers = sizes_of_layers
        self.activation_functions = activation_functions
        self.trainingset_results = []
        self.validationset_results = []
        self.testset_result = ""
        self.trainingset_size = trainingset_size
        self.testset_size = testset_size
        self.validationset_size = validationset_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.LOG_TRAINING_SET = LOG_TRAINING_SET

    def add_trainingset_result(self, result):
        if( self.LOG_TRAINING_SET):
            self.trainingset_results.append(result)
        else:
            raise LOG_TRAINING_SET_ERROR("LOG_TRAINING_SET is set to false but add_trainingset_result() is called anyway")
    
    def add_validationset_result(self, result):
        self.validationset_results.append(result)

    def add_testset_result(self, result):
        self.testset_result = result
    
    def write_to_file(self, filename):

        outputfile = open(filename, 'w')
        outputfile.write(
            r"\begin{center}" + "\n")
        
        if(not self.LOG_TRAINING_SET):
            outputfile.write(
                r"\begin{tabular}{ |c|cc| }"+ "\n" +
                r"\hline" + "\n" + 
                r"\textbf{Hyperparameters, Algorithms, Net Topology} & \multicolumn{2}{|c|}{\textbf{Accuracy during Training}}  \\" + "\n")
        else:
            outputfile.write(
                r"\begin{tabular}{ |c|ccc| }"+ "\n" +
                r"\hline" + "\n" + 
                r"\textbf{Hyperparameters, Algorithms, Net Topology} & \multicolumn{3}{|c|}{\textbf{Accuracy during Training}}  \\" + "\n")
            
        outputfile.write(
            r"\hline" + "\n" +
            r"\multirow{10}{8cm}{" + r"torch.cuda.manual\_seed(" + "{})".format(self.RANDOM_SEED) + r"\\" + "\n" +
            r"torch.manual\_seed(" + "{})".format(self.RANDOM_SEED) + r"\\" + "\n" + 
            r"NUMBER\_OF\_EPOCHS =" + "{}".format(self.NUMBER_OF_EPOCHS) + r"\\"  + "\n" +
            r"BATCH\_SIZE = " + "{}".format(self.BATCH_SIZE) + r"\\" + "\n" + 
            r"LEARNING\_RATE =" +  "{}".format(self.LEARNING_RATE) +  r"\\"  + "\n" +
            r"MOMENTUM = " + "{}".format(self.MOMENTUM) + r"\\" + "\n" +
            r"sizes\_of\_layers = "  + "{}".format(self.sizes_of_layers) + r"\\" + "\n" +
            r"activation\_functions = " + "{}".format(self.activation_functions) + r"\\"  + "\n" +
            r"loss\_function = " + str(self.loss_function) + r"\\" + "\n"
            r"optimizer = " + str(self.optimizer) + r"\\" + 
            r"} & \textbf{Epoch} & ")
        
        if(self.LOG_TRAINING_SET):
            outputfile.write(
                r"\textbf{Trainingset} & \textbf{Validationset} \\" + "\n")
        else:
            outputfile.write(
                r"\textbf{Validationset} \\" + "\n")
        
        highest_index_training = 0
        highest_value_training = 0
        if(self.LOG_TRAINING_SET):
            for i in range(len(self.trainingset_results)):
                if(self.trainingset_results[i] >= highest_value_training):
                    highest_index_training = i
                    highest_value_training = self.trainingset_results[i]

        highest_index_validation = 0
        highest_value_validation = 0
        for i in range(len(self.validationset_results)):
            if(self.validationset_results[i] >= highest_value_validation):
                highest_index_validation = i
                highest_value_validation = self.validationset_results[i]
        
        if(self.LOG_TRAINING_SET):
            for i in range(len(self.validationset_results)):
                outputfile.write(" & {} & ".format(i + 1))
                if(i == highest_index_training):
                    outputfile.write(r"\textbf{" +  str(self.trainingset_results[i])  + 
                            " out of {}".format(self.trainingset_size) + r"} & ")
                    if (i == highest_index_validation):
                        outputfile.write(
                            r"\textbf{" +  str(self.validationset_results[i])  + 
                            " out of {}".format(self.validationset_size) + r"}\\" + "\n")
                    else:
                        outputfile.write(
                            str(self.validationset_results[i]) + 
                            " out of {}".format(self.validationset_size) + r"\\" + "\n")
                else:
                    outputfile.write(str(self.trainingset_results[i])  + 
                            " out of {}".format(self.trainingset_size) + " & ")
                    if (i == highest_index_validation):
                        outputfile.write(
                            r"\textbf{" +  str(self.validationset_results[i])  + 
                            " out of {}".format(self.validationset_size) + r"}\\" + "\n")
                    else:
                        outputfile.write(
                            str(self.validationset_results[i]) + 
                            " out of {}".format(self.validationset_size) + r"\\" + "\n")
        else:
            for i in range(len(self.validationset_results)):
                outputfile.write(" & {} & ".format(i + 1))
                if (i == highest_index_validation):
                    outputfile.write(
                        r"\textbf{" +  str(self.validationset_results[i])  + 
                        " out of {}".format(self.validationset_size) + r"}\\" + "\n")
                else:
                    outputfile.write(
                        str(self.validationset_results[i]) + 
                        " out of {}".format(self.validationset_size) + r"\\" + "\n")
                

        outputfile.write(
            r"\hline" + "\n"
        )   
        if (self.LOG_TRAINING_SET):
            outputfile.write(
                r"\multicolumn{4}{|c|}{"+ "\n" + 
                r"\textbf{Accuracy on Testset after Training:}} \\" + "\n" + 
                r"\multicolumn{4}{|c|}{" + "\n")
        else:
            outputfile.write(
                r"\multicolumn{3}{|c|}{"+ "\n" + 
                r"\textbf{Accuracy on Testset after Training:}} \\" + "\n" + 
                r"\multicolumn{3}{|c|}{" + "\n")
        outputfile.write(
            r"\textbf{" + str(self.testset_result) + " out of {}".format(self.testset_size) + r"}" + "\n" + 
            r"} \\" + "\n" + 
            r"\hline" + "\n" + 
            r"\end{tabular}" + "\n" + 
            r"\end{center}" + "\n")
        
        outputfile.close()
        