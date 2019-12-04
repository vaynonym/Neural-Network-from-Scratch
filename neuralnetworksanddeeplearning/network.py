import numpy as np

import random

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, activations):
        # Return the output of the Network if activations is input

        # b = biases, w = weights
        for b, w in zip(self.biases, self.weights):
            """ 
                the result of the previous layer (activations) is multiplied
                with the matrix correspondings to the weights, such that
                each entry of activations is matched with the correct weight
                lastly we add the bias-vector to the result and take the sigmoid function
                to compute the output of this layer
                we do this for every layer via the loop above
            """
            activations = self.sigmoid((np.dot(w , activations) + b))

        return (activations)

    def SGD(self, trainingData, epochs, miniBatchSize, epsilon, testData=None):
            """
                Trains the neural network using mini-batch stochastic gradient descent.
                trainingData is a list of tuples (x,y) where x is the training input and y
                the label (desired output). epochs is the number of steps we'll take.
            """

            if testData:
                nTest = len(testData)
                n = len(trainingData)

            for j in range(epochs):
                random.shuffle(trainingData)
                miniBatches = [trainingData[k:k+miniBatchSize] for k in range(0, n, 
                miniBatchSize)]
                
                for miniBatch in miniBatches:
                    self.updateMiniBatch(miniBatch, epsilon)

            if testData:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))
    
    def updateMiniBatch(self, miniBatch, epsilon):
        """ Update the network's weight and biases by applying gradient descent using
            backpropagation to a single mini batch. The minibatch is a list of tuples
            (x,y), and epsilon is the learning rate
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]


        """ go over each data point in the miniBatch, calculate the
            gradient via backprop, sum up the suggested change for 
            each data point. The end result is the direction we move toward
            for each bias and weight, but not yet averaged
        """
        for x,y in miniBatch:
            delta_nabla_b , delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # adjust weights and bias in the direction determined before,
        # taking into account the learning rate and averaging by dividing
        # through the length of the mini-batch
        self.weights = [w-(epsilon/len(miniBatch))*nw for w, nw in zip(self.weights,
            nabla_w)]
        self.biases = [b-(epsilon/len(miniBatch))*nb for b, nb in zip(self.biases,
            nabla_b)]



    def backprop(self, x, y):
      
        d_nabla_b = [np.zeros(b.shape) for b in self.biases]
        d_nabla_w = [np.zeros(w.shape) for w in self.weights]
       
        """A is the table of a at each node, i.e. sigmoid(z) (or the input for the first layer) """
        A = [np.zeros(y) for y in self.sizes]
        """Z is the table of z at each node, i.e. dot(w[L, j]  A[L-1]) + b[j]"""
        Z = [np.zeros(y) for y in self.sizes]
        for b, w, L in zip(self.biases, self.weights, range(self.num_layers)):
            if (L == 0):
                A[L] = x
            else:
                Z[L] = (np.dot(w , A[L-1]) + b)
                A[L] = self.sigmoid(Z[L])

        d_cost = self.cost_derivative(A[self.num_layers - 1], y)

        """ build table of partial derivatives from the top up
            these are only the general partial derivatives with respect
            to z. Like this, we only have to consider one layers per
            edge then just multiply with the value in the table
            this should boost efficiency considerably.
        """
        derivative_table = [np.zeros(y for y in self.sizes[1:])]
        for l in list(reversed(range(self.num_layers)))[:-1]:
            if (l == self.num_layers - 1): # dC/dZ[self.num_layers]
                derivative_table[l] = [ self.sigmoid_prime(Z[l][o]) * d_cost for o in self.sizes[l]]
            else: # dC/dz[L][o] = da[L][o]/dz[L][o] *  sum over q of dC/dz[L+1][q] * dz[L+1][q]/da[L][o]
                derivative_table[l] = [ self.sigmoid_prime(Z[l][o]) * np.dot(derivative_table[l+1], 
                                        (np.transpose(self.weights[l+1])[o])) for o in self.sizes[l]]

        # since the final step for the biases is dC/dz[l][j] * dz[l][j] / db[l][j] and the 
        # derivative with respect to the the bias is 1, we can just set the bias derivatives equal to the derivative_table 
        d_nabla_b = derivative_table

        for l in range(1, self.num_layers):
            for j in range(self.sizes[l]):
                for k in range(self.sizes[l-1]):
                    """ we are now determining nabla_w[l][k][j], which is the derivative of our cost function
                        with respect to w[l][k][j].
                        The last step partial derivative is unique to each edge, dC/ dz[l][j]  *  dz[l][j] / dw[l][j][k]
                    """
                    d_nabla_w[l][k][j] = derivative_table[l][j] * A[l-1][k]
        

        return (d_nabla_b, d_nabla_w)


                

    def errorFunction(self, data):
        
        """mean squared error"""
        for x, y in data:
            result += (x-y)**2
        return (result/(2* len(data)))

    def evaluate(self, testData):
        """argmax returns the index of the final layer's highest neuron
        """
        testResults = [(np.argmax(self.feedforward(x)),y) for (x, y) in testData]
    
        return (sum(int(x == y) for (x,y) in testResults))

    def cost_derivative(self, outputActivations, y):
        """return the vector of partial derivatives \partial C_x /
            \partial a for the output activations. 
        """
        return (outputActivations - y)

    def sigmoid(self, z):
        """sigmoid function"""
        return (1.0/(1.0+np.exp(-z)))


    def sigmoid_prime(self, z):
        """derivative of the sigmoid function"""
        return (self.sigmoid(z) * (1-self.sigmoid(z)))

    


    