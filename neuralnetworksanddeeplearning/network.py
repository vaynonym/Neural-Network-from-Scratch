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



    """ L is the current layer
        k is the index of the neuron from the outgoing layer and 
        j is the index of the neuron from the layer where the edge with 
        weight w[l,j,k] is incoming

        r is the index for sum over each neuron from the previous layer we are connected to

        z = w a + b
    

        How to calculate the derivatives via chain rule
        d_C / d_w[l,j,k] = sum over r of d_cost * d_a[L,k]/d_z[L,r] * d_z[L,r] / d_w[l,j,k]
                            = sum over r of d_cost * sigmoid_prime(z[L,r]) * d_z[L,r] / d_w[l,j,k]
        when z is directly connected to w[l,j,k], this can be directly calculated
        
        otherwise we will need to keep going with the chain rule:
        d_z[L,r] / d_w[l,j,k] = sum over r_next of d_a[L,r]/d_z[L-1,r] * d_z[L-1,r_next]/d_w[l,j,k]
        If d_a[l-1,r_next] is not directly connected to d_w[l,j,k], we must continue to apply the chain
        rule. This is the case while L != l
        
        When L == l, we can calculate a direct value and the recursion stops:
        d_a[l,j] / d_w[l,j,k] = a[l-1,k] * sigmoid_prime(z)
        We will reach this base case because L will is decreased in each step by 1 until L == l and L >= l

        Since we'll do things in for loops as they are more efficient than doing things recursively, we do things backwards
    """
    def backprop(self, x, y):
      
        d_nabla_b = [np.zeros(b.shape) for b in self.biases]
        d_nabla_w = [np.zeros(w.shape) for w in self.weights]
        # loop over each training example
        for Xi,Yi in zip(x,y):
            d_cost = self.cost_derivative(Xi, Yi)
            
            """A is the table of z at each node, i.e. w[L, j, k] * A[L-1, k] + b[j]"""
            A = [np.zeros(y) for y in self.sizes]
            for b, w, L in zip(self.biases, self.weights, range(self.num_layers)):
                if (L == 0):
                    A[L] == Xi
                else:
                    A[L] = self.sigmoid( (np.dot(w , A[L-1]) + b))


            for l in range(1, self.num_layers):
                for j in range(self.sizes[l]):
                    # weights
                    for k in range(self.sizes[l-1]):
                        """ we are now determining nabla_w[l][k][j], which is the derivative of our cost function
                            with respect to w[l][k][j]. To calculate this, we go downward in the layers using
                            the chain rule outlined above.
                        """
                        # the table is used to calculate the derivatives as we go up the layers
                        # as we'll often use values we have already calculated
                        table = [np.zeros(y) for y in self.sizes[1:]]
                        for L in range(l, self.num_layers):
                            if(L == l): # a[L][k]/d_w[l][j][k] can be directly calculated:       
                                table[L][k] = A[l-1][k] * self.sigmoid_prime(A[l][k])
                            elif (L-1 == l):
                                # move the layers up, using previous results to calculate the results for the next layer
                                # here only the kth neuron of the previous layer has a connection to w[l][j][k]
                                for o in range(self.sizes[L]):
                                    table[L][o] = self.sigmoid_prime(A[L][o]) * table[L-1][j] * self.weights[L][o][k]
                            else: # each neuron will have a connection to w[l][j][k]
                                for o in range(self.sizes[L]):
                                    table[L][o] = self.sigmoid_prime(A[L][o]) * np.dot(table[L-1], self.weights[L][o])
                                    
                                    
                                    # for r in range(sizes[L-1]):
                                        # can I cut this by one loop by using dot-product on table and weights, ignoring r?
                                        # pretty sure I can
                                    #    table[L][o] = table[L][o] + table[L-1][r] * self.weights[L][o][r]
                                    # table[L][o] = table[L][o] * self.sigmoid_prime(A[L][o])
                                    
                        # since we have now determined the chain rule up to the last layer, we just need to multiply
                        # with the term from the derivative of the cost function with respect to the output of the network 
                        
                        d_nabla_w [l][k][j] = d_nabla_w[l][k][j] + sum(d_cost * table[self.num_layers])
                    
                    # biases

                    table = [np.zeros(y) for y in self.sizes[1:]]
                    for L in range(l, self.num_layers):
                        if(L == 1):
                            table[L][j] = 1
                        elif(L-1 == 1):
                            for o in range(self.sizes[L]):
                                table[L][o] = self.sigmoid_prime(A[L][j]) * table[L-1][j] * self.weights[L][o][j]
                        else:
                            for o in range(self.sizes[L]):
                                table[L][o] = self.sigmoid_prime(A[L][o]) * np.dot(table[L-1], self.weights[L][o])

                    d_nabla_b [l][j] = d_nabla_b[l][j]+ sum( d_cost * table[self.num_layers])



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

    


    