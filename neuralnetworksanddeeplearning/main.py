from network import Network
import dataReader as dataReader

trainingData, validationData, testData = dataReader.load_data_wrapper()
net = Network([28*28,50, 10])
net.SGD(trainingData, 28, 10, 3.0, testData)
