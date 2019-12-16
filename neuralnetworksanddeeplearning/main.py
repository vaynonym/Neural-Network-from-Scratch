from network import Network
import dataReader

training_data, validation_data, test_data = dataReader.load_data_wrapper()

net = Network([784,30,10])
net.SGD(training_data, 30, 10, 3.0, testData = test_data)

