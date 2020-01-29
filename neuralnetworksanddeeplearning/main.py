from network import Network
import dataReader

training_data, validation_data, test_data = dataReader.load_data_wrapper()
net = Network([28*28,50, 10])
net.SGD(training_data, 20, 10, 3.0, test_data)
