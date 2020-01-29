import torch
import torch.nn as nn
from torch.autograd import Variable # for computational graphs
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
from feedforward import FeedforwardNet
from laTeX_log import laTeX_log

def adjust_learning_rate(optimizer, epoch, lr_decay, lr_decay_epoch):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch or epoch == 0:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
        print("Adjusted learning rate by a factor of {}".format(lr_decay))
    return optimizer

def adjust_momentum(optimizer, maximum, step_size):
    
    for param_group in optimizer.param_groups:
        if(param_group['momentum'] + step_size < maximum):
            param_group['momentum'] += step_size
    return optimizer
   
def train_network(net, optimizer, NUMBER_OF_EPOCHS, loss_function, trainloader, validationloader, log, SAVE_STATE_PATH, CUDA_FLAG, TRAINING_ERROR):
    # determines which state of the network will be saved
    best_validation_rate = 0


    for epoch in range(NUMBER_OF_EPOCHS):
        net.train() # enter training mode
        
        train_loader_iter = iter(trainloader)
        for batch_idx, (inputs, labels) in enumerate(train_loader_iter):
            if CUDA_FLAG:
                inputs, labels = (Variable(inputs).float()).cuda(), Variable(labels).cuda()
            else:
                inputs, labels = Variable(inputs).float(), Variable(labels)
            
            # inference
            output = net(inputs)
            loss = loss_function(output, labels.long())
            # the iconic trifecta 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Iteration: " + str(epoch + 1))
        
        
        net.eval() # leave training mode
        if(TRAINING_ERROR):
            correct = 0
            total = 0
            for inputs, labels in trainloader:
                
                if CUDA_FLAG:
                    labels = labels.cuda()
                    outputs= net(Variable(inputs.cuda()))
                else:
                    outputs = net(Variable(inputs))
                
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted==labels).sum()
            print("Accuracy on training set: {} out of {}".format(correct,total))
            log.add_trainingset_result(int(correct))

        correct = 0
        total = 0
        for (inputs, labels) in validationloader:
            if CUDA_FLAG:
                labels = labels.cuda()
                outputs= net(Variable(inputs.cuda()))
            else:
                outputs = net(Variable(inputs))

            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Accuracy on the validation set: {} out of {}'.format(correct, total))

        
        # reduce learning rate whenever a certain number of epochs are reached to better converge
        optimizer = adjust_learning_rate(optimizer, epoch, 0.9, 1)
        optimizer = adjust_momentum(optimizer, 0.9, 0.02)

        log.add_validationset_result(int(correct))

        if(correct/total >= best_validation_rate):
            best_validation_rate = correct/total
            net.save_NN_state(SAVE_STATE_PATH)