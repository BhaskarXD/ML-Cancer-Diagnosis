import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
import sys
import pdb
from copy import deepcopy
import aggregation
import nets
import utils

def main(args):
    
    num_workers = args.num_workers
    num_rounds = args.num_rounds
    num_iters = args.num_iters
    
    if args.gpu == -1: device = torch.device('cpu')
    else: device = torch.device('cuda')

    filename = args.exp

    train_data, test_data, net, num_inputs, num_outputs, lr_init, batch_size = utils.load_data(args.dataset)
    net.to(device) 
    distributed_data, distributed_labels, wts = utils.distribute_data(train_data, args.bias, num_workers, num_outputs, device) 
    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(num_rounds)
    
    batch_idx = np.zeros(num_workers)
    faba_client_list = []
    fg_client_list = []
    weight = torch.ones(num_workers)
    for rnd in range(num_rounds):
        grad_list = []
        if (args.dataset == 'cifar10'):
            lr = utils.get_lr(rnd, num_rounds, lr_init)
        else: lr = lr_init
        for worker in range(num_workers):
            net_local = deepcopy(net) 
            net_local.train()
            optimizer = optim.SGD(net_local.parameters(), lr=lr)

            for local_iter in range(num_iters):
                optimizer.zero_grad()
                #sample local dataset in a round-robin manner
                if (batch_idx[worker]+batch_size < distributed_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                    batch_idx[worker] = batch_idx[worker] + batch_size
                else: 
                    minibatch = np.asarray(list(range(int(batch_idx[worker]), distributed_data[worker].shape[0]))) 
                    batch_idx[worker] = 0
                output = net_local(distributed_data[worker][minibatch].to(device))
                loss = criterion(output, distributed_labels[worker][minibatch].to(device))
                loss.backward()
                optimizer.step()
                    
            ##append all gradients in a list
            grad_list.append([(x-y).detach() for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
            
            del net_local, output, loss
            torch.cuda.empty_cache()
        
        ###Do the aggregation
        if (args.aggregation == 'fedsgd'):
            net = aggregation.FEDSGD(device, lr, grad_list, net, wts) 
        elif (args.aggregation == 'krum'):
            net = aggregation.krum(device, lr, grad_list, net, args.cmax)         
        elif (args.aggregation == 'trim'):
            net = aggregation.trim(device, lr, grad_list, net, args.cmax)
        elif (args.aggregation == 'faba'):
            net, faba_list = aggregation.faba(device, lr, grad_list, net, args.cmax)    
            faba_client_list.append(faba_list)
        elif (args.aggregation == 'foolsgold'):
            net, fg_list = aggregation.foolsgold(device, lr, grad_list, net, args.cmax)
            fg_client_list.append(fg_list.cpu().numpy())
        elif (args.aggregation == 'median'):
            net = aggregation.median(device, lr, grad_list, net, args.cmax)

        del grad_list
        torch.cuda.empty_cache()
        
        ##Evaluate the learned model on test dataset
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_acc[rnd] = correct/total                
            print ('Iteration: %d, test_acc: %f' %(rnd, test_acc[rnd]))      
    np.save(filename+'_test_acc.npy', test_acc)
    torch.save(net.state_dict(), filename+'_model.pth')
if __name__ == "__main__":
    args = utils.parse_args()
    main(args)
