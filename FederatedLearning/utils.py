import argparse
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import nets
import custom_dataset

## Read the command line argumentacsA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--num_workers", help="# workers", default=10, type=int)
    parser.add_argument("--cmax", help="# suspected workers", default=0, type=int)
    parser.add_argument("--num_rounds", help="# rounds", default=50, type=int)
    parser.add_argument("--num_iters", help="# local iterations", default=1, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--aggregation", help="aggregation rule", default='fedsgd', type=str)
    parser.add_argument("--exp", help="Experiment name", default='', type=str)
    return parser.parse_args()

#Learning rate scheduler used to train CIFAR-10
def get_lr(rnd, num_rounds, lr):

    mu = num_rounds/4
    sigma = num_rounds/4
    max_lr = lr
    if (rnd < num_rounds/4):
        return max_lr*(1-np.exp(-25*(rnd/num_rounds)))
    else:
        return max_lr*np.exp(-0.5*(((rnd-mu)/sigma)**2))

def load_data(dataset_name):
    
    ###Load datasets
    if (dataset_name == 'mnist'):
        batch_size = 32
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]) 
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset        
        num_inputs = 28 * 28
        num_outputs = 10
        lr = 0.01
        net = nets.DNN()

    elif dataset_name == 'cifar10':
        batch_size = 128
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
        num_inputs = 32*32*3
        num_outputs = 10
        lr = 0.1
        net = nets.ResNet18()

    elif dataset_name == 'pathology':
        train = True

        batch_size = 32
        # Add more transformations if needed
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1/255, 1/255, 1/255)),
        ])
  
        train_csv = r'C:\Users\hp\Documents\A1 CSDS Assignments\btp\NonFederated ML model\train_labels.csv'
        train_root = r'D:\btp data\train'

        subset_size = 100000  # Number of random samples to include for training
        if train:
            train_dataset = custom_dataset.CustomPathologyDataset(train_csv, train_root, transform=transform, train=train, subset_size=subset_size)
        else:
            train_dataset = custom_dataset.CustomPathologyDataset(train_csv, train_root, transform=transform, train=train)

        train_data = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = custom_dataset.CustomPathologyDataset(train_csv, train_root, transform=transform, train=False)
        test_data= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        num_inputs = 96 * 96 * 3  # Update according to your image dimensions and channels
        num_outputs = 2  # Assuming binary classification (0 or 1)
        lr = 0.01  # Adjust the learning rate as needed

        # Assuming you have a neural network model for pathology
        net = nets.YourPathologyModel()  # Replace with the actual model class

    else:
        sys.exit('Not Implemented Dataset!')

    return train_data, test_data, net, num_inputs, num_outputs, lr, batch_size

def distribute_data(train_data, bias_weight, num_workers, num_outputs, device):

    ####Distribute data samples according to a given non-IID bias
    other_group_size = (1-bias_weight) / (num_outputs-1)
    worker_per_group = num_workers / (num_outputs)
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)] 
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1-bias_weight) / (num_outputs-1) + bias_weight
            lower_bound = (y.item()) * (1-bias_weight) / (num_outputs-1)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()

            # assign a data point to a worker
            rd = np.random.random_sample()
            selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
            if (bias_weight == 0): selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x.to(device))
            each_worker_label[selected_worker].append(y.to(device))

    # concatenate the data for each worker
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data] 
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]
    
    # random shuffle the workers
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    #define weights proportional to data size of a client for FEDSGD
    wts = torch.zeros(len(each_worker_data)).to(device)
    for i in range(len(each_worker_data)):
        wts[i] = len(each_worker_data[i])
    wts = wts/torch.sum(wts)

    return each_worker_data, each_worker_label, wts
