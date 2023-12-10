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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import tabulate

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
    train_loss = np.empty(num_rounds)

    precision_list = np.empty(num_rounds)
    recall_list = np.empty(num_rounds)
    f1_list = np.empty(num_rounds)

    # Add metrics tracking variables
    all_predictions = []
    all_labels = []

    batch_idx = np.zeros(num_workers)
    faba_client_list = []
    fg_client_list = []
    weight = torch.ones(num_workers)

    table = [["Global Model Update Round", "Train Loss", "Test Accuracy", "Precision", "Recall", "F1 Score"]]

    for rnd in range(num_rounds):
        grad_list = []
        if (args.dataset == 'cifar10'):
            lr = utils.get_lr(rnd, num_rounds, lr_init)
        else:
            lr = lr_init
        for worker in range(num_workers):
            net_local = deepcopy(net) 
            net_local.train()
            optimizer = optim.SGD(net_local.parameters(), lr=lr)

            for local_iter in range(num_iters):
                optimizer.zero_grad()
                # sample local dataset in a round-robin manner
                if (batch_idx[worker]+batch_size < distributed_data[worker].shape[0]):
                    minibatch = np.asarray(list(range(int(batch_idx[worker]), int(batch_idx[worker])+batch_size)))
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

            # del net_local, output, loss
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

        ## Evaluate the learned model on test dataset
        correct = 0
        total = 0
        predictions = []
        labels = []

        with torch.no_grad():
            for data in test_data:
                images, labels_batch = data
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch.to(device)).sum().item()

                # Store predictions and labels for further metrics
                predictions.extend(predicted.cpu().numpy())
                labels.extend(labels_batch.cpu().numpy())

                # Store predictions and labels for ROC curve
                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())


        test_acc[rnd] = correct / total
        train_loss[rnd] = loss.item()  # Add training loss

        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')

        precision_list[rnd] = precision
        recall_list[rnd] = recall

        confusion_mat = confusion_matrix(labels, predictions)
        recall_list[rnd] = recall
        f1_list[rnd] = f1

        confusion_mat = confusion_matrix(labels, predictions)

        # Display metrics
        print(f"Iteration: {rnd}, Train Loss: {train_loss[rnd]}, Test Accuracy: {test_acc[rnd]}")
        # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        # print(f"Confusion Matrix:\n{confusion_mat}")
        row = [rnd, train_loss[rnd], test_acc[rnd], precision, recall, f1]
        table.append(row)

        # Save metrics to files
        np.save(filename + '_test_acc.npy', test_acc)
        np.save(filename + '_train_loss.npy', train_loss)
        np.save(filename + '_precision.npy', precision_list)
        np.save(filename + '_recall.npy', recall_list)
        np.save(filename + '_f1.npy', f1_list)
        torch.save(net.state_dict(), filename + '_model.pth')
        # print(f"Metrics saved to: {filename + '_test_acc.npy'}, {filename + '_train_loss.npy'}, {filename + '_precision.npy'}, {filename + '_recall.npy'}, {filename + '_f1.npy'}, {filename + '_model.pth'}")
        if rnd == num_rounds - 1:  # Check if it's the last iteration
          # Print confusion matrix only for the last iteration
          print(f"Confusion Matrix (Last Iteration):\n{confusion_mat}")
    
    print(f"Metrics saved to: {filename + '_test_acc.npy'}, {filename + '_train_loss.npy'}, {filename + '_precision.npy'}, {filename + '_recall.npy'}, {filename + '_f1.npy'}, {filename + '_model.pth'}")
    # Print the table
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, np.array(all_predictions)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()

if __name__ == "__main__":
    args = utils.parse_args()
    main(args)
