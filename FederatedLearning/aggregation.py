import torch
import numpy as np
import torch.nn as nn
import time
import pdb

##FEDSGD - weighted mean aggregation weighed by their data size
def FEDSGD(device, lr, grad_list, net, wts):
    start = time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), wts.reshape(-1,1))
    
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params
    #print (time.time()-start)
    return net


##FoolsGold
def foolsgold(device, lr, grad_list, net):
    start = time.time()    
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    num_workers = len(param_list)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(device)
    cs = torch.zeros((num_workers, num_workers)).to(device)
    for i in range (num_workers):
        for j in range (i):
            ## compute cosine similarity
            cs[i,j] = cos(param_list[i], param_list[j])
            cs[j,i] = cs[i,j]
    ###The foolsgold algorithm implemented below
    v = torch.zeros(num_workers).to(device)        
    for i in range (num_workers):
        v[i] = torch.max(cs[i])
      
    alpha = torch.zeros(num_workers).to(device)
    for i in range (num_workers):
        for j in range (num_workers):
            if (v[j] > v[i]):
                cs[i,j] = cs[i,j]*v[i]/v[j]
        alpha[i] = 1 - torch.max(cs[i])
    
    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/(torch.max(alpha))
    alpha[alpha == 1] = 0.99
    alpha = torch.log(alpha/(1-alpha)) + 0.5
    alpha[(torch.isinf(alpha) + (alpha > 1))] = 1
    alpha[alpha < 0] = 0
    alpha = alpha/torch.sum(alpha).item()
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), alpha.reshape(-1,1))
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params
    #print(time.time()-start)
    return net, alpha

#FABA 
def faba(device, lr, grad_list, net, cmax):
    start = time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    faba_client_list = np.ones(len(param_list)) #contains the current benign clients
    dist = np.zeros(len(param_list))
    G0 = torch.mean(param_list, dim=0)
    for i in range(cmax):
        for j in range(len(param_list)):
            if faba_client_list[j]:
                dist[j] = torch.norm(G0-param_list[j]).item()      
        outlier = int(np.argmax(dist))
        faba_client_list[outlier] = 0 #outlier removed as suspected 
        dist[outlier] = 0
        G0 = (G0*(len(param_list)-i) - param_list[outlier])/(len(param_list)-i-1) #mean recomputed

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
               param[1].data += G0[idx:(idx+param[1].nelement())].reshape(param[1].shape)
               idx += param[1].nelement()

    del param_list
    #print(time.time()-start)
    return net, faba_client_list  

#KRUM aggregation
def krum(device, lr, grad_list, net, cmax):
    start = time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    k = len(param_list)-cmax-2
    #Computing distance between every pair of clients
    dist = torch.zeros((len(param_list), len(param_list))).to(device)
    for i in range(len(param_list)):
        for j in range(i):
            dist[i][j] = torch.norm(param_list[i]-param_list[j])
            dist[j][i] = dist[i][j]       
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += param_list[model_selected][idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list
    #print (time.time()-start)
    return net   


#TRIMMED MEAN
def trim(device, lr, grad_list, net, cmax): 
    start=time.time()
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    #Sorting every parameter
    sorted_array = torch.sort(param_list, axis=0)
    #Trimmin the ends
    trimmed = torch.mean(sorted_array[0][cmax:len(param_list)-cmax,:], axis=0)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += trimmed[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
                
    del param_list, sorted_array, trimmed
    #print(time.time()-start)
    return net  

#MEDIAN aggregation
def median(device, lr, grad_list, net, cmax):
    param_list = torch.stack([(torch.cat([xx.reshape((-1)) for xx in x], dim=0)).squeeze(0) for x in grad_list])
    sorted_array = torch.sort(param_list, axis=0)
    if (len(param_list)%2 == 1):
        med = sorted_array[0][int(len(param_list)/2),:]
    else:
        med = (sorted_array[0][int(len(param_list)/2)-1,:] + sorted_array[0][int(len(param_list)/2),:])/2

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += med[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
    del param_list, sorted_array
    return net
    
