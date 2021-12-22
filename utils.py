import numpy as np
import torch
import random
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50, CUB200, TinyImageNet200, OxfordFlower102


def set_seeds(seed):
    # Reproducibility seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_prototype(x, model, bs, cuda=True):
    device = CUDA_DEVICE if cuda else 'cpu'

    n = x.shape[0]
    if bs > n:
        bs = n
    iters = n // bs
    features = None
    for i in range(iters):
        init = i * bs
        end = init + bs
        out = model(x[init : end].to(device)).cpu()
        if features == None:
            features = out
        else:
            features = torch.cat((features, out), dim=0)

    if end < n:
        out = model(x[end:].to(device)).cpu()
        features = torch.cat((features, out), dim=0)

    return features.mean(dim=0)


def get_prototype(x, pretr_models, test=False, cuda=True):
    """ Forward data to the models and create a prototype (mean of the features) """
    with torch.no_grad():
        device = CUDA_DEVICE if cuda else 'cpu'
        prototype = []
        for model in pretr_models:
            model.eval()
            model.to(device)
            x = x.to(device)

            if test:
                prototype.append(model(x).mean(dim=0).view(-1).cpu())
            else:
                prototype.append(compute_prototype(x, model, 64, cuda))

            model.to('cpu')
            x.to('cpu')
    
    return prototype


def prototype_distance(prototype1, prototype2, cuda=True):
    """ Forward data to the models and create a prototype """
    device = CUDA_DEVICE if cuda else 'cpu'

    tot_dist = 0.
    for s1, s2 in zip(prototype1, prototype2):
        s1 = s1.to(device)
        s2 = s2.to(device)
        #tot_dist += torch.dist(s1, s2, p=2)
        tot_dist += 1-torch.nn.CosineSimilarity()(s1.unsqueeze(0), s2.unsqueeze(0))
    return tot_dist.item()
    

def get_closest_prototype(memory, prototype2, cuda=True):
    """ compute L2 of a prototype vs all the elements in the memory"""
    min_k = -1
    min_dist = 10e+10
    for i, k in enumerate(memory.keys()):

        prototype1 = memory[k]
        dist = prototype_distance(prototype1, prototype2, cuda)
        #print(k, dist)
        if dist < min_dist:
            min_dist = dist
            min_k = k
    return min_k




def update_memory(features, pseudo_labels, memory):
    
    for y in pseudo_labels.unique():

        indices = torch.where(pseudo_labels == y)[0]
        y_prototype = features[indices].mean(dim=0).unsqueeze(0)

        if memory.numel() == 0:
            memory = y_prototype
        else:
            memory = torch.cat((memory, y_prototype), dim=0)

    return memory

"""
def update_memory(features, pseudo_labels, memory):

    for y in pseudo_labels.unique():
        
        idxs = torch.where(pseudo_labels == y)[0]
        
        memory[y.item()] = features[idxs].mean(dim=0)
    
    return
"""


def get_dset(data_path, dset_name, train, download=True):
    """ Returns a tuple with the dataset object, the total number of 
    classes and the shape of the original data
    """
    if dset_name == 'CIFAR100':
        return (CIFAR100(data_path=data_path, download=download, train=train), 100, (3, 32, 32))
    elif dset_name == 'CIFAR10': 
        return (CIFAR10(data_path=data_path, download=download, train=train), 10, (3, 32, 32))
    elif dset_name == 'MNIST':
        return (MNIST(data_path=data_path, download=download, train=train), 10, (1, 28, 28))
    else:
        raise NotImplementedError