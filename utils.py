import numpy as np
import torch
import random
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50, CUB200, TinyImageNet200, OxfordFlower102
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def set_seeds(seed):
    # Reproducibility seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def update_memory(features, pseudo_labels, memory):
    
    for y in pseudo_labels.unique():

        indices = torch.where(pseudo_labels == y)[0]
        y_prototype = features[indices].mean(dim=0).unsqueeze(0)

        if memory.numel() == 0:
            memory = y_prototype
        else:
            memory = torch.cat((memory, y_prototype), dim=0)

    return memory


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


def get_transform(dset_name, resize=(224,224)):
    """ Returns the proper transformations """

    if dset_name == 'CIFAR100':
        return [transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    elif dset_name == 'CIFAR10': 
        return [transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

    elif dset_name == 'MNIST':
        return [transforms.ToTensor()]
    else:
        raise NotImplementedError



    
def forward_taskset(opt, model, taskset):

    for n, (x, y, t) in enumerate(DataLoader(taskset, batch_size=opt.batch_size)):
            
        if opt.cuda:
            x = x.to(opt.gpu)

        out = model(x).cpu()

        if n==0:
            features = out
            labels = y
        else:
            features = torch.cat((features, out), dim=0)
            labels = torch.cat((labels, y), dim=-1)
        
    return features, labels