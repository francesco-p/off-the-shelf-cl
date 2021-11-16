import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from einops import rearrange
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import random
from models import ELM
import timm
import pickle




def get_prototype(x, pretr_models, cuda=True):
    """ Forward data to the models and create a prototype (mean of the features) """
    with torch.no_grad():
        device = 'cuda' if cuda else 'cpu'
        
        prototype = []
        for model in pretr_models:
            model.eval()
            model.to(device)
            x = x.to(device)
            prototype.append(model.forward_features(x).mean(dim=0).view(-1).cpu())
            model.to('cpu')
            x.to('cpu')
    
    return prototype


def prototype_distance(prototype1, prototype2, cuda=True):
    """ Forward data to the models and create a prototype """
    device = 'cuda' if cuda else 'cpu'

    tot_dist = 0.
    for s1, s2 in zip(prototype1, prototype2):
        s1 = s1.to(device)
        s2 = s2.to(device)
        tot_dist += torch.dist(s1, s2, p=2)
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

###############################################################################
###############################################################################
###############################################################################

DATA_PATH = "./data"
SEED = 0
CUDA = True
MEMORY_FNAME = './pkls/memory_resnet50.pkl'

# Reproducibility seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

########################
### Pretrained Model ###
########################

# You can combine more than one feature extractor
# all models pretrined in imagenet available here: timm.list_models()

names = ['resnet50'] 
models = []
for name in names:
    model =  timm.create_model(name, pretrained=True)
    model.to('cpu')
    models.append(model) 

# This is the pretrained weights of the paper 'Resnets strikes back'
# model.load_state_dict(torch.load('pretrained_models/resnet50_a1_0-14fe96d1.pth'))


###############################
### Create Prototype Memory ###
###############################

# Task split in each class (corresponds to increment==1)
tr_scenario = ClassIncremental(
    CIFAR100(data_path=DATA_PATH, download=True, train=True),
    increment=1,
    transformations=[
                     transforms.Resize(224), #Imagenet original data size
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

memory = {}
for task_id, tr_taskset in enumerate(tr_scenario):
    print(f"train\n{ task_id :-^50}")
    for x, y, t in DataLoader(tr_taskset, batch_size=len(tr_taskset)):
        break

    memory[task_id] = get_prototype(x, models, cuda=CUDA)


#######################################
### Save/Load the Prototype Memory  ###
#######################################

with open(MEMORY_FNAME, 'wb') as f:
    pickle.dump(memory, f)

#with open(MEMORY_FNAME, 'rb') as f:
#    memory = pickle.loads(f.read())


############
### Test ###
############

te_scenario = ClassIncremental(
    CIFAR100(data_path=DATA_PATH, download=True, train=False),
    increment=10,
    transformations=[
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

tot = 0
n = 0

for task_id, te_taskset in enumerate(te_scenario):
    print(f"test\n{task_id}")

    for x, y, t in DataLoader(te_taskset, batch_size=1):
        n+= 1
        x_sign = get_prototype(x, models, cuda=False)

        pred = get_closest_prototype(memory, x_sign, cuda=True)
        
        if pred == y.item():
            tot += 1
        print(tot, n, pred, y)

# correct predictions, total elements
print(tot, n)
