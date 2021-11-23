import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50, CUB200
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from einops import rearrange
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import random
import timm
import pickle

CUDA_DEVICE = 'cuda:2'

def compute_prototype(x, model, bs, cuda=True):
    device = CUDA_DEVICE if cuda else 'cpu'

    n = x.shape[0]
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
                prototype.append(compute_prototype(x, model, 256))

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

###############################################################################
###############################################################################
###############################################################################

DATA_PATH = "~/data"
SEED = 0
CUDA = True
MEMORY_FNAME = './pkls/CUB200_memory_resnet_50.pkl'
MODEL = 'resnet50'

# Reproducibility seeds
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

########################
### Pretrained Model ###
########################

# You can combine more than one feature extractor
# all models pretrined in imagenet available here: timm.list_models()

names = [MODEL] 
models = []
for name in names:
    model =  timm.create_model(name, pretrained=True, num_classes=0)
    #model =  torchvision.models.resnet152(pretrained=True)
    model.to('cpu')
    models.append(model) 

# This is the pretrained weights of the paper 'Resnets strikes back'
#model.load_state_dict(torch.load('model_300.pt')['model_state_dict'])


###############################
### Create Prototype Memory ###
###############################

memory = {}
# Task split in each class (corresponds to increment==1)
tr_scenario = ClassIncremental(
    CUB200(data_path=DATA_PATH, download=True, train=True),
    increment=1,
    transformations=[
                     transforms.Resize(224), #Imagenet original data size
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

for task_id, tr_taskset in enumerate(tr_scenario):
    #if task_id < 50:
        #continue
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
    CUB200(data_path=DATA_PATH, download=True, train=False),
    increment=1,
    transformations=[
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

tot = 0
n = 0

rel_mat = torch.zeros(100, 100)
for task_id, te_taskset in enumerate(te_scenario):
    #if task_id < 50:
        #continue
    print(f"test\n{task_id}")

    for x, y, t in DataLoader(te_taskset, batch_size=1):
        n+= 1
        x_sign = get_prototype(x, models, test=True, cuda=CUDA)

        pred = get_closest_prototype(memory, x_sign, cuda=True)
        
        if pred == y.item():
            tot += 1
        print(tot, n, pred, y.item(), f"{(tot/n):.3f}")
        rel_mat[y.item(), pred] += 1

# correct predictions, total elements
print('CUB200-RESNET50',tot, n)
plt.imshow(rel_mat)
plt.savefig(f"./pngs/CUB_200_{MODEL}.png")
#import ipdb; ipdb.set_trace()
