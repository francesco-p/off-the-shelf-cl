import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50, CUB200, TinyImageNet200, OxfordFlower102
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import timm
import pickle
from sklearn.cluster import KMeans
import utils
from sklearn.metrics.cluster import adjusted_rand_score
from parse import get_opt
import seaborn as sns
import argparse




######################
### parse all opt ###
######################
opt = get_opt()


############################
### set replication seed ###
############################
utils.set_seeds(opt.seed)


########################
### Pretrained Model ###
########################
model =  timm.create_model(opt.model, pretrained=True, num_classes=0)
model.to(opt.gpu)
model.eval()

for param in model.parameters():
    param.requires_grad = False

################
### Datasets ###
################
train_dset, n_classes, data_shape = utils.get_dset(opt.data_path, opt.dataset, train=True)
tr_scenario = ClassIncremental(
    train_dset,
    increment=opt.increment,
    transformations = utils.get_transform(opt.dataset)
)

test_dataset, _, _ = utils.get_dset(opt.data_path, opt.dataset, train=False, download=False)
te_scenario = ClassIncremental(
    test_dataset,
    increment = opt.increment,
    transformations = utils.get_transform(opt.dataset)
)



###############################
### Create Prototype Memory ###
###############################
memory = torch.tensor([])
n_tasks = n_classes // opt.increment
mat = torch.zeros(n_tasks, n_tasks)

for train_task_id, tr_taskset in enumerate(tr_scenario):

    features, labels = utils.forward_taskset(opt, model, tr_taskset)
    kmeans = KMeans(n_clusters=opt.increment, random_state=opt.seed).fit(features.numpy())
    pseudo_labels = torch.tensor(kmeans.labels_, dtype=torch.long) + (train_task_id * opt.increment)
    memory = utils.update_memory(features, pseudo_labels, memory)
    ari = adjusted_rand_score(labels, pseudo_labels)
    
    ### TEST ###
    for test_task_id, te_taskset in enumerate(te_scenario):

        features, labels = utils.forward_taskset(opt, model, te_taskset)
        prediction = torch.cdist(features, memory).argmax(dim=1)
        ari = adjusted_rand_score(labels, prediction)
        print(f'{train_task_id} - {test_task_id} = {ari}')
        mat[train_task_id, test_task_id] = ari

        if test_task_id == train_task_id:
            break



sns.set()
sns.heatmap(mat, annot=True, cmap='RdBu_r', vmin=0, vmax=1)
plt.show()
######################################
### Save/Load the Prototype Memory ###
######################################
memory_fname = f'./pkls/unsup_{opt.dataset}_{opt.model}_s{opt.seed}.pt'

if opt.load:
    memory = torch.load(memory_fname)
else:
    torch.save(memory, memory_fname)


