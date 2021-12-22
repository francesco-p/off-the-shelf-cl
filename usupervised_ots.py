import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from continuum import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, Core50, CUB200, TinyImageNet200, OxfordFlower102
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from einops import rearrange
from torchvision.utils import save_image
from torch.nn.functional import one_hot
import timm
import pickle
from sklearn.cluster import KMeans
import utils
from sklearn.metrics.cluster import adjusted_rand_score
from parse import get_opt


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


########################
### Training Dataset ###
########################
train_dset, n_classes, data_shape = utils.get_dset(opt.data_path, opt.dataset, train=True)
tr_scenario = ClassIncremental(
    train_dset,
    increment=opt.increment,
    transformations=[
                     transforms.Resize((224,224)), #Imagenet original data size
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)


###############################
### Create Prototype Memory ###
###############################
memory = torch.tensor([])

if not opt.load:
    with torch.no_grad():
        for task_id, tr_taskset in enumerate(tr_scenario):

            for n, (x, y, t) in enumerate(DataLoader(tr_taskset, batch_size=opt.batch_size)):
                
                if opt.cuda:
                    x = x.to(opt.gpu)

                out = model(x).cpu()
                
                if n==0:
                    features = out
                    labels = y
                else:
                    features = torch.cat((features, out), dim=0)
                    labels = torch.cat((labels, y), dim=-1)

            kmeans = KMeans(n_clusters=opt.increment, random_state=opt.seed).fit(features.numpy())
            pseudo_labels = torch.tensor(kmeans.labels_, dtype=torch.long) + (task_id * opt.increment)
            
            memory = utils.update_memory(features, pseudo_labels, memory)
            print(adjusted_rand_score(labels, pseudo_labels))


######################################
### Save/Load the Prototype Memory ###
######################################
memory_fname = f'./pkls/unsup_{opt.dataset}_{opt.model}_s{opt.seed}.pkl'

if opt.load:
    torch.load(memory_fname)
else:
    torch.save(memory, memory_fname)




############
### Test ###
############
test_dataset, _, _ = utils.get_dset(opt.data_path, opt.dataset, train=False, download=False)
te_scenario = ClassIncremental(
    test_dataset,
    increment=opt.increment,
    transformations=[
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

n_tasks = n_classes // opt.increment

mat = torch.zeros(n_tasks, n_tasks)
for task_id, te_taskset in enumerate(te_scenario):
    print(f"test\n{task_id}")

    for n, (x, y, t) in enumerate(DataLoader(te_taskset, batch_size=1)):
        
        if opt.cuda:
            x = x.to(opt.gpu)

        out = model(x).cpu()

        if n==0:
            features = out
            labels = y
        else:
            features = torch.cat((features, out), dim=0)
            labels = torch.cat((labels, y), dim=-1)
