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
import random
import timm
import pickle
from parse import get_opt
import utils
import time


# parse all opt
opt = get_opt()

# set replication seed
utils.set_seeds(opt.seed)

# Pretrained fext
model =  timm.create_model(opt.model, pretrained=True, num_classes=0)
model.to(opt.gpu)
model.eval()

for param in model.parameters():
    param.requires_grad = False

################
### Datasets ###
################
train_dset, n_classes, data_shape = utils.get_dset(opt.data_path, opt.dataset, download=True, train=True)
tr_tasks = ClassIncremental(
    train_dset,
    increment=1,
    transformations = utils.get_transform(opt.dataset, resize=(224,224)) #pay attention to the transform, for vits you need 224,224
)


test_dataset, _, _ = utils.get_dset(opt.data_path, opt.dataset, train=False, download=True)
te_tasks = ClassIncremental(
    test_dataset,
    increment = 1,
    transformations = utils.get_transform(opt.dataset, resize=(224,224))
)

################################################################################


# Train
start = time.time()
if not opt.load:
    for train_task_id, class_group in enumerate(tr_tasks):
        print('train', train_task_id)
        features_group, _ = utils.forward_taskset(opt, model, class_group)
        class_prototype = features_group.mean(dim=0).unsqueeze(0)
        if train_task_id == 0:
            memory_bank = class_prototype
        else:
            memory_bank = torch.cat((memory_bank, class_prototype), dim=0)
else:
    memory_bank = torch.load(f'./memories/{opt.dataset}_{opt.model}.pt')
end = time.time()

torch.save(memory_bank, f'./memories/{opt.dataset}_{opt.model}.pt')


# Test
x = []
y = []
tot_n = 0
tot_hits = 0
for test_task_id, class_group in enumerate(te_tasks):
    print('test', test_task_id)
    features, labels = utils.forward_taskset(opt, model, class_group)
    # we need to compute pairwise cosine similarity
    prediction = torch.cdist(features, memory_bank).argmin(dim=1)
    n = labels.shape[0]
    hits = (prediction == labels).sum().item()

    x.append(n)
    y.append(hits)

    tot_n += n
    tot_hits += hits
    print(n, hits, tot_n, tot_hits, tot_hits / tot_n)

print('task examples', 'task hits', 'total examples', 'total hits', 'total accuracy')

elapsed = end - start
acc = tot_hits / tot_n
with open('results.csv', 'a') as f:
    f.write(f'{opt.dataset},{opt.model},{tot_n},{tot_hits},{acc},{elapsed}\n')

#plt.bar(range(n_classes), y)
#plt.show()
#import ipdb; ipdb.set_trace()
#tot_dist += 1-torch.nn.CosineSimilarity()(s1.unsqueeze(0), s2.unsqueeze(0))
