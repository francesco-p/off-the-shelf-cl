import torch
from torch.utils.tensorboard import SummaryWriter
import glob

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#sw = SummaryWriter()
files = sorted(glob.glob('./memories/*'))
for f in files:
    mat = torch.load(f)
    f = '_'.join(f[11:-3].split('_')[:3])
    lbls = [x for x in range(len(mat))]
    #sw.add_embedding(mat, lbls, tag=f)

    dist_mat = torch.cdist(mat, mat)

    print(f, torch.frobenius_norm(mat), dist_mat.sum())

    #fig = plt.figure()
    #plt.imshow(dist_mat)
    #plt.savefig(f'./imgs/{f}.png')
    #plt.close()



    #fig = plt.figure()
    #g = nx.from_numpy_array(dist_mat.numpy())
    #edges, weights = zip(*nx.get_edge_attributes(g,'weight').items())
    #weights = np.array(weights)
    #weights /= weights.max()
    #nx.draw(g, node_color='r', with_labels=True, edge_color=weights, width=weights, cmap=plt.cm.jet)#, pos=nx.circular_layout(g))

    #plt.savefig(f'./imgs/{f}.png')
    #plt.close()



