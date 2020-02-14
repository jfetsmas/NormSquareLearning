# The global network

import numpy as np
from matplotlib import pyplot as plt

import math

import os
import timeit
import argparse
import h5py


# parameters
d = 32
N_epochs = 1000
N_nodes = 50
lr = 0.01
#BATCH_SIZE = args.batch_size
Nsamples = 100000
data_folder = 'data/'
models_folder = 'models/' # folder to save the models (weights)


script__name = 'global_network_L1_py'

filenameMatrix1 = 'square_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_1e-08_weight1.txt'



weight1 = np.loadtxt(filenameMatrix1)
matrix = np.zeros((d*5,d))
for p in range(d*5):
  for q in range(d):
    matrix[p,q] = np.max(weight1[10*p:10*(p+1), q])
im = plt.imshow(matrix, norm=matplotlib.colors.Normalize())
plt.colorbar(im)
plt.show()


