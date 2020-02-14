# The global network

import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Input, Lambda
from keras.utils import np_utils
import keras.optimizers as optimizers
from keras import regularizers
from keras.engine.topology import Layer
from matplotlib import pyplot as plt
import matplotlib
plt.switch_backend('agg')
# from IPython.display import clear_output
import math
import keras.backend as K
from keras.utils.generic_utils import get_custom_objects
import scipy.io as sio
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

import os
import timeit
import argparse
import h5py


# parameters
d = 60
N_epochs = 1000
N_nodes = 50
lr = 0.01
#BATCH_SIZE = args.batch_size
Nsamples = 100000
data_folder = 'data/'
models_folder = 'models/' # folder to save the models (weights)


script__name = 'global_network_py'

str_best_model =  'models/'   +   \
              'square_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_best.h5'




# N_nodes is the number of hidden nodes per input node in the local network
DenseNodes = N_nodes*d

layerInput       = Input(shape=(d,))
layerHidden1     = Dense(DenseNodes, activation='relu', use_bias=True)(layerInput)
layerHidden2     = Dense(DenseNodes, activation='relu', use_bias=True)(layerHidden1)
layerOutput_pre  = Dense(1, activation='linear', use_bias=False)(layerHidden2)
layerOutput      = Lambda(lambda x: x / d)(layerOutput_pre)
model            = Model(inputs=layerInput, outputs=layerOutput)

model.summary()


adam = optimizers.Adam(lr=lr)
model.compile(loss='mean_squared_error', optimizer=adam)

model = load_model(str_best_model, custom_objects={'d': d})
weight1 = np.transpose(model.layers[1].get_weights()[0])
#weight2 = np.transpose(model.layers[2].get_weights()[0])
#weight3 = np.transpose(model.layers[3].get_weights()[0])
filenameMatrix1 = 'square_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_weight1.txt'
np.savetxt(filenameMatrix1, weight1)

filenameMatrix1 = 'square_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_weight1.png'
matrix = np.zeros((d*5,d))
for p in range(d*5):
 for q in range(d):
   matrix[p,q] = np.max(np.abs(weight1[10*p:10*(p+1), q]))
im = plt.imshow(matrix, norm=matplotlib.colors.Normalize())
plt.colorbar(im)              
plt.savefig(filenameMatrix1)
plt.clf()



