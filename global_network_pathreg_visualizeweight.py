from matplotlib import pyplot as plt
import matplotlib
plt.switch_backend('agg')
# from IPython.display import clear_output
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint

import os
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


script__name = 'global_network_pathreg_py'

str_best_model =  'models/'   +   \
              'pathreg_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_reg_1e-05_best.h5'




# N_nodes is the number of hidden nodes per input node in the local network
DenseNodes = N_nodes*d

model = Sequential()
model.add(Dense(DenseNodes, activation='relu',use_bias=True, input_dim=d))
model.add(Dense(DenseNodes, activation='relu',use_bias=True))
model.add(Dense(1, activation='linear',use_bias=False))
model.add(Lambda(lambda x: x / d))
model.summary()



best_model = tf.keras.models.load_model(str_best_model, custom_objects={'d': d}, compile=False)
weight1 = np.transpose(best_model.layers[0].get_weights()[0])
#weight2 = np.transpose(model.layers[2].get_weights()[0])
#weight3 = np.transpose(model.layers[3].get_weights()[0])
filenameMatrix1 = 'square_model_'  + script__name +   \
              '_d_'       + str(d) + \
              '_epochs_'  + str(N_epochs) + \
              '_Nsamples_'+ str(Nsamples) + '_weight1.txt'
print(weight1.shape)
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



