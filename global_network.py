# The global network

import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Input, Lambda
from keras.utils import np_utils
import keras.optimizers as optimizers
from keras import regularizers
from keras.engine.topology import Layer
# from matplotlib import pyplot as plt
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

parser = argparse.ArgumentParser(description='NormLearning')
parser.add_argument('--epoch', type=int, default=1000, metavar='N',
                    help='input number of epochs for training (default: 1000)')
parser.add_argument('--d', type=int, default='4', metavar='N',
                    help='input dimension (default: 4)')
parser.add_argument('--Nsamples', type=int, default=1000, metavar='N',
                    help='number of training samples (default 1000')
args = parser.parse_args()
# parameters
d = args.d
N_epochs = args.epoch
N_nodes = 50
lr = 0.01
#BATCH_SIZE = args.batch_size
Nsamples = args.Nsamples
data_folder = 'data/'
log_folder = 'logs/'
models_folder = 'models/' # folder to save the models (weights)
image_folder = 'images/'



# this is generic for al the test_files
dir__name =  os.getcwd()
script__name = os.path.basename(__file__)
script__name = script__name.replace(".", "_")

str_best_model =  'models/'   +   \
              'cosine_model_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'


filenameIpt = data_folder + 'cosine_x_'  + str(args.d) + '_Nsample_1000000.dat'
filenameOpt = data_folder + 'cosine_y_' + str(args.d) + '_Nsample_1000000.dat'

xraw_huge  = np.loadtxt(filenameIpt, delimiter=None, usecols=range(d))
yraw_huge  = np.loadtxt(filenameOpt, delimiter=None, usecols=range(1))

xraw = xraw_huge[0:Nsamples,:]
yraw = yraw_huge[0:Nsamples]/d

n_train = int(0.8*Nsamples)
n_test = int(0.2*Nsamples)

BATCH_SIZE = n_train//100


# pre-treating the data
# sorting procedure according to distance to origin, works in N dimension
xraw2 = np.zeros(xraw.shape)
for i in range(0,Nsamples):
    tmp = np.reshape(xraw[i,:],(d,1))
    norm = np.linalg.norm(tmp, axis=1)
    tmp2 = tmp[norm.argsort()]
    tmp3 = tmp2.ravel()
    xraw2[i,:] = tmp3

ytest = yraw[0:n_test]
ytrain = yraw[Nsamples-n_train:Nsamples]

xtest = xraw2[0:n_test,:]
xtrain = xraw2[Nsamples-n_train:Nsamples,:]


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
checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=False, \
                               mode='auto', period=1)
history = model.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=0,callbacks=[checkpointer])
model = load_model(str_best_model, custom_objects={'d': d})


yNNT = model.predict(xtrain[0:8*(n_train//10)+1,:])
errTrain = np.linalg.norm(yNNT.reshape(yNNT.size)-ytrain[0:8*(n_train//10)+1])**2/(8*(n_train//10))

yNNV = model.predict(xtrain[8*(n_train//10):n_train,:])
errVal = np.linalg.norm(yNNV.reshape(yNNV.size)-ytrain[8*(n_train//10):n_train])**2/(n_train-8*(n_train//10))

yNN = model.predict(xtest)
errTest = np.linalg.norm(yNN.reshape(yNN.size)-ytest)**2/n_test

print('error with best model')
print("mean squared error of train data in NN: %.1e" % (errTrain))
print("mean squared error of validation data in NN: %.1e" %  (errVal))
print("mean squared error of test data in NN: %.1e" %  (errTest))

# Change the output filename here
log_os = open('summary_global_network_cosine_0207.txt', "a")
log_os.write('%d\t' % (args.d))
log_os.write('%d\t%d\t' % (N_epochs, Nsamples))
log_os.write('%d\t' % (model.count_params()))
log_os.write('%.3e\t%.3e\t%.3e\t' % (errTrain, errVal, errTest))
log_os.write('\n')
log_os.close()
