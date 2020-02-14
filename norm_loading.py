# version 20190926 with weights loaded from cnn to dense
import numpy as np
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Conv1D, Input, Lambda
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
parser.add_argument('--epoch', type=int, default=10, metavar='N',
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
decay = 0.3
#BATCH_SIZE = args.batch_size
Nsamples = args.Nsamples
data_folder = 'data/'
log_folder = 'logs/'
models_folder = 'models/' # folder to save the models (weights)
image_folder = 'images/'

dir__name =  os.getcwd()
script__name = os.path.basename(__file__)
script__name = script__name.replace(".", "_")

str_best_model =  'models/'   +   \
              'compact_model_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'

filenameIpt = data_folder + 'square_x_'  + str(args.d) + '_Nsample_1000000.dat'
filenameOpt = data_folder + 'square_y_' + str(args.d) + '_Nsample_1000000.dat'

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
model_dense      = Model(inputs=layerInput, outputs=layerOutput)

model_dense.summary()


adam = optimizers.Adam(lr=lr, decay=decay)
model_dense.compile(loss='mean_squared_error', optimizer=adam)
checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=True, \
                               mode='auto', period=1)
history_dense = model_dense.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=0,callbacks=[checkpointer])
model_dense.load_weights(filepath=str_best_model)
partly_trained_dense =  'models/'   +   \
              'partly_trained_dense_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'
model_dense.save(filepath=partly_trained_dense)









xtest  = np.reshape(xtest, (xtest.shape[0],xtest.shape[1], 1))
xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1], 1))

alpha = N_nodes
layerInput   = Input(shape=(d,1))
layerHidden1 = Conv1D(alpha, 1, strides=1, activation='relu', use_bias=True)(layerInput)
layerHidden2 = Conv1D(alpha, 1, strides=1, activation='relu', use_bias=True)(layerHidden1)
layerOutput  = Conv1D(1, 1, strides=1, activation='linear', use_bias=False)(layerHidden2)
Sum          = Lambda(lambda x: K.sum(x, axis=1), name='sum')
layerSum_pre = Sum(layerOutput)
layerSum     = Lambda(lambda x: x / d)(layerSum_pre)
model        = Model(inputs=layerInput, outputs=layerSum)


str_best_model =  'models/'   +   \
              'weights_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'

adam = optimizers.Adam(lr=lr, decay=decay)
model.compile(loss='mean_squared_error', optimizer=adam)
checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=True, \
                               mode='auto', period=1)
history_cnn = model.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=1,callbacks=[checkpointer])
model.load_weights(filepath=str_best_model)

partly_trained_cnn =  'models/'   +   \
              'partly_trained_cnn_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'
model.save(filepath=partly_trained_cnn)

yNNT = model.predict(xtrain[0:8*(n_train//10)+1,:])
errTrain = np.linalg.norm(yNNT.reshape(yNNT.size)-ytrain[0:8*(n_train//10)+1])**2/(8*(n_train//10))

yNNV = model.predict(xtrain[8*(n_train//10):n_train,:])
errVal = np.linalg.norm(yNNV.reshape(yNNV.size)-ytrain[8*(n_train//10):n_train])**2/(n_train-8*(n_train//10))

yNN = model.predict(xtest)
errTest = np.linalg.norm(yNN.reshape(yNN.size)-ytest)**2/n_test










weight = np.reshape(model.layers[1].get_weights()[0],(1,N_nodes))
bias = model.layers[1].get_weights()[1]
new_bias = np.tile(bias,d)
new_weight = np.kron(np.identity(d),weight)
model_dense.layers[1].set_weights([new_weight,new_bias])

weight = np.reshape(model.layers[2].get_weights()[0],(N_nodes,N_nodes))
bias = model.layers[2].get_weights()[1]
new_bias = np.tile(bias,d)
new_weight = np.kron(np.identity(d),weight)
model_dense.layers[2].set_weights([new_weight,new_bias])
    
weight = np.reshape(model.layers[3].get_weights()[0],(N_nodes,1))
new_weight = np.tile(weight,(d,1))
model_dense.layers[3].set_weights([new_weight])



yNNdense = model_dense.predict(np.reshape(xtest,(n_test,d)))
errTest_check = np.linalg.norm(yNNdense.reshape(yNNdense.size)-ytest)**2/n_test
print(errTest_check)
print(errTest)




xtest  = np.reshape(xtest, (xtest.shape[0],xtest.shape[1]))
xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1]))
str_best_model =  'models/'   +   \
              'weightsdense2_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'


partly_trained_denseCNN =  'models/'   +   \
              'partly_trained_denseCNN_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + '_best.h5'
model_dense.save(filepath=partly_trained_denseCNN)

# del model_dense
# del model
# from keras.models import load_model
model_dense2 = load_model(filepath=partly_trained_denseCNN, custom_objects={'d': d})
yNN = model_dense2.predict(xtest)
errTest = np.linalg.norm(yNN.reshape(yNN.size)-ytest)**2/n_test
print(errTest)

checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=True, \
                               mode='auto', period=1)
history_dense2 = model_dense2.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=1,callbacks=[checkpointer])
model_dense2.load_weights(filepath=str_best_model)





model_dense3 = load_model(filepath=partly_trained_dense, custom_objects={'d': d})
checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=True, \
                               mode='auto', period=1)
history_dense3 = model_dense3.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=1,callbacks=[checkpointer])


xtest  = np.reshape(xtest, (xtest.shape[0],xtest.shape[1], 1))
xtrain = np.reshape(xtrain, (xtrain.shape[0],xtrain.shape[1], 1))

model_cnn2 = load_model(filepath=partly_trained_cnn, custom_objects={'d': d})
checkpointer = ModelCheckpoint(filepath=str_best_model, \
                               verbose=1, save_best_only=True, \
                               monitor='val_loss', save_weights_only=True, \
                               mode='auto', period=1)
history_cnn2 = model_cnn2.fit(xtrain, ytrain,validation_split=0.2, batch_size=BATCH_SIZE, \
                    epochs=N_epochs, verbose=1,callbacks=[checkpointer])




log_outputfilename = 'norm_loading_decay_0.3_d_'+str(d)+'_epoch_'+str(N_epochs)+'.txt'
log_os = open(log_outputfilename, "w")
log_os.write('d\t%d\n' % (args.d))
log_os.write('Epoch\t%d\n' % (N_epochs))
#log_os.write('%d\t%d\t%d\t' % (n_train, n_test, model.count_params()))
log_os.write('history_GN_loss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense.history['loss'][it]))
log_os.write('\n')
log_os.write('history_GN_valloss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense.history['val_loss'][it]))
log_os.write('\n')
log_os.write('history_GN2_loss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense2.history['loss'][it]))
log_os.write('\n')
log_os.write('history_GN2_valloss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense2.history['val_loss'][it]))
log_os.write('\n')
log_os.write('history_GN3_loss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense3.history['loss'][it]))
log_os.write('\n')
log_os.write('history_GN3_valloss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_dense3.history['val_loss'][it]))
log_os.write('\n')

log_os.write('history_LN_loss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_cnn.history['loss'][it]))
log_os.write('\n')
log_os.write('history_LN_valloss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_cnn.history['val_loss'][it]))
log_os.write('\n')
log_os.write('history_LN2_loss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_cnn2.history['loss'][it]))
log_os.write('\n')
log_os.write('history_LN2_valloss\n')
for it in range(N_epochs):
    log_os.write('%.3e\t' % (history_cnn2.history['val_loss'][it]))
log_os.write('\n')

log_os.close()








