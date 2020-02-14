# Tensorflow 2.0 code for Norm-squared learning with pathnorm regularization
# Trajectory of training is saved, and best model with lowest validation loss is saved.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import argparse
import h5py
# this is necessary in Mac
os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = argparse.ArgumentParser(description='NormLearning')
parser.add_argument('--epoch', type=int, default=1000, metavar='N',
                    help='input number of epochs for training (default: 1000)')
parser.add_argument('--d', type=int, default='4', metavar='N',
                    help='input dimension (default: 4)')
parser.add_argument('--Nsamples', type=int, default=1000, metavar='N',
                    help='number of training samples (default 1000')
parser.add_argument('--reg', type=float, default=0.0001, metavar='N',
                    help='regularization constant (default: 0.0001)')
args = parser.parse_args()
# parameters
d = args.d
N_epochs = args.epoch
N_nodes = 50
lr = 0.01
reg = args.reg
Nsamples = args.Nsamples
data_folder = 'data/'
models_folder = 'models/' # folder to save the models (weights)

# this is generic for al the test_files
dir__name =  os.getcwd()
script__name = os.path.basename(__file__)
script__name = script__name.replace(".", "_")

str_best_model =  models_folder   +   \
              'pathreg_model_'  + script__name +   \
              '_d_'       + str(args.d) + \
              '_epochs_'  + str(args.epoch) + \
              '_Nsamples_'+ str(args.Nsamples) + \
              '_reg_'     + str(args.reg) + '_best.h5'


filenameIpt = data_folder + 'square_x_' + str(args.d) + '_Nsample_1000000.dat'
filenameOpt = data_folder + 'square_y_' + str(args.d) + '_Nsample_1000000.dat'

xraw_huge  = np.loadtxt(filenameIpt, delimiter=None, usecols=range(d))
yraw_huge  = np.loadtxt(filenameOpt, delimiter=None, usecols=range(1))
xraw = xraw_huge[0:Nsamples,:]
yraw = yraw_huge[0:Nsamples]/d

n_train = int(0.8*Nsamples)
n_test = int(0.2*Nsamples)


batch_size = n_train//100


xraw2 = np.zeros(xraw.shape)
for i in range(0,Nsamples):
    tmp = np.reshape(xraw[i,:],(d,1))
    norm = np.linalg.norm(tmp, axis=1)
    tmp2 = tmp[norm.argsort()]
    tmp3 = tmp2.ravel()
    xraw2[i,:] = tmp3

y_test = yraw[0:n_test]
y_train = yraw[n_test:]

x_test = xraw2[0:n_test,:]
x_train = xraw2[n_test:,:]

# split the training data into training and validation
n_train2 = int(0.8*n_train)
n_val = int(0.2*n_train)

y_val = y_train[n_train2:]
y_train2 = y_train[0:n_train2]

x_val = x_train[n_train2:,:]
x_train2 = x_train[0:n_train2,:]

DenseNodes = N_nodes*d

model = Sequential()
model.add(Dense(DenseNodes, activation='relu',use_bias=True, input_dim=d))
model.add(Dense(DenseNodes, activation='relu',use_bias=True))
model.add(Dense(1, activation='linear',use_bias=False))
model.add(Lambda(lambda x: x / d))
model.summary()


# we define the optimizer and the loss
optimizer = tf.optimizers.Adam(learning_rate=lr)
loss_mse = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(inputs, outputs):
# funtion to perform one training step
  with tf.GradientTape() as tape:
    # we use the model the predict the outcome
    predictions = model(inputs,training=True)

    # we extract the parameters
    params = model.trainable_variables

    # we compute the l1 norm
    path_norm_loss = tf.abs(params[0])
    for i in range(2,len(params),2):
      # we only multiplyt the weight matrices
      # we don't consider the biases in this case
      path_norm_loss = tf.linalg.matmul(path_norm_loss, tf.abs(params[i]))

    path_norm_loss = tf.reduce_sum(path_norm_loss) / d

    # fidelity loss usin mse
    pred_loss = loss_mse(outputs, tf.squeeze(predictions,[1]))
    # we add the regularization
    total_loss = pred_loss + reg*path_norm_loss

  # compute the gradients of the total loss with respect to the trainable variables
  gradients = tape.gradient(total_loss, model.trainable_variables)
  # update the parameters of the network
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return (pred_loss, total_loss)


#we create our dataset
print(batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train2, y_train2))
train_dataset = train_dataset.batch(batch_size)

train_loss_array = []
val_loss_array = []
min_val_err = 1000
# training loop
# Try replacing the for loop by model.fit
for epoch in range(N_epochs):
  print('Start of epoch %d' % (epoch,))

  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    loss_value = train_step(x_batch_train, y_batch_train)

  yNNT = model(x_train2,training=False)
  train_loss = loss_mse(tf.squeeze(yNNT,[1]), y_train2)

  yNNV = model(x_val,training=False)
  val_loss = loss_mse(tf.squeeze(yNNV,[1]), y_val)

  train_loss_array.append(train_loss)
  val_loss_array.append(val_loss)

  print('Train loss at epoch %s: %s' % (epoch, float(train_loss)))
  print('Validation loss at epoch %s: %s' % (epoch, float(val_loss)))

  if val_loss < min_val_err:
    print('Lower validation loss achieved!!! Saving best model with validation loss %.3e' % (val_loss))
    min_val_err = val_loss
    model.save(str_best_model)


# adam = tf.optimizers.Adam(learning_rate=lr)
# model.compile(loss='mean_squared_error', optimizer=adam)

# checkpointer = ModelCheckpoint(filepath=str_best_model, \
#                                verbose=1, save_best_only=True, \
#                                monitor='loss', save_weights_only=False, \
#                                mode='auto', period=1)
# history = model.fit(x_train, y_train,validation_split=0.2, batch_size=batch_size, \
#                     epochs=epochs, verbose=0,callbacks=[checkpointer])

best_model = tf.keras.models.load_model(str_best_model, custom_objects={'d': d}, compile=False)


yNNT = best_model.predict(x_train2)
errTrain = np.linalg.norm(yNNT.reshape(yNNT.size)-y_train2)**2/n_train2

yNNV = best_model.predict(x_val)
errVal = np.linalg.norm(yNNV.reshape(yNNV.size)-y_val)**2/n_val

yNN = best_model.predict(x_test)
errTest = np.linalg.norm(yNN.reshape(yNN.size)-y_test)**2/n_test

best_params = best_model.trainable_variables

# we compute the path norm (this should be encapsulated)
path_norm = np.abs(best_params[0])
for i in range(2,len(best_params),2):
  # we only multiplyt the weight matrices
  # we don't consider the biases in this case
  path_norm = np.matmul(path_norm, np.abs(best_params[i]))
  
path_norm = np.sum(path_norm) / d
  

print('error with best model')
print("mean squared error of train data in NN: %.3e" % (errTrain))
print("mean squared error of validation data in NN: %.3e" %  (errVal))
print("mean squared error of test data in NN: %.3e" %  (errTest))
print("path norm of the best model: %.3e" % (path_norm))

# Change the output filename here
log_os = open('summary_pathnormreg_0210', "a")
log_os.write('%d\t%.3e\t' % (args.d, args.reg))
log_os.write('%d\t%d\t' % (args.epoch, Nsamples))
log_os.write('%d\t' % (model.count_params()))
log_os.write('%.3e\t%.3e\t%.3e\t%.3e\t' % (errTrain, errVal, errTest, path_norm))
log_os.write('\n')
log_os.close()

