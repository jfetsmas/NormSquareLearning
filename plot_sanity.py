import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

filenamepre = 'norm_loading_decay_0.003_d_40_epoch_500'
filename = filenamepre+'.txt'
f = open(filename,"r")
f1 = f.readlines()
history_dense_loss = []
history_dense_valloss = []
history_dense2_loss = []
history_dense2_valloss = []
history_dense3_loss = []
history_dense3_valloss = []
history_cnn_loss = []
history_cnn_valloss = []
history_cnn2_loss = []
history_cnn2_valloss = []
for i_line, line in enumerate(f1):
	if line.startswith('history_GN_loss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense_loss.append(float(word))
	if line.startswith('history_GN_valloss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense_valloss.append(float(word))
	if line.startswith('history_GN2_loss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense2_loss.append(float(word))
	if line.startswith('history_GN2_valloss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense2_valloss.append(float(word))	
	if line.startswith('history_GN3_loss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense3_loss.append(float(word))
	if line.startswith('history_GN3_valloss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_dense3_valloss.append(float(word))
	if line.startswith('history_LN_loss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_cnn_loss.append(float(word))
	if line.startswith('history_LN_valloss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_cnn_valloss.append(float(word))
	if line.startswith('history_LN2_loss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_cnn2_loss.append(float(word))
	if line.startswith('history_LN2_valloss'):
		dataline = f1[i_line+1]
		for word in dataline.split():
			history_cnn2_valloss.append(float(word))
			
			
			
			
plt.plot(history_dense_loss+history_dense2_loss, 'r')
plt.plot(history_dense_valloss+history_dense2_valloss,'r:')
plt.plot(history_dense_loss+history_dense3_loss,'b')
plt.plot(history_dense_valloss+history_dense3_valloss,'b:')
plt.plot(history_cnn_loss+history_cnn2_loss, 'g')
plt.plot(history_cnn_valloss+history_cnn2_valloss,'g:')
plt.yscale('log')
#plt.ylim(0,1)
plt.ylabel('loss',size=20)
plt.xlabel('epoch',size=20)
plt.legend(['GN train after loading', 'GN validation after loading','GN train','GN validation','LN train','LN validation'], loc='upper right')
#plt.savefig('model loss', bbox_inches='tight')
imagefilename = filenamepre+'.png'
plt.savefig(imagefilename, bbox_inches='tight')
plt.show()
			
			
			
			
			
			