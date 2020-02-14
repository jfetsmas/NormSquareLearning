#plot comparisons of three architectures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

filename= 'summary_global_network_cosine_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
table = results[results[:,0].argsort()]

x = table[:,0]
trainloss1 = np.multiply(table[:,4],np.square(x))
testloss1 = np.multiply(table[:,6],np.square(x))
# trainloss1 = table[:,4]
# testloss1 = table[:,6]

filename= 'summary_locally_connected_network_cosine_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
table = results[results[:,0].argsort()]

trainloss2 = np.multiply(table[:,4],np.square(x))
testloss2 = np.multiply(table[:,6],np.square(x))
# trainloss2 = table[:,4]
# testloss2 = table[:,6]

filename= 'summary_local_network_cosine_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
table = results[results[:,0].argsort()]

trainloss3 = np.multiply(table[:,4],np.square(x))
testloss3 = np.multiply(table[:,6],np.square(x))
# trainloss3 = table[:,4]
# testloss3 = table[:,6]


# plt.subplot(1,2,1)
# plt.plot(x,y,'ko')
# plt.ylabel('test_loss')
# plt.xlabel('Ne')
# plt.yscale('log')

plt.rcParams.update({'font.size': 12})
plt.plot(x,trainloss1,'--ro',label='GN train')
plt.plot(x,testloss1,'-ro',label='GN test')
plt.plot(x,trainloss2,'--bo',label='LCN train')
plt.plot(x,testloss2,'-bo',label='LCN test')
plt.plot(x,trainloss3,'--go',label='LN train')
plt.plot(x,testloss3,'-go',label='LN test')
plt.ylabel('Loss',size=24)
plt.xlabel('input dimension',size=24)
plt.yscale('log')
plt.legend(loc='upper left')
#plt.xscale('log')

imagefilename = 'plot_compare_arch_cosine.png'
plt.savefig(imagefilename, bbox_inches='tight')
#plt.tight_layout()
#plt.show()