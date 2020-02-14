#plot comparisons of three architectures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

filename= 'pathnorm_global_network_square_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(8))
table = results[results[:,0].argsort()]

x = table[:,0]
trainloss1 = np.multiply(table[:,4],np.square(x))
testloss1 = np.multiply(table[:,6],np.square(x))
pathnorm1 = table[:,7]




par = np.polyfit(np.log(x),np.log(pathnorm1),1,full=True)
slope=par[0][0]
intercept=par[0][1]
print(slope)
print(intercept)


# plt.subplot(1,2,1)
# plt.plot(x,y,'ko')
# plt.ylabel('test_loss')
# plt.xlabel('Ne')
# plt.yscale('log')

plt.rcParams.update({'font.size': 16})
#plt.plot(x,trainloss1,'--ks',label='GN train')
#plt.plot(x,testloss1,'-ko',label='GN test')
plt.plot(x,pathnorm1,'--ko',label='pathnorm')
plt.plot(x,np.exp(np.poly1d([slope,intercept])(np.log(x))),'r-')
plt.ylabel('norm',size=24)
plt.xlabel('input dimension',size=24)
plt.yscale('log')
plt.legend(loc='upper left')
plt.xscale('log')
plt.title('slope of the best fitted line: %.3f' %slope, size=16)

#imagefilename = 'plot_pathnorm_square.png'
#plt.savefig(imagefilename, bbox_inches='tight')
#plt.tight_layout()
plt.show()