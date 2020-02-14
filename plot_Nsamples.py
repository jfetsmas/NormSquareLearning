#plot comparisons of three architectures
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

filename= 'summary_variousNsamples_fixedratio.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
table = results[results[:,2].argsort()]

x = table[:,0]
trainloss1 = np.multiply(table[:,4],np.square(x))
testloss1 = np.multiply(table[:,6],np.square(x))
Nsamples = np.multiply(table[:,2],0.64)

par = np.polyfit(np.log(Nsamples),np.log(testloss1),1,full=True)
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
plt.plot(Nsamples,trainloss1,'--ks',label='train loss')
plt.plot(Nsamples,testloss1,'-ko',label='test loss')
plt.plot(Nsamples,np.exp(np.poly1d([slope,intercept])(np.log(Nsamples))),'r-')
plt.ylabel('mean squared error loss',size=24)
plt.xlabel('number of the training samples',size=24)
plt.yscale('log')
plt.legend(loc='upper right')
plt.xscale('log')
#plt.title('slope of the best fitted line: %.3f' %slope, size=16)

imagefilename = 'plot_variousNsamples_fixedratio.png'
plt.savefig(imagefilename, bbox_inches='tight')
#plt.tight_layout()
plt.show()