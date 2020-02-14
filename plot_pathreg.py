import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

filename = 'summary_global_network_square_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
results = results[results[:,0].argsort()]
d1 = results[:,0]
train_loss1 = np.multiply(results[:,4],np.square(d1))
test_loss1 = np.multiply(results[:,6],np.square(d1))


filenamepre = 'summary_pathnormreg_0210'
filename = filenamepre+'.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(8))
results = results[results[:,1].argsort()]

results1 = results[150:180]
table1 = results1[results1[:,0].argsort()]
print(table1)

d2 = table1[:,0]
train_loss2 = np.multiply(table1[:,5],np.square(d2))
test_loss2 = np.multiply(table1[:,7],np.square(d2))

# train_loss2 = table2[:,5]
# test_loss2 = table2[:,7]

par = np.polyfit(np.log(d1),np.log(test_loss1),1,full=True)
slope=par[0][0]
intercept=par[0][1]
print(slope)
print(intercept)
par = np.polyfit(np.log(d2),np.log(test_loss2),1,full=True)
slope=par[0][0]
intercept=par[0][1]
print(slope)
print(intercept)


filename = 'summary_local_network_square_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
results = results[results[:,0].argsort()]
d3 = results[:,0]
train_loss3 = np.multiply(results[:,4],np.square(d3))
test_loss3 = np.multiply(results[:,6],np.square(d3))

filename = 'summary_locally_connected_network_square_0207.txt'
results  = np.loadtxt(filename, delimiter=None, usecols=range(7))
results = results[results[:,0].argsort()]
d4 = results[:,0]
train_loss4 = np.multiply(results[:,4],np.square(d4))
test_loss4 = np.multiply(results[:,6],np.square(d4))

plt.rcParams.update({'font.size': 16})
# plt.plot(d2,test_loss2, 'b-', label='reg 1E-8')
# plt.plot(d2,train_loss2, 'b--')
plt.plot(d1,test_loss1,'k-', label='GN')
plt.plot(d1,train_loss1,'k--')
# plt.plot(results1[0:30,0], results1[0:30,7],'ko')
# plt.plot(results1[30:60,0], results1[30:60,7],'ks')
# plt.plot(results1[60:90,0], results1[60:90,7],'bo')
# plt.plot(results1[90:120,0], results1[90:120,7],'bs')
# plt.plot(results1[120:150,0], results1[120:150,7],'k-')
plt.plot(d2,test_loss2,'b-',label='pathreg 1E-5')
plt.plot(d2,train_loss2,'b--')
plt.plot(d2,np.exp(np.poly1d([slope,intercept])(np.log(d2))),'y-')
plt.plot(d4,test_loss4, 'g-', label='LCN')
plt.plot(d4,train_loss4, 'g--')
plt.plot(d3,test_loss3, 'r-', label='LN')
plt.plot(d3,train_loss3, 'r--')


#plt.plot(d,np.exp(np.poly1d([slope,intercept])(np.log(d))),'r-')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('input dimension', size=24)
plt.ylabel('mean squared error loss',size=24)
plt.legend(loc='best')
plt.title('slope of the best fitted line (yellow): %.3f' %slope, size=16)

# x = d[8:]
# y = test_loss2[8:]

# par = np.polyfit(np.log(x),np.log(y),1,full=True)
# slope=par[0][0]
# intercept=par[0][1]
# print(slope)

# plt.subplot(1,2,1)
# plt.plot(x,y,'ko')
# plt.ylabel('test_loss')
# plt.xlabel('Ne')
# plt.yscale('log')

# plt.subplot(1,2,2)
# plt.plot(x,y,'ko')
# plt.ylabel('test_loss')
# plt.xlabel('Ne')
# plt.yscale('log')
# plt.xscale('log')

imagefilename = filenamepre+'_results.png'
#imagefilename = 'test_Nsample1E5.png'
plt.savefig(imagefilename, bbox_inches='tight')
plt.tight_layout()
plt.show()