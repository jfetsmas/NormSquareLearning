import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(description='NormSampleCompact')
parser.add_argument('--d', type=int, default='4', metavar='N',
                    help='number of electrons (default: 4)')
parser.add_argument('--Nsample', type=int, default=1000, metavar='N',
                    help='input number of nodes per layer (default: 1000)')
args = parser.parse_args()



d = args.d
Nsample = args.Nsample

np.random.seed(1234) #for reproducibility
x_traj = np.random.uniform(-1,1,(Nsample,d))
y_traj = np.sum(np.power(x_traj,2),axis=1)

data_folder = 'data/'
filenameIpt = data_folder + 'square_'+ 'x_' + str(d) + '_Nsample_' + str(Nsample) +'.dat'
filenameOpt = data_folder + 'square_'+ 'y_' + str(d) + '_Nsample_' + str(Nsample) +'.dat'

np.savetxt(filenameIpt, x_traj)
np.savetxt(filenameOpt, y_traj)
