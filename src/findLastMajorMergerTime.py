import numpy as np
import scipy as sp
import h5py
import copy
import math
import time
import scipy.signal
import argparse
import pandas as pd
from groupcat import *
from sublink import *
from req_functions import *
#from util import *
#from snapshot import *
import os
import matplotlib

parser = argparse.ArgumentParser(description='Define parameters')
parser.add_argument('--res', type=str, default='75',
                    help='define the resolution of TNG')
parser.add_argument('--parts', type=str, default='1820',
                    help='define the number of particles')
parser.add_argument('--save_id', type=str, default='none',
                    help='define the crtierion step to visualize')
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')
parser.add_argument('--folder', type=str, default='newResults',
                    help='define the folder to save the numbers')
parser.add_argument('--n', type=str, default='10',
                    help='define number of std')

args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
snapshot = 99
#path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'
sample = args.sample_type
if sample =='comb':
    path_to_save = '/u/cgomez11/newSample/illustrisTNG/n_' + args.n  + '/L' + args.res + 'n'+ args.parts + 'TNG/'
else:
    path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'

zs = np.load(path_to_save + 'z_' + args.res + '_' + args.parts + '.npy') #en orden ascendente
sub_subhalo_id = np.load(path_to_save + sample + '_subhalos_ID_afterMass.npy')

z_all_sub_subhalos = []
snap_all_sub_subhalos = []
ratio =1/5
fields = ['SubhaloID', 'NextProgenitorID', 'MainLeafProgenitorID', 'FirstProgenitorID', 'SubhaloMassType', 'SnapNum', 'SubhaloMass']
t1 = time.time()
for count, each_id in enumerate(sub_subhalo_id):
	tree = loadTree(basePath, snapshot, each_id,fields=fields)
	if tree is not None:
		nMergers, list_snaps = numMergers(tree,minMassRatio=ratio)
		if len(list_snaps)>0 and nMergers>0: #si encontro mergers
			snap_all_sub_subhalos.append(list_snaps[0]) #el primero que encontro es el mas reciente
			z_all_sub_subhalos.append(zs[list_snaps[0]])
		else:
			snap_all_sub_subhalos.append(0)
			z_all_sub_subhalos.append(0)
	else:
		snap_all_sub_subhalos.append(0)
		z_all_sub_subhalos.append(0)


position = np.arange(sub_subhalo_id.shape[0])
all_times = np.stack((position, sub_subhalo_id, snap_all_sub_subhalos, z_all_sub_subhalos), axis=1)
filename_ft = path_to_save + sample +'_LastMajorMergerTime.txt'
np.savetxt(filename_ft, all_times, fmt=['%u','%u','%u','%.2e'], header = 'Pos ID Snap Z')

tend = time.time()
print('Elapsed time:',tend-t1) 
