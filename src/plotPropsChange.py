import six
from os.path import isfile
import numpy as np
import h5py
import copy
import math
import time
import argparse
import pandas as pd
from groupcat import *
from req_functions import *
#from util import *
#from snapshot import *
import os
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
plt.ioff()

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

parser = argparse.ArgumentParser(description='Define parameters')
parser.add_argument('--res', type=str, default='75',
                    help='define the resolution of TNG')
parser.add_argument('--parts', type=str, default='1820',
                    help='define the number of particles')
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')
parser.add_argument('--folder', type=str, default='results',
                    help='define the folder/perspective')
parser.add_argument('--n', type=str, default='10',
                    help='define number of std')
args = parser.parse_args()
path_to_save1 = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res+ 'n' + args.parts + 'TNG/avgProps/'
path_to_save2 = '/u/cgomez11/newSample/illustrisTNG/n_' + args.n  + '/L' + args.res + 'n'+ args.parts + 'TNG/avgProps/'
shift = 0.15

mass_stages = np.array([1,2,3,4])
#load properties from stellar perspective
stellar_mass_avg = np.load(path_to_save1 + 'stellar_avgMass.npy')
stellar_mass_std = np.load(path_to_save1 + 'stellar_stdMass.npy')

stellar_distance_avg = np.load(path_to_save1 + 'stellar_avgDist.npy')
stellar_distance_std = np.load(path_to_save1 + 'stellar_stdDist.npy')

stellar_rad_vel_avg = np.load(path_to_save1 + 'stellar_avgRadVel.npy')
stellar_rad_vel_std = np.load(path_to_save1 + 'stellar_stdRadVel.npy')

stellar_tan_vel_avg = np.load(path_to_save1 + 'stellar_avgTanVel.npy')
stellar_tan_vel_std = np.load(path_to_save1 + 'stellar_stdTanVel.npy')

stellar_other_mass_avg = np.load(path_to_save1 + 'stellar_avgOtherMass.npy')
stellar_other_mass_std = np.load(path_to_save1 + 'stellar_stdOtherMass.npy')

stellar_formation_avg = np.load(path_to_save1 + 'stellar_avgFormTime.npy')
stellar_formation_std = np.load(path_to_save1 + 'stellar_stdFormTime.npy')

#load properties from DM 
DM_mass_avg = np.load(path_to_save1 + 'DM_avgMass.npy')
DM_mass_std = np.load(path_to_save1 + 'DM_stdMass.npy')

DM_distance_avg = np.load(path_to_save1 + 'DM_avgDist.npy')
DM_distance_std = np.load(path_to_save1 + 'DM_stdDist.npy')

DM_rad_vel_avg = np.load(path_to_save1 + 'DM_avgRadVel.npy')
DM_rad_vel_std = np.load(path_to_save1 + 'DM_stdRadVel.npy')

DM_tan_vel_avg = np.load(path_to_save1 + 'DM_avgTanVel.npy')
DM_tan_vel_std = np.load(path_to_save1 + 'DM_stdTanVel.npy')

DM_other_mass_avg = np.load(path_to_save1 + 'DM_avgOtherMass.npy')
DM_other_mass_std = np.load(path_to_save1 + 'DM_stdOtherMass.npy')

DM_formation_avg = np.load(path_to_save1 + 'DM_avgFormTime.npy')
DM_formation_std = np.load(path_to_save1 + 'DM_stdFormTime.npy')

#load avg properties from combined perspectives
comb_mass_avg = np.load(path_to_save2 + 'comb_avgMass.npy')
comb_mass_std = np.load(path_to_save2 + 'comb_stdMass.npy')

comb_distance_avg = np.load(path_to_save2 + 'comb_avgDist.npy')
comb_distance_std = np.load(path_to_save2 + 'comb_stdDist.npy')

comb_rad_vel_avg = np.load(path_to_save2 + 'comb_avgRadVel.npy')
comb_rad_vel_std = np.load(path_to_save2 + 'comb_stdRadVel.npy')

comb_tan_vel_avg = np.load(path_to_save2 + 'comb_avgTanVel.npy')
comb_tan_vel_std = np.load(path_to_save2 + 'comb_stdTanVel.npy')

comb_other_mass_avg = np.load(path_to_save2 + 'comb_avgOtherMass.npy')
comb_other_mass_std = np.load(path_to_save2 + 'comb_stdOtherMass.npy')

comb_formation_avg = np.load(path_to_save2 + 'comb_avgFormTime.npy')
comb_formation_std = np.load(path_to_save2 + 'comb_stdFormTime.npy')

f1 = plt.figure()
f1.suptitle('Average properties at each selection step')
ax1 = plt.subplot(231)
#plt.figure(figsize=(20,20))
plt.errorbar(mass_stages, stellar_mass_avg, stellar_mass_std, linestyle='None', fmt='o',ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages + shift, DM_mass_avg, DM_mass_std, linestyle='None', fmt='o',ecolor='green',lw=1, label = 'DM')
#print(DM_mass_avg)
plt.errorbar(mass_stages + shift*2, comb_mass_avg, comb_mass_std, linestyle='None', fmt='o',ecolor='red',lw=1, label = 'Broad')
plt.legend(loc=4,  prop={'size': 7})
#plt.xlabel('Selection step')
plt.ylabel('Average log stellar mass/[$M_\odot$]', fontsize=9)
plt.xlim(xmin=0, xmax=5)
plt.ylim(ymin=7, ymax=11)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2 = plt.subplot(232)
plt.errorbar(mass_stages, stellar_other_mass_avg, stellar_other_mass_std, linestyle='None', fmt='o', ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages+shift, DM_other_mass_avg, DM_other_mass_std, linestyle='None', fmt='o', ecolor='green',lw=1, label='DM')
plt.errorbar(mass_stages+shift*2, comb_other_mass_avg, comb_other_mass_std, linestyle='None', fmt='o', ecolor='red',lw=1, label='Broad')
plt.ylabel('Average log DM mass/[$M_\odot$]', fontsize=9)
#plt.legend(loc=4,  prop={'size': 7})
plt.ylim(ymin=8, ymax=14)
plt.xlim(xmin=0, xmax=5)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

ax3 = plt.subplot(233)
plt.errorbar(mass_stages, stellar_distance_avg, stellar_distance_std, linestyle='None', fmt='o', ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages+shift, DM_distance_avg, DM_mass_std, linestyle='None', fmt='o', ecolor='green',lw=1, label='DM')
plt.errorbar(mass_stages+shift*2, comb_distance_avg, comb_distance_std, linestyle='None', fmt='o', ecolor='red',lw=1, label='Broad')
plt.ylabel('Average log distance [kpc]', fontsize=9)
#plt.legend(loc=4,  prop={'size': 7})
plt.ylim(ymin=2, ymax=3.5)
plt.xlim(xmin=0, xmax=5)
#ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

ax4 = plt.subplot(234)
plt.errorbar(mass_stages, stellar_rad_vel_avg, stellar_rad_vel_std, linestyle='None', fmt='o', ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages+shift, DM_rad_vel_avg, DM_rad_vel_std, linestyle='None', fmt='o', ecolor='green',lw=1, label='DM')
plt.errorbar(mass_stages+shift*2, comb_rad_vel_avg, comb_rad_vel_std, linestyle='None', fmt='o', ecolor='red',lw=1, label='Broad')
plt.ylabel('Average Radial velocity [km/s]', fontsize=9)
plt.xlabel('Selection step',fontsize=9)
#plt.legend(loc=4,  prop={'size': 7})
plt.ylim(ymin=-200, ymax=100)
plt.xlim(xmin=0, xmax=5)
ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

ax5 = plt.subplot(235)
plt.errorbar(mass_stages, stellar_tan_vel_avg, stellar_tan_vel_std, linestyle='None', fmt='o', ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages+shift, DM_tan_vel_avg, DM_tan_vel_std, linestyle='None', fmt='o', ecolor='green',lw=1, label='DM')
plt.errorbar(mass_stages+shift*2, comb_tan_vel_avg, comb_tan_vel_std, linestyle='None', fmt='o', ecolor='red',lw=1, label='Broad')
plt.ylabel('Average Tangential velocity [km/s]', fontsize=9)
plt.xlabel('Selection step', fontsize=9)
#plt.legend(loc=4,  prop={'size': 7})
plt.ylim(ymin=-50, ymax=350)
plt.xlim(xmin=0, xmax=5)
ax5.yaxis.set_major_locator(MaxNLocator(integer=True))

ax6 = plt.subplot(236)
plt.errorbar(mass_stages, stellar_formation_avg, stellar_formation_std, linestyle='None', fmt='o', ecolor='blue',lw=1, label='stellar')
plt.errorbar(mass_stages+shift, DM_formation_avg, DM_formation_std, linestyle='None', fmt='o', ecolor='green',lw=1, label='DM')
plt.errorbar(mass_stages+shift*2, comb_formation_avg, comb_formation_std, linestyle='None', fmt='o', ecolor='red',lw=1, label='Broad')
plt.ylabel('Formation time [z]', fontsize=9)
plt.xlabel('Selection step', fontsize=9)
#plt.legend(loc=4,  prop={'size': 7})
plt.ylim(ymin=0, ymax=3)
plt.xlim(xmin=0, xmax=5)
#ax6.yaxis.set_major_locator(MaxNLocator(integer=True))

f1.tight_layout()
f1.subplots_adjust(top=0.88)
f1.savefig(path_to_save1 + 'ALLavgProps.pdf')
plt.close(f1)





