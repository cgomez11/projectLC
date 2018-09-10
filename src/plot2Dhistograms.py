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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
plt.ioff()

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)  

parser = argparse.ArgumentParser(description='Define parameters')
parser.add_argument('--res', type=str, default='75',
                    help='define the resolution of TNG')
parser.add_argument('--parts', type=str, default='1820',
                    help='define the number of particles')
parser.add_argument('--vis_step', type=str, default='none',
                    help='define the crtierion step to visualize')
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')
parser.add_argument('--folder', type=str, default='newStrategy',
                    help='define the folder to save')
parser.add_argument('--n', type=str, default='10',
                    help='define number of std')

formato = '.pdf'
args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
sample = args.sample_type
snapshot = 99
#path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'
if sample =='comb':
    path_to_save = '/u/cgomez11/newSample/illustrisTNG/n_' + args.n  + '/L' + args.res + 'n'+ args.parts + 'TNG/'
else:
    path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'

sub_subhalo_id = np.load(path_to_save + sample + '_subhalos_ID_afterMass.npy')
df_pairs_info = np.load(path_to_save + sample + '_df_pairCandidatesAfterMass.pkl')
#load the IDs to view properties at any step
if args.vis_step == '1':
    if sample == 'comb':
        arr_indices = np.load(path_to_save + sample + '_pair_mass_ids.npy')
    else: #ya tengo los DF de los pair candidates que cumplen el criterio de masa
        arr_indices = np.load(path_to_save + sample + '_pair_ids.npy')
elif args.vis_step == '2':
    arr_indices = np.load(path_to_save + sample + '_thresh_ids.npy')
elif args.vis_step == '3':
    arr_indices = np.load(path_to_save + sample + '_isolated_ids.npy')
elif args.vis_step == '4':
    if sample == 'comb':
        arr_indices = np.load(path_to_save + sample + '_tangential_ids.npy')
    else:
        arr_indices = np.load(path_to_save + sample + '_velocity_ids.npy')

halo_A_pair = arr_indices[:,0]
halo_B_pair = arr_indices[:,1]

#extract subhalos by ID 
current_ind_A = sub_subhalo_id[halo_A_pair]
current_ind_B = sub_subhalo_id[halo_B_pair]

#initialize arrays to save the properties that we want to plot
radial_vels = np.empty((0), dtype=int)
tangential_vels = np.empty((0), dtype=int)
distances = np.empty((0), dtype=int)
stellar_mass_min = np.empty((0), dtype=int)
stellar_mass_max = np.empty((0), dtype=int)
DM_mass_min = np.empty((0), dtype=int)
DM_mass_max = np.empty((0), dtype=int)

for count, ii in enumerate(current_ind_A):
    row_pair = df_pairs_info[df_pairs_info['ID_A']==ii] #objeto tipo serie
    radial_vels = np.append(radial_vels, row_pair['Radial_vel'].values[0])
    tangential_vels = np.append(tangential_vels, row_pair['Tan_vel'].values[0])
    distances = np.append(distances, row_pair['Distance'].values[0])
    ms_A = row_pair['Stellar_massA'].values[0]
    ms_B = row_pair['Stellar_massB'].values[0]
    if ms_A <= ms_B:
        stellar_mass_min = np.append(stellar_mass_min, ms_A)
        stellar_mass_max = np.append(stellar_mass_max, ms_B)
    else: #mB < mA
        stellar_mass_min = np.append(stellar_mass_min, ms_B)
        stellar_mass_max = np.append(stellar_mass_max, ms_A)

    dm_A = row_pair['DM_massA'].values[0]
    dm_B = row_pair['DM_massB'].values[0]
    if dm_A < dm_B:
        DM_mass_min = np.append(DM_mass_min, dm_A)
        DM_mass_max = np.append(DM_mass_max, dm_B)
    else: #mB < mA
        DM_mass_min = np.append(DM_mass_min, dm_B)
        DM_mass_max = np.append(DM_mass_max, dm_A)
distances = distances/ 0.704 #kpc
R_stellar = stellar_mass_min/stellar_mass_max
R_DM = DM_mass_min/DM_mass_max

path_to_save_ims = path_to_save + 'newImages/'
makedir(path_to_save_ims)

#plot 1: 2D histogram of radial and tangential velocity

f1 = plt.figure()
plt.subplot(211)
plt.hist2d(tangential_vels, radial_vels, norm=mpl.colors.LogNorm(), bins=64)
plt.xlim([0,200])
plt.ylim([-200,100])
plt.colorbar()
plt.clim(1e0,1e4)
someX, someY = 0, -127
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX, someY), 22, 8, fill=None, alpha=1))
plt.text(15, -100, s='M31-MW', fontsize=10)
plt.xlabel('Tangential Velocity [km/s]', fontsize=10)
plt.ylabel('Radial Velocity [km/s]', fontsize=10)
plt.title(sample + ' distribution of velocities after ' + args.vis_step, fontsize=12)

#print('cociente dM: ',R_DM)
#plot 2: 2D histogram of mass ratio stellar y DM
plt.subplot(212)
plt.hist2d(R_DM, R_stellar, norm=mpl.colors.LogNorm(), bins=64);
someX, someY = 0.435, 0.472
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX, someY), 0.117, 0.053, fill=None, alpha=1))
plt.text(0.43, 0.57, s='M31-MW', fontsize=10)
plt.xlabel('DM Mass ratio', fontsize=10)
plt.ylabel('Stellar Mass ratio', fontsize=10)
plt.colorbar()
plt.clim(1e0,1e4)
plt.xlim([0,1])
plt.ylim([0,1])
plt.title(sample + ' mass ratios after ' + args.vis_step, fontsize=12)
f1.tight_layout()
f1.savefig(path_to_save_ims + sample +'_2Dhists_' + args.vis_step  + formato)
plt.close(f1)

#2D plot of formation and last merger time
table_formation = np.genfromtxt(path_to_save + sample + '_FormationTimes.txt')
z_half_formation = table_formation[:,4]
table_lastMerger = np.genfromtxt(path_to_save + sample + '_LastMajorMergerTime.txt')
z_last_merger = table_lastMerger[:,3]

sub_z_formation = z_half_formation[np.concatenate([halo_A_pair, halo_B_pair])]
new_sub_z_formation = sub_z_formation[sub_z_formation>0]
sub_z_merger = z_last_merger[np.concatenate([halo_A_pair, halo_B_pair])]
sub_z_merger = sub_z_merger[sub_z_formation>0]
f2 = plt.figure()
plt.hist2d(new_sub_z_formation, sub_z_merger, norm=mpl.colors.LogNorm(), bins=64);
plt.xlabel('Formation time [z]', fontsize=10)
plt.ylabel('Last major merger time [z]', fontsize=10)
plt.colorbar()
plt.clim(1e0,1e4)
plt.xlim([0,5])
plt.ylim([0,8])
plt.title(sample + ' times from merger times ' + args.vis_step, fontsize=12)
f2.tight_layout()
f2.savefig(path_to_save_ims + sample +'_2DhistTimes_' + args.vis_step  + formato)
plt.close(f2)
