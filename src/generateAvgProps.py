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
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')
parser.add_argument('--folder', type=str, default='newStrategy',
                    help='define the folder to save')
parser.add_argument('--step', type=str, default='save',
                    help='define step')
parser.add_argument('--n', type=str, default='10',
                    help='define number of std')

args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
sample = args.sample_type
snapshot = 99

if sample =='comb':
	path_to_save = '/u/cgomez11/newSample/illustrisTNG/n_' + args.n  + '/L' + args.res + 'n'+ args.parts + 'TNG/'
else:
	path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'

table = np.genfromtxt(path_to_save + sample + '_FormationTimes.txt')
z_half_mass = table[:,4]
snap_half_mass = table[:,5]
true_z_half_mass = table[:,6]

sub_subhalo_id = np.load(path_to_save + sample + '_subhalos_ID_afterMass.npy')
df_pairs_info = np.load(path_to_save + sample + '_df_pairCandidatesAfterMass.pkl')
#load indices at each step
if sample == 'comb': #ind de masas y velocidades es distinto
	arr_id = np.load(path_to_save + sample + '_pair_mass_ids.npy')
	arr_velocity = np.load(path_to_save + sample + '_tangential_ids.npy')
else:
	arr_id = np.load(path_to_save + sample + '_pair_ids.npy')
	arr_velocity = np.load(path_to_save + sample + '_velocity_ids.npy')
halo_A = arr_id[:,0]
halo_B = arr_id[:,1]
arr_pair_thresh = np.load(path_to_save + sample + '_thresh_ids.npy')
halo_A_pair_thresh = arr_pair_thresh[:,0]
halo_B_pair_thresh = arr_pair_thresh[:,1]
arr_isolated = np.load(path_to_save + sample + '_isolated_ids.npy')
halo_A_isolated = arr_isolated[:,0]
halo_B_isolated = arr_isolated[:,1]
halo_A_velocity = arr_velocity[:,0]
halo_B_velocity = arr_velocity[:,1]


#properties to save
mass_avg = np.empty((0), dtype=int)
mass_std = np.empty((0), dtype=int)
distance_avg = np.empty((0), dtype=int)
distance_std = np.empty((0), dtype=int)
radial_avg = np.empty((0), dtype=int)
radial_std = np.empty((0), dtype=int)
tan_avg = np.empty((0), dtype=int)
tan_std = np.empty((0), dtype=int)
other_mass_avg = np.empty((0), dtype=int)
other_mass_std = np.empty((0), dtype=int)

#avg properties at step 1
if sample != 'comb':
    all_masses_1 = np.concatenate([df_pairs_info['Stellar_massA'], df_pairs_info['Stellar_massB']])
    all_other_mass_1 = np.concatenate([df_pairs_info['DM_massA'], df_pairs_info['DM_massB']])
    all_rad_vels_1 = df_pairs_info['Radial_vel']
    all_tan_vels_1 = df_pairs_info['Tan_vel']
    all_distances_1 = df_pairs_info['Distance']

else: #si es comb: agregar otros inds de masa
	current_ind_halo_A = sub_subhalo_id[halo_A]
	all_rad_vels_1 = np.empty((0), dtype=int)
	all_tan_vels_1 = np.empty((0), dtype=int)
	all_distances_1 = np.empty((0), dtype=int)
	all_masses_1 = np.empty((0), dtype=int)
	all_other_mass_1 = np.empty((0), dtype=int)
	#avg properties at step 1
	for count, ii in enumerate(current_ind_halo_A):
	    row_pair = df_pairs_info[df_pairs_info['ID_A']==ii] #objeto tipo serie
	    all_rad_vels_1 = np.append(all_rad_vels_1, row_pair['Radial_vel'].values[0])
	    all_tan_vels_1 = np.append(all_tan_vels_1, row_pair['Tan_vel'].values[0])
	    all_distances_1 = np.append(all_distances_1, row_pair['Distance'].values[0])
	    all_masses_1 = np.append(all_masses_1, row_pair['Stellar_massA'].values[0])
	    all_masses_1 = np.append(all_masses_1, row_pair['Stellar_massB'].values[0])
	    all_other_mass_1 = np.append(all_other_mass_1, row_pair['DM_massA'].values[0])
	    all_other_mass_1 = np.append(all_other_mass_1, row_pair['DM_massB'].values[0])

#find avg properties
log_mass_1 = np.log10(all_masses_1)
log_other_mass_1 = np.log10(all_other_mass_1)
log_distances_1 = np.log10(all_distances_1)
mass_avg = np.append(mass_avg, np.mean(log_mass_1))
mass_std = np.append(mass_std, np.std(log_mass_1))
other_mass_avg = np.append(other_mass_avg, np.mean(log_other_mass_1))
other_mass_std = np.append(other_mass_std, np.std(log_other_mass_1))
distance_avg = np.append(distance_avg, np.mean(log_distances_1))
distance_std = np.append(distance_std, np.std(log_distances_1))
tan_avg = np.append(tan_avg, np.mean(all_tan_vels_1))
tan_std = np.append(tan_std, np.std(all_tan_vels_1))
radial_avg = np.append(radial_avg, np.mean(all_rad_vels_1))
radial_std = np.append(radial_std, np.std(all_rad_vels_1))

#avg properties after step 2
current_ind_halo_A_thresh = sub_subhalo_id[halo_A_pair_thresh]
all_rad_vels_2 = np.empty((0), dtype=int)
all_tan_vels_2 = np.empty((0), dtype=int)
all_distances_2 = np.empty((0), dtype=int)
all_masses_2 = np.empty((0), dtype=int)
all_other_mass_2 = np.empty((0), dtype=int)
for count, ii in enumerate(current_ind_halo_A_thresh):
    row_pair = df_pairs_info[df_pairs_info['ID_A']==ii] #objeto tipo serie
    all_rad_vels_2 = np.append(all_rad_vels_2, row_pair['Radial_vel'].values[0])
    all_tan_vels_2 = np.append(all_tan_vels_2, row_pair['Tan_vel'].values[0])
    all_distances_2 = np.append(all_distances_2, row_pair['Distance'].values[0])
    all_masses_2 = np.append(all_masses_2, row_pair['Stellar_massA'].values[0])
    all_masses_2 = np.append(all_masses_2, row_pair['Stellar_massB'].values[0])
    all_other_mass_2 = np.append(all_other_mass_2, row_pair['DM_massA'].values[0])
    all_other_mass_2 = np.append(all_other_mass_2, row_pair['DM_massB'].values[0])

log_mass_2 = np.log10(all_masses_2)
log_other_mass_2 = np.log10(all_other_mass_2)
log_distances_2 = np.log10(all_distances_2)

mass_avg = np.append(mass_avg, np.mean(log_mass_2))
mass_std = np.append(mass_std, np.std(log_mass_2))
other_mass_avg = np.append(other_mass_avg, np.mean(log_other_mass_2))
other_mass_std = np.append(other_mass_std, np.std(log_other_mass_2))
distance_avg = np.append(distance_avg, np.mean(log_distances_2))
distance_std = np.append(distance_std, np.std(log_distances_2))
tan_avg = np.append(tan_avg, np.mean(all_tan_vels_2))
tan_std = np.append(tan_std, np.std(all_tan_vels_2))
radial_avg = np.append(radial_avg, np.mean(all_rad_vels_2))
radial_std = np.append(radial_std, np.std(all_rad_vels_2))

#avg properties after step 3
all_rad_vels_3 = np.empty((0), dtype=int)
all_tan_vels_3 = np.empty((0), dtype=int)
all_distances_3 = np.empty((0), dtype=int)
all_masses_3 = np.empty((0), dtype=int)
all_other_mass_3 = np.empty((0), dtype=int)
current_ind_halo_A_isolated = sub_subhalo_id[halo_A_isolated]
for count, ii in enumerate(current_ind_halo_A_thresh):
    row_pair = df_pairs_info[df_pairs_info['ID_A']==ii] #objeto tipo serie
    all_rad_vels_3 = np.append(all_rad_vels_3, row_pair['Radial_vel'].values[0])
    all_tan_vels_3 = np.append(all_tan_vels_3, row_pair['Tan_vel'].values[0])
    all_distances_3 = np.append(all_distances_3, row_pair['Distance'].values[0])
    all_masses_3 = np.append(all_masses_3, row_pair['Stellar_massA'].values[0])
    all_masses_3 = np.append(all_masses_3, row_pair['Stellar_massB'].values[0])
    all_other_mass_3 = np.append(all_other_mass_3, row_pair['DM_massA'].values[0])
    all_other_mass_3 = np.append(all_other_mass_3, row_pair['DM_massB'].values[0])

log_mass_3 = np.log10(all_masses_3)
log_other_mass_3 = np.log10(all_other_mass_3)
log_distances_3 = np.log10(all_distances_3)

mass_avg = np.append(mass_avg, np.mean(log_mass_3))
mass_std = np.append(mass_std, np.std(log_mass_3))
other_mass_avg = np.append(other_mass_avg, np.mean(log_other_mass_3))
other_mass_std = np.append(other_mass_std, np.std(log_other_mass_3))
distance_avg = np.append(distance_avg, np.mean(log_distances_3))
distance_std = np.append(distance_std, np.std(log_distances_3))
tan_avg = np.append(tan_avg, np.mean(all_tan_vels_3))
tan_std = np.append(tan_std, np.std(all_tan_vels_3))
radial_avg = np.append(radial_avg, np.mean(all_rad_vels_3))
radial_std = np.append(radial_std, np.std(all_rad_vels_3))

#avg properties after step 4
all_rad_vels_4 = np.empty((0), dtype=int)
all_tan_vels_4 = np.empty((0), dtype=int)
all_distances_4 = np.empty((0), dtype=int)
all_masses_4 = np.empty((0), dtype=int)
all_other_mass_4 = np.empty((0), dtype=int)
current_ind_halo_A_velocity = sub_subhalo_id[halo_A_velocity]
for count, ii in enumerate(current_ind_halo_A_velocity):
    row_pair = df_pairs_info[df_pairs_info['ID_A']==ii] #objeto tipo serie
    all_rad_vels_4 = np.append(all_rad_vels_4, row_pair['Radial_vel'].values[0])
    all_tan_vels_4 = np.append(all_tan_vels_4, row_pair['Tan_vel'].values[0])
    all_distances_4 = np.append(all_distances_4, row_pair['Distance'].values[0])
    all_masses_4 = np.append(all_masses_4, row_pair['Stellar_massA'].values[0])
    all_masses_4 = np.append(all_masses_4, row_pair['Stellar_massB'].values[0])
    all_other_mass_4 = np.append(all_other_mass_4, row_pair['DM_massA'].values[0])
    all_other_mass_4 = np.append(all_other_mass_4, row_pair['DM_massB'].values[0])

log_mass_4 = np.log10(all_masses_4)
log_other_mass_4 = np.log10(all_other_mass_4)
log_distances_4 = np.log10(all_distances_4)

mass_avg = np.append(mass_avg, np.mean(log_mass_4))
mass_std = np.append(mass_std, np.std(log_mass_4))
other_mass_avg = np.append(other_mass_avg, np.mean(log_other_mass_4))
other_mass_std = np.append(other_mass_std, np.std(log_other_mass_4))
distance_avg = np.append(distance_avg, np.mean(log_distances_4))
distance_std = np.append(distance_std, np.std(log_distances_4))
tan_avg = np.append(tan_avg, np.mean(all_tan_vels_4))
tan_std = np.append(tan_std, np.std(all_tan_vels_4))
radial_avg = np.append(radial_avg, np.mean(all_rad_vels_4))
radial_std = np.append(radial_std, np.std(all_rad_vels_4))

formation_time_avg = np.empty((0), dtype=int)
formation_time_std = np.empty((0), dtype=int)

z1 = z_half_mass[np.concatenate([halo_A, halo_B])]
formation_time_avg = np.append(formation_time_avg, np.mean(z1[z1>0]))
formation_time_std = np.append(formation_time_std, np.std(z1[z1>0]))
z2 = z_half_mass[np.concatenate([halo_A_pair_thresh, halo_B_pair_thresh])]
formation_time_avg = np.append(formation_time_avg, np.mean(z2[z2>0]))
formation_time_std = np.append(formation_time_std, np.std(z2[z2>0]))
z3 = z_half_mass[np.concatenate([halo_A_isolated, halo_B_isolated])]
formation_time_avg = np.append(formation_time_avg, np.mean(z3[z3>0]))
formation_time_std = np.append(formation_time_std, np.std(z3[z3>0]))
z4 = z_half_mass[np.concatenate([halo_A_velocity, halo_B_velocity])]
formation_time_avg = np.append(formation_time_avg, np.mean(z4[z4>0]))
formation_time_std = np.append(formation_time_std, np.std(z4[z4>0]))

path_to_props = path_to_save + 'avgProps/'
makedir(path_to_props)
#save avg props
np.save(path_to_props + sample + '_avgMass.npy', mass_avg)
np.save(path_to_props + sample + '_stdMass.npy', mass_std)

np.save(path_to_props + sample + '_avgOtherMass.npy', other_mass_avg)
np.save(path_to_props + sample + '_stdOtherMass.npy', other_mass_std)

np.save(path_to_props + sample + '_avgDist.npy', distance_avg)
np.save(path_to_props + sample + '_stdDist.npy', distance_std)

np.save(path_to_props + sample + '_avgTanVel.npy', tan_avg)
np.save(path_to_props + sample + '_stdTanVel.npy', tan_std)

np.save(path_to_props + sample + '_avgRadVel.npy', radial_avg)
np.save(path_to_props + sample + '_stdRadVel.npy', radial_std)

np.save(path_to_props + sample + '_avgFormTime.npy', formation_time_avg)
np.save(path_to_props + sample + '_stdFormTime.npy', formation_time_std)

