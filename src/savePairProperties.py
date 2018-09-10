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
import os
import matplotlib
matplotlib.use('Agg')

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
                    help='define folder to save')

args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
path_to_save = '/u/cgomez11/' + args.folder + '/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'

sample = args.sample_type
sub_subhalo_id = np.load(path_to_save + sample + '_subhalos_ID_afterMass.npy')
snapshot = 99
header = loadHeader(basePath, snapshot)
BoxSize = header['BoxSize']

#load subhalos catalog
fields_subhalos = ['SubhaloMassType', 'SubhaloMass', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassInRadType','SubhaloGrNr']
subhalos = loadSubhalos(basePath,snapshot, fields=fields_subhalos)
#define positions, velocities
pos_ckpc = subhalos['SubhaloPos'] #* (1 / 0.704)
vel_km_s = subhalos['SubhaloVel']
parent_fof = subhalos['SubhaloGrNr']

#load halos (group) catalog
fields_halos = ['Group_M_Crit200', 'GroupPos', 'GroupVel', 'GroupMassType', 'Group_M_Mean200','GroupFirstSub']
halos = loadHalos(basePath, snapshot, fields= fields_halos)
central_subhalos = halos['GroupFirstSub'][parent_fof] #solo me importan los groups parents de subhalos
is_central = np.zeros(parent_fof.size) #tantos como subhalos
is_central[central_subhalos] = 1

#define mass
subhalo_stellar_mass = subhalos['SubhaloMassInRadType'][:,4] * 1e10 / 0.704
subhalo_M_Crit200 = halos['Group_M_Crit200'][parent_fof] * 1e10 / 0.704 #subhalo_M_Crti200

#define mass condition: poder indexar propiedades
if sample == 'stellar':
	stellar_min = 1e10
	stellar_max= 1.5e11
	mass_condition = (subhalo_stellar_mass>stellar_min) & (subhalo_stellar_mass<stellar_max)
elif sample == 'DM':
	DM_min = 1.0e11
	DM_max = 1e12
	mass_condition = (subhalo_M_Crit200>DM_min) & (subhalo_M_Crit200<DM_max) & (is_central==1) & (subhalo_stellar_mass!=0) 
else: #combined sample
	stellar_min = 1e10
	stellar_max= 1e11
	DM_min = 1e11
	DM_max = 1e13
	mass_condition = (subhalo_M_Crit200>DM_min) & (subhalo_M_Crit200<DM_max) & (subhalo_stellar_mass>stellar_min) & (subhalo_stellar_mass<stellar_max) &(is_central==1)

n_S = np.sum(mass_condition)
sub_stellar_mass = subhalo_stellar_mass[mass_condition]
sub_DM_mass = subhalo_M_Crit200[mass_condition]
print('Number of galaxies that satisfy the mass criterion: ', n_S)
sub_pos_ckpc = pos_ckpc[mass_condition]
sub_vel_km_s = vel_km_s[mass_condition]
sub_parent_fof = parent_fof[mass_condition]

#load distances and halo_A/B ids
distances = np.load(path_to_save + sample + '_neighbor_distances.npy')
arr_id = np.load(path_to_save + sample + '_pair_ids.npy')
halo_A = arr_id[:,0]
halo_B = arr_id[:,1]

ID_A = sub_subhalo_id[halo_A].tolist()
ID_B = sub_subhalo_id[halo_B].tolist()
stellar_mass_A = sub_stellar_mass[halo_A].tolist()
stellar_mass_B = sub_stellar_mass[halo_B].tolist()
DM_mass_A = sub_DM_mass[halo_A].tolist()
DM_mass_B = sub_DM_mass[halo_B].tolist()
dist = distances[halo_A].tolist() #igual para A y B
pos_A = sub_pos_ckpc[halo_A].tolist()
pos_B = sub_pos_ckpc[halo_B].tolist()
GrNb_A = sub_parent_fof[halo_A].tolist()
GrNb_B = sub_parent_fof[halo_B].tolist()

rad_vel = np.empty((0), dtype=int)
tan_vel = np.empty((0), dtype=int)
rad_vel, tan_vel = find_all_velocities(halo_A, halo_B, rad_vel, tan_vel, sub_stellar_mass,
sub_vel_km_s, sub_pos_ckpc)

rad_vel = rad_vel.tolist()
tan_vel = tan_vel.tolist()
#round values
stellar_mass_A = list(map(lambda x: round(x,4), stellar_mass_A))
stellar_mass_B = list(map(lambda x: round(x,4), stellar_mass_B))
DM_mass_A = list(map(lambda x: round(x,4), DM_mass_A))
DM_mass_B = list(map(lambda x: round(x,4), DM_mass_B))
dist = list(map(lambda x: round(x,4), dist))
pos_A = list(map(lambda x: [round(y,4) for y in x], pos_A))
pos_B = list(map(lambda x: [round(y,4) for y in x], pos_B))
rad_vel = list(map(lambda x: round(x,4), rad_vel))
tan_vel = list(map(lambda x: round(x,4), tan_vel))

data = {'ID_A' : ID_A, 'ID_B' : ID_B,
        'Stellar_massA': stellar_mass_A, 'Stellar_massB': stellar_mass_B, 'DM_massA': DM_mass_A, 'DM_massB': DM_mass_B,
        'Distance': dist,'Radial_vel': rad_vel, 'Tan_vel': tan_vel,
        'Position_A':pos_A, 'Position_B': pos_B, 'Parent_A' : GrNb_A, 'Parent_B': GrNb_B}

df_pairs =  pd.DataFrame(data)
df_pairs.to_pickle(path_to_save + sample + '_df_pairCandidatesAfterMass.pkl')
