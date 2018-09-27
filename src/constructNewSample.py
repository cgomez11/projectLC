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
parser.add_argument('--save_id', type=str, default='none',
                    help='define the crtierion step to visualize')
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')

args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
n = 10
stellar_min = 1e10
stellar_max= 1e11
DM_min = 1e11
DM_max = 1e13
sample = args.sample_type
snapshot = 99
header = loadHeader(basePath, snapshot)
BoxSize = header['BoxSize'] #kpc

#based on the sample type: create the catalogue
#load subhalos catalogue
fields_subhalos = ['SubhaloMassType', 'SubhaloMass', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassInRadType','SubhaloGrNr']
subhalos = loadSubhalos(basePath,snapshot, fields=fields_subhalos)
#define positions, velocities
pos_ckpc = subhalos['SubhaloPos'] #* (1 / 0.704)
vel_km_s = subhalos['SubhaloVel']
parent_fof = subhalos['SubhaloGrNr']
subhalo_MCrit200_b = subhalos['SubhaloMass']
subhalo_ID = np.arange(len(parent_fof))

#load halos (group) catalogue
fields_halos = ['Group_M_Crit200', 'GroupPos', 'GroupVel', 'GroupMassType', 'Group_M_Mean200','GroupFirstSub']
halos = loadHalos(basePath, snapshot, fields= fields_halos)
central_subhalos = halos['GroupFirstSub'][parent_fof] #solo me importan los groups parents de subhalos
is_central = np.zeros(parent_fof.size) #tantos como subhalos
is_central[central_subhalos] = 1
subhalo_stellar_mass = subhalos['SubhaloMassInRadType'][:,4] * 1e10 / 0.704
subhalo_M_Crit200 = halos['Group_M_Crit200'][parent_fof] * 1e10 / 0.704 #subhalo_M_Crti200
print('Original number of galaxies: ', len(subhalo_stellar_mass))
"""
if sample == 'stellar':
    #mass to aply the constraint
    mass_msun = subhalo_stellar_mass
    print('Original number of galaxies: ', len(mass_msun))
    mass_condition = (mass_msun>stellar_min) & (mass_msun<stellar_max)
    

elif sample == 'DM':
    #define M_crit200 of subhalos
    mass_msun = subhalo_M_Crit200
    print('Original number of galaxies: ', len(mass_msun))
    mass_condition = (mass_msun>DM_min) & (mass_msun<DM_max) & (is_central==1) & (subhalo_stellar_mass!=0)
"""
#one big mass constraint: to capture all the possible candidates but reduce the search sample
mass_condition = (subhalo_M_Crit200>DM_min) & (subhalo_M_Crit200<DM_max) & (subhalo_stellar_mass>stellar_min) & (subhalo_stellar_mass<stellar_max) &(is_central==1)

    
#apply firts condition: mass constraint    
n_S = np.sum(mass_condition)
sub_stellar_mass = subhalo_stellar_mass[mass_condition]
sub_DM_mass = subhalo_M_Crit200[mass_condition]
print('Number of galaxies that satisfy the mass criterion: ', n_S)
sub_pos_ckpc = pos_ckpc[mass_condition]
sub_vel_km_s = vel_km_s[mass_condition]
sub_parent_fof = parent_fof[mass_condition]
sub_subhalo_ID = subhalo_ID[mass_condition] #save the indices of the subhalos that survive to the mass criterion: se puede indexar con los indices en cada crit


#copy files
S_pad_pos = sub_pos_ckpc.copy()
S_pad_vel = sub_vel_km_s.copy()
#S_pad_stellar_mass = sub_mass_msun.copy()
S_pad_fof = sub_parent_fof.copy()
S_pad_id = np.arange(n_S) #llevar la cuenta de donde ya se hizo padding

#apply padding: sin modificar las unidades de la posicion
for i in (0,1,-1):
    for j in (0,1,-1):
        for k in (0,1,-1):
            new_pos = sub_pos_ckpc.copy()
            if(i):
                new_pos[:,0] = new_pos[:,0] + i*BoxSize
            if(j):
                new_pos[:,1] = new_pos[:,1] + j*BoxSize
            if(k):
                new_pos[:,2] = new_pos[:,2] + k*BoxSize
                
            if((i!=0) | (j!=0) | (k!=0)):
                S_pad_pos = np.append(S_pad_pos, new_pos, axis=0)
                S_pad_vel = np.append(S_pad_vel, sub_vel_km_s, axis=0)
                #S_pad_stellar_mass = np.append(S_pad_stellar_mass, sub_mass_msun)
                S_pad_id = np.append(S_pad_id, np.arange(n_S))
                S_pad_fof = np.append(S_pad_fof, sub_parent_fof)

#encontrar el par de cada uno
from sklearn.neighbors import NearestNeighbors
#only ask for the two closest neighbors: itself and the closest 
#se le pasan las posiciones con el padding
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(S_pad_pos)
nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(S_pad_pos)
distances, indices = nbrs.kneighbors(S_pad_pos)

my_closest = indices[:,1]
resting_neighbors = indices[:,2:]
#list to save one of the members of the pair
list_index = []

list_index_pairs = []
list_pairs = []

#arrays to save pairs
halo_A = np.empty((0), dtype=int)
halo_B = np.empty((0), dtype=int)

#arrays to save pairs after threshold distance
halo_A_pair_thresh = np.empty((0), dtype=int)
halo_B_pair_thresh = np.empty((0), dtype=int)

#define specific mass constraints 
sigma_stellar = 0.57*n
sigma_halo = 0.3*n
#mass A is the less massive object of the pair
mass_A_stellar_min = (5.43-sigma_stellar)*1e10
mass_A_stellar_max = (5.43+sigma_stellar)*1e10
mass_B_stellar_min = (10.86-sigma_stellar)*1e10
mass_B_stellar_max = (10.86+sigma_stellar)*1e10

mass_A_DM_min = (1.3-sigma_halo)*1e12
mass_A_DM_max = (1.3+sigma_halo)*1e12
mass_B_DM_min = (2.6-sigma_halo)*1e12
mass_B_DM_max = (2.6+sigma_halo)*1e12

halo_A_mass = np.empty((0), dtype=int)
halo_B_mass = np.empty((0), dtype=int)

#ojo: no se itera sobre todas las distancias: hasta el numero de elementos de mi muestra
sigma_distance = 25
min_distance = 787 - sigma_distance*n
max_distance = 787 + sigma_distance*n
for ee in range(n_S):
    #my_closest[ee] es index
    if ee == my_closest[my_closest[ee]]% n_S: #puede que lo haya encontrado en su forma de padding
        other = my_closest[ee] % n_S
        each_distance = distances[ee,1]
        if (ee not in list_index) and (other not in list_index):
            list_index.append(ee)
            halo_A = np.append(halo_A, int(ee))
            halo_B = np.append(halo_B, int(other))
            #if (not (other in halo_A)) & (not (ee in halo_B)):
            #### check mass constraint ####
            #find the min stellar and DM mass
            stellar_mass_ee = sub_stellar_mass[ee]
            stellar_mass_other = sub_stellar_mass[other]
            DM_mass_ee = sub_DM_mass[ee]
            DM_mass_other = sub_DM_mass[other]
            if stellar_mass_ee < stellar_mass_other:
                A_stellar_mass = stellar_mass_ee
                B_stellar_mass = stellar_mass_other
            else:
                A_stellar_mass = stellar_mass_other
                B_stellar_mass = stellar_mass_ee

            if DM_mass_ee < DM_mass_other:
                A_DM_mass = DM_mass_ee
                B_DM_mass = DM_mass_other
            else:
                A_DM_mass = DM_mass_other
                B_DM_mass = DM_mass_ee
            #print('stellar masses:', A_stellar_mass, B_stellar_mass)
            #print('DM masses:', A_DM_mass, B_DM_mass)
            #if (A_stellar_mass>mass_A_stellar_min) and (A_stellar_mass<mass_A_stellar_max) and (B_stellar_mass>mass_B_stellar_min) and 
#(B_stellar_mass<mass_B_stellar_max) and (A_DM_mass>mass_A_DM_min) and (A_DM_mass<mass_A_DM_max) and (B_DM_mass>mass_B_DM_min) and (B_DM_mass<mass_B_DM_max) and 
#(A_stellar_mass+B_stellar_mass > mass_A_stellar_min+mass_B_stellar_min) and (A_stellar_mass+B_stellar_mass < mass_A_stellar_max+mass_B_stellar_max) and 
#(A_DM_mass+B_DM_mass > mass_A_DM_min+mass_B_DM_min) and (A_DM_mass+B_DM_mass < mass_A_DM_max+mass_B_DM_max):
            if ((A_DM_mass+B_DM_mass) > (mass_A_DM_min+mass_B_DM_min)) and ((A_DM_mass+B_DM_mass) <( mass_A_DM_max+mass_B_DM_max)) and ((A_stellar_mass+B_stellar_mass) > (mass_A_stellar_min+mass_B_stellar_min)) and ((A_stellar_mass+B_stellar_mass) < (mass_A_stellar_max+mass_B_stellar_max)):
                halo_A_mass = np.append(halo_A_mass, int(ee))
                halo_B_mass = np.append(halo_B_mass, int(other))
                #check the distance contraint
                if each_distance/0.704 > min_distance and each_distance/0.704 <max_distance: #changes depending on the number of sigmas
                    list_index_pairs.append(ee)
                    list_pairs.append([ee, other])
                    halo_A_pair_thresh = np.append(halo_A_pair_thresh, int(ee))
                    halo_B_pair_thresh = np.append(halo_B_pair_thresh, int(other))

max_stellar_pairs = np.max(sub_stellar_mass[np.concatenate([halo_A, halo_B], axis=0)])
min_stellar_pairs = np.min(sub_stellar_mass[np.concatenate([halo_A, halo_B], axis=0)])

max_DM_pairs = np.max(sub_DM_mass[np.concatenate([halo_A, halo_B], axis=0)])
min_DM_pairs = np.min(sub_DM_mass[np.concatenate([halo_A, halo_B], axis=0)])

print('ranges stellar mass for pair cands: ', min_stellar_pairs, max_stellar_pairs)
print('ranges DM mass for pair cands: ', min_DM_pairs, max_DM_pairs)

print('Number of pairs based on mutual proximity', np.shape(halo_B))
print('Number of pairs in the mass range: ', halo_A_mass.shape)
print('Number of pairs based on the threshold distance: ', len(list_pairs))
array_index_pairs = np.asarray(list_index_pairs) #con el indice de la muestra basta para recuperar distancias
#dABs = distances[array_index_pairs, :] #en 1 estan las closest distances

from scipy.spatial import distance

list_not_discard_pairs = []
list_index_not_discard_pairs = []

#apply isolation criterion
t1 = time.time()
#isolation criterion 
halo_A_isolated = np.empty((0), dtype=int)
halo_B_isolated = np.empty((0), dtype=int)
n_pairs = 0
for cc, pair in enumerate(list_pairs):
    #sample mass son todas las masas
    mass_A = sub_stellar_mass[pair[0]]
    mass_B = sub_stellar_mass[pair[1]]
    if mass_A < mass_B:
        min_mass = mass_A
    else:
        min_mass = mass_B
    #thresh_distance = dABs[cc,1] #la distancia entre cada par
    thresh_distance = 2500-200*n #kpc 
    #iterar sobre los otros vecinos
    cont_no = 0
    for jj in range(np.shape(resting_neighbors)[1]):
        #compare closest neighbors to both components of the pair
        #las columnas de indices contiene los 20 vecinos mas cercanos
        neighbor_compare1 = indices[pair[0], jj+2] % n_S
        distance_1 = distances[pair[0], jj+2] /0.704 #kpc
        neighbor_compare2 = indices[pair[1], jj+2] % n_S
        distance_2 = distances[pair[1], jj+2]/0.704 #kpc
        
        if ((sub_stellar_mass[neighbor_compare1] >= min_mass and distance_1<=3*thresh_distance) or
            (sub_stellar_mass[neighbor_compare2] >= min_mass and distance_2<=3*thresh_distance)):
            cont_no += 1
    if cont_no ==0: #ninguno de los vecinos esta muy muy cerca
        halo_A_isolated = np.append(halo_A_isolated, pair[0])
        halo_B_isolated = np.append(halo_B_isolated, pair[1])
        list_index_not_discard_pairs.append(cc)
        list_not_discard_pairs.append(pair)        
   
tend = time.time()
#print('Elapsed time in the isolation criterion: ', tend-t1)    
#calculate velocities of the isolated pairs
rad_vel_isolated = np.empty((0), dtype=int)
tan_vel_isolated = np.empty((0), dtype=int)
for kk, ids in enumerate(halo_A_isolated):
    mass_A = sub_stellar_mass[ids]
    mass_B = sub_stellar_mass[halo_B_isolated[kk]]
    if mass_A < mass_B:
        ref_obj = ids
        other_obj = halo_B_isolated[kk]
    else: 
        ref_obj = halo_B_isolated[kk]
        other_obj = ids

    vel_ref = sub_vel_km_s[ref_obj,:]
    vel_other = sub_vel_km_s[other_obj,:] 
    
    pos_ref = sub_pos_ckpc[ref_obj,:]
    pos_other = sub_pos_ckpc[other_obj,:]

    VB_para, VB_perp = compute_velocities(r1=pos_ref, r2=pos_other, v1=vel_ref, v2=vel_other)
    rad_vel_isolated = np.append(rad_vel_isolated, VB_para)
    tan_vel_isolated = np.append(tan_vel_isolated, VB_perp)    

print('Number of pairs based on their neighborhood: ', len(list_not_discard_pairs))
halo_A_tangential = np.empty((0), dtype=int)
halo_B_tangential = np.empty((0), dtype=int)

halo_A_velocity = np.empty((0), dtype=int)
halo_B_velocity = np.empty((0), dtype=int)
final_pairs = []
vel_rel = []
vel_tan=[]
for cc, pair in enumerate(list_not_discard_pairs):
    mass_A = sub_stellar_mass[pair[0]]
    mass_B = sub_stellar_mass[pair[1]]
    if mass_A < mass_B:
        ref_obj = pair[0]
        other_obj = pair[1]
    else: 
        ref_obj = pair[1]
        other_obj = pair[0]
    #compute relative velocity
    vel_ref = sub_vel_km_s[ref_obj,:]
    vel_other = sub_vel_km_s[other_obj,:] 
    
    pos_ref = sub_pos_ckpc[ref_obj,:]
    pos_other = sub_pos_ckpc[other_obj,:] 
    #subtract vectors
    rAB = pos_other - pos_ref 
    #rAB = (rAB / 0.704) #* 3.0857e16#ckpc-> km
    VB = vel_other - vel_ref
    #find the parallel component of VB'
    VB_para = np.dot(rAB, VB)/np.linalg.norm(rAB)
    #find the tangential component of VB
    mycos = np.dot(rAB, VB)/(np.linalg.norm(rAB)*np.linalg.norm(VB))
    VB_perp = np.linalg.norm(VB)*np.sin(math.acos(mycos))
    #si en algun componente es negativo?
    #print(VB_para)
    min_vr = -123-4*n
    max_vr = -123+4*n
    if VB_para <max_vr and VB_para >min_vr: #todos los componentes de la velocidad cumplen el criterio
        halo_A_velocity = np.append(halo_A_velocity, pair[0])
        halo_B_velocity = np.append(halo_B_velocity, pair[1])
        final_pairs.append(pair)
        vel_rel.append(VB_para)
        vel_tan.append(VB_perp)

        #ya cumple con criterio de velocidad radial: verificar vel tangencial
        if VB_perp >0 and VB_perp<=22*n:
            halo_A_tangential = np.append(halo_A_tangential, pair[0]) 
            halo_B_tangential = np.append(halo_B_tangential, pair[1])
    
print('Number of pairs based on velocity criterion: ', len(final_pairs))
print('Number of pairs after tangential velocity constraint: ', halo_A_tangential.shape)
list_all_pairs =[]
list_all_distances = []
list_all_posA = []
list_all_posB = []
list_all_velA = []
list_all_velB = []
list_all_massA = []
list_all_massB = []
for member in final_pairs:
    list_all_pairs.append(member)
    list_all_distances.append(distances[member[0],1]/0.704)
    list_all_posA.append(sub_pos_ckpc[member[0],:]/0.704)
    list_all_posB.append(sub_pos_ckpc[member[1],:]/0.704)
    list_all_velA.append(sub_vel_km_s[member[0],:])
    list_all_velB.append(sub_vel_km_s[member[1],:])
    #list_all_massA.append(sub_mass_msun[member[0]])
    #list_all_massB.append(sub_mass_msun[member[1]])
    
data = {'Pair_Index' : list_all_pairs, 'Distance' : list_all_distances,
        'Position_A' : list_all_posA, 'Position_B': list_all_posB, 
        'Vel_A': list_all_velA, 'Vel_B': list_all_velB, 
        'Mass_A': list_all_massA, 'Mass_B': list_all_massB, 'Radial_vel':vel_rel, 'Tan_vel': vel_tan}
#df_pairs =  pd.DataFrame(data)
path_to_save = '/u/cgomez11/newSample/illustrisTNG/n_' + str(n)  + '/L' + args.res + 'n'+ args.parts + 'TNG/' 
makedir(path_to_save)
#df_pairs.to_pickle(path_to_save + sample + '_df_illustrisTNG_pad.pkl')

if args.save_id != 'none':
    #df_pairs.to_pickle(path_to_save + sample + '_df_illustrisTNG_pad.pkl')
    #save all indices at each step
    #concatenate ids into a single matrix
    np.save(path_to_save + sample + '_subhalos_ID_afterMass.npy', sub_subhalo_ID)
    arr_id = np.concatenate([np.expand_dims(halo_A,axis=1), np.expand_dims(halo_B, axis=1)], axis=1)
    arr_masses = np.concatenate([np.expand_dims(halo_A_mass,axis=1), np.expand_dims(halo_B_mass, axis=1)], axis=1)
    arr_pair_thresh = np.concatenate([np.expand_dims(halo_A_pair_thresh,axis=1), np.expand_dims(halo_B_pair_thresh,axis=1)], axis=1)
    arr_isolated = np.concatenate([np.expand_dims(halo_A_isolated,axis=1), np.expand_dims(halo_B_isolated,axis=1)], axis=1)
    arr_velocity = np.concatenate([np.expand_dims(halo_A_velocity,axis=1), np.expand_dims(halo_B_velocity,axis=1)], axis=1)
    arr_tangential = np.concatenate([np.expand_dims(halo_A_tangential,axis=1), np.expand_dims(halo_B_tangential,axis=1)], axis=1)
    #save mass indices
    np.save(path_to_save + sample + '_pair_ids.npy', arr_id) #all pair candidates in the preliminary mass constraint
    np.save(path_to_save + sample + '_pair_mass_ids.npy', arr_masses) #pair candidates within the combined mass range
    np.save(path_to_save + sample + '_thresh_ids.npy', arr_pair_thresh) #pair candidates at the thresh distance
    np.save(path_to_save + sample + '_isolated_ids.npy', arr_isolated) #pair candidates that are isolated
    np.save(path_to_save + sample + '_velocity_ids.npy', arr_velocity)
    np.save(path_to_save + sample + '_tangential_ids.npy', arr_tangential)
    np.save(path_to_save + sample + '_neighbor_distances.npy', distances[:,1]) #solo la distancia al mas cercano 

path_to_save_ims = path_to_save + 'images/'
makedir(path_to_save_ims)