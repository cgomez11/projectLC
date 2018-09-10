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

def makedir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

parser = argparse.ArgumentParser(description='Define parameters')
parser.add_argument('--res', type=str, default='75',
                    help='define the resolution of TNG')
parser.add_argument('--parts', type=str, default='1820',
                    help='define the number of particles')
parser.add_argument('--save_id', type=str, default='none',
                    help='define the crtierion step to visualize')
parser.add_argument('--sample_type', type=str, default='stellar',
                    help='define the sample to create')
parser.add_argument('--folder', type=str, default='newStrategy',
                    help='define the folder to save the numbers')
parser.add_argument('--n', type=str, default='newStrategy',
                    help='define number of std')

args = parser.parse_args()
basePath = '/virgo/simulations/IllustrisTNG/L' + args.res + 'n' + args.parts + 'TNG/output/'
snapshot = 99
sample = args.sample_type
if sample == 'comb':
    path_to_save = '/u/cgomez11/newSample/illustrisTNG/n_' + args.n  + '/L' + args.res + 'n'+ args.parts + 'TNG/'
else:
    path_to_save = '/u/cgomez11/newStrategy/illustrisTNG/L' + args.res + 'n'+ args.parts + 'TNG/'

#load
zs = np.load(path_to_save + 'z_' + args.res + '_' + args.parts + '.npy') #en orden ascendente
sub_subhalo_id = np.load(path_to_save + sample + '_subhalos_ID_afterMass.npy')
print('Subsample size: ', sub_subhalo_id.shape[0])
fields = ['SubhaloMass','SubfindID','SnapNum', 'SubhaloMassInRadType']
#z_all_sub_subhalos = np.empty((0), dtype=int)
z_all_sub_subhalos = []
all_z_max_mass = []
all_snapshot_max_mass = []
snapshot_all_sub_subhalos = []
true_z = []
t1 = time.time()
for count, each_id in enumerate(sub_subhalo_id):
    flag = False
    tree = loadTree(basePath, snapshot, each_id,fields=fields,onlyMPB=True)
    #check that the tree is not empty
    if tree is not None:
        if sample == 'stellar': #si existe el subhalo 
            all_masses = tree['SubhaloMassInRadType'][:,4] #la masa del ultimo snapshot esta de primera 
        else: #comb y DM usar la del halo
            all_masses = tree['SubhaloMass'] #la masa del ultimo snapshot esta de primera 
        #smooth the mass function
        #all_masses = np.array(sp.signal.medfilt(all_masses, [3]))
        all_masses = np.convolve(all_masses,[1/3,1/3,1/3], 'same')
        snap_num = tree['SnapNum'] #del mas reciente: se puede usar para indexar 
        half_mass = np.max(all_masses)/2
        pos_max_mass = np.argmax(all_masses)
        snapshot_max_mass = snap_num[pos_max_mass]
        z_max_mass = zs[snapshot_max_mass]
        all_z_max_mass.append(z_max_mass)
        all_snapshot_max_mass.append(snapshot_max_mass)
        #1) apply smoothing to the mass function
        #2) go backwards: preguntarle a todas las masas
        #si recorro all_masses empiezo en el snapshot mas reciente
        for cc, each_mass in enumerate(all_masses):
            #begin asking to the mass at the present
            if each_mass <= half_mass:
                pos_1_half_snap = snap_num[cc]
                pos_1_half_z = zs[pos_1_half_snap]
                mass_pos_1 = each_mass
                pos_2_half_snap = snap_num[cc-1]
                pos_2_half_z = zs[pos_2_half_snap]
                mass_pos_2 = all_masses[cc-1]
                #define x and y points
                ys = [mass_pos_1, mass_pos_2]
                xs_z = [pos_1_half_z, pos_2_half_z]
                xs_snap = [pos_1_half_snap, pos_2_half_snap]
                #define the interpolation
                coefficients_z = np.polyfit(xs_z, ys, 1)
                coefficients_snap = np.polyfit(xs_snap, ys, 1)
                z_half_mass = (half_mass - coefficients_z[1])/coefficients_z[0] #x=(y-b)/a
                snap_half_mass = (half_mass - coefficients_snap[1])/coefficients_snap[0] #x=(y-b)/a
                z_all_sub_subhalos.append(z_half_mass)
                #snapshot_all_sub_subhalos.append(int(round(snap_half_mass,0)))
                #print(type(snapshot_all_sub_subhalos))
                if not math.isnan(snap_half_mass): #verificar que no sea nan el snap
                    if round(snap_half_mass,0) >0 and round(snap_half_mass,0)<=99:
                        true_z.append(zs[int(round(snap_half_mass,0))])
                        snapshot_all_sub_subhalos.append(int(round(snap_half_mass,0)))
                    else: #ya garantice que snapshot no es Nan
                        true_z.append(0)
                        snapshot_all_sub_subhalos.append(int(round(snap_half_mass,0)))
                else: #si es nan: deberia quedar un z negativo
                    true_z.append(0)
                    snapshot_all_sub_subhalos.append(0)
                flag = True
                break #no hay que buscar mas: salirse del for
        if flag == False: #nunca encontro la middle mass
        #nunca se alcanza la middle mass?
            z_all_sub_subhalos.append(0)
            snapshot_all_sub_subhalos.append(0)
            true_z.append(0)
    else:
        all_z_max_mass.append(0)
        all_snapshot_max_mass.append(0)
        z_all_sub_subhalos.append(0)
        snapshot_all_sub_subhalos.append(0)
        true_z.append(0)
print('shape subhalos: ', sub_subhalo_id.shape)
print('shape snapshot mas: ', len(all_snapshot_max_mass))
print('shape z max: ', len(all_z_max_mass))
print('shape z found: ', len(z_all_sub_subhalos))
print('shape snapshot: ', len(snapshot_all_sub_subhalos))
print('shape true z:', len(true_z))
position = np.arange(sub_subhalo_id.shape[0]) #define position in the sub_subhalo array
#save results
all_formation_times = np.stack((position, sub_subhalo_id, all_snapshot_max_mass,all_z_max_mass, z_all_sub_subhalos, snapshot_all_sub_subhalos, true_z), axis=1)
filename_ft = path_to_save + sample +'_FormationTimes.txt'
np.savetxt(filename_ft, all_formation_times, fmt=['%u','%u','%u','%.2e','%.2e','%u','%.2e'], header = 'Pos ID SnapMax ZMax Z Snap true_Z')
#np.savetxt(filename_ft, all_formation_times,  fmt=['%u', '%u'], header = 'Pos ID  Z Snap true_Z')

tend = time.time()
print('Elapsed time:',tend-t1) 

