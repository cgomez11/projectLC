#script to define auxiliary functions 
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ioff()

def plot_mass_hist (n_fig, hA, hB, step, sam_type):
    a = plt.hist(np.log10(sub_mass_msun[hA] + sub_mass_msun[hB]), bins=n_bins, alpha=0.5)
    plt.xlabel('log ' +sam_type+' Mass/[$M_\odot$]')
    plt.ylabel('Count')
    plt.title( sam_type + ' mass distribution of pair candidates after ' + step)
    #plt.legend()
    n_fig.savefig(path_to_save_ims + sam_type +'_MassDistr_' + step  + '.pdf')
    plt.close(n_fig)

def plot_positions(n_fig, hA, hB, step, sam_type):
    ax = n_fig.add_subplot(111)#, projection='3d')
    #for cc, ii in enumerate(hA):
    #    posA = sub_pos_ckpc[ii,:]
    #    posB = sub_pos_ckpc[hB[cc],:]
    #    ax.scatter(posA[0], posA[1], posA[2], c='b', marker='x')
    #    ax.scatter(posB[0], posB[1], posB[2], c='g', marker='x')
    #define N as the sum of masses of the pair
    colors = np.log10(sub_mass_msun[hA] + sub_mass_msun[hB])
    #N = np.shape(hA)[0]
    #colors = np.random.rand(N)
    cax = ax.scatter(sub_pos_ckpc[hA,0], s=7,c=colors, marker = 'x', cmap='jet')
    cax = ax.scatter(sub_pos_ckpc[hB,0], s=7,c=colors, marker = 'x', cmap='jet')
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    #ax.set_zlabel('Z Label')
    ax.set_title(sam_type + ' position distribution of pair candidates after '+ step)
    cbar = n_fig.colorbar(cax, ax=ax, ticks=[np.min(colors), np.max(colors)], orientation='vertical')
    cbar.ax.set_xticklabels([str(np.min(colors)), str(np.max(colors))])
    n_fig.savefig(path_to_save_ims + sam_type +'_PosDistr_' + step  + '.pdf')
    plt.close(n_fig)

def plot_dist_hist(n_fig, hA, hB, step, sam_type):
    a = plt.hist(distances[hA,1]/0.704, bins=n_bins, alpha=0.5) #same distance from B to A
    plt.xlabel('Distance [ckpc]')
    plt.ylabel('Count')
    plt.title(sam_type + ' distance distribution of pair candidates after ' + step)
    n_fig.savefig(path_to_save_ims + sam_type +'_DistanceDistr_' + step  + '.pdf')
    plt.close(n_fig)

def compute_velocities(r1, r2, v1, v2): #r1 and v1 are references
    #subtract vectors
    rAB = r2 - r1
    rAB = (rAB / 0.704) 
    
    VB = v2 - v1
    #find the parallel component of VB'
    VB_para = np.dot(rAB, VB)/np.linalg.norm(rAB)
    #find the tangential component of VB
    mycos = np.dot(rAB, VB)/(np.linalg.norm(rAB)*np.linalg.norm(VB))
    VB_perp = np.linalg.norm(VB)*np.sin(math.acos(mycos))
    
    return VB_para, VB_perp
    
def find_all_velocities(hA, hB, l_rad_vel, l_tan_vel, sub_mass_msun, sub_vel_km_s, sub_pos_ckpc):
    for kk, ids in enumerate(hA):
        mass_A = sub_mass_msun[ids]
        mass_B = sub_mass_msun[hB[kk]]
        if mass_A < mass_B:
            ref_obj = ids
            other_obj = hB[kk]
        else:
            ref_obj = hB[kk]
            other_obj = ids

        vel_ref = sub_vel_km_s[ref_obj,:]
        vel_other = sub_vel_km_s[other_obj,:]

        pos_ref = sub_pos_ckpc[ref_obj,:]
        pos_other = sub_pos_ckpc[other_obj,:]

        VB_para, VB_perp = compute_velocities(r1=pos_ref, r2=pos_other, v1=vel_ref, v2=vel_other)
        l_rad_vel = np.append(l_rad_vel, VB_para)
        l_tan_vel = np.append(l_tan_vel, VB_perp)
    return l_rad_vel, l_tan_vel
