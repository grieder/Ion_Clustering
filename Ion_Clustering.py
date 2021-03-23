import ase
import ase.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
#from sitator.util.mcl import markov_clustering as mc
from sitator.dynamics import MergeSitesByDynamics #import _markov_clustering as mc
from ase.neighborlist import neighbor_list as nl
import matplotlib.pyplot as plt
import time
from ase import Atoms,Atom

def average_pos(cluster,a):
    ave_pos = np.zeros([3])
    for ind in cluster:
        ave_pos += a.get_positions()[ind]
    cluster_pos = ave_pos / len(cluster)
    return cluster_pos

def count_Li_clusters(traj, f_in, cutoff, min_ions = 2):
    n_clusters = []
    MSBD = MergeSitesByDynamics()
    mc = MSBD._markov_clustering

    with open(f_in, 'r') as f:
        f.readline()
        lat = f.readline().split(' ')[-1]
        lat = float(lat.strip('"\r\n'))
    for ind,a in enumerate(traj):

        a.set_cell([lat,lat,lat])
        a.set_pbc([True,True,True])
        a.wrap()

        il,jl = nl('ij', a, {('Li','Li'):cutoff})


        if il.max() == jl.max():
            index_offset = il.min()
        else:
            print("Error: i_neighborlist indexs != j_neighborlist indexs")


        il -= index_offset
        jl -= index_offset

        if il.max() == jl.max():
            num_Li = il.max() + 1
        else:
            print("Error: i_neighborlist != j_neighborlist")

        ij_matrix = np.zeros(shape=(num_Li,num_Li))
        np.fill_diagonal(ij_matrix, 1)
        #for di,(i,j) in enumerate(zip(il,jl)):
        #    ij_matrix[i,j] += 1. / (d[di] / cutoff)
        for di,(i,j) in enumerate(zip(il,jl)):
            ij_matrix[i,j] += 1.

        clusters_t = mc(ij_matrix,pruning_threshold = 0.00001)

        clusters = []
        for clust in clusters_t:
            new_cluster_list = []
            if len(clust) >= min_ions:
                for Li_ind in clust:
                    new_cluster_list.append(Li_ind+index_offset)
                clusters.append(tuple(new_cluster_list))

        n_clusters.append([ind,len(clusters), clusters])

    return np.asarray(n_clusters,dtype=object)

import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def make_cluster_traj(traj, clusters, out_name):

    cluster_traj = []
    for i in range(len(clusters)):
        frame = Atoms()
        for cluster in clusters[i,2]:
            ave_pos = np.zeros([3])
            if len(cluster) > 16:
                ion = ion_dict[16]
                print(len(cluster))
            else:
                ion = ion_dict[len(cluster)]
            for ind in cluster:
                ave_pos += traj[i].get_positions()[ind]
            cluster_pos = ave_pos / len(cluster)
            frame.append(Atom(ion,position=cluster_pos))
        cluster_traj.append(frame)

    ase.io.write(out_name,cluster_traj)

from scipy.linalg import norm
import sys
sys.path.append('/home/shared/Python_Scripts/lethal_lithium')
from md_analysis import rotations


def cluster_lifetime(clusters,traj,similatiry_cut):
    ## Cluster changes overt time

    cluster_ids = {}
    life_times = []
    cluster_msd = []
    
    ## initialize clusters
    frame_n = 0
    for i, cluster in enumerate(amor_smalldt_clusters[0,2]):
        cell = amor_smalldt_traj[0].cell
        pos = average_pos(cluster,amor_smalldt_traj[0])
        cluster_ids[i] = {'o_cluster':cluster, 'c_cluster':[cluster], 'len' : 1,'o_pos':pos,'msd' : [0]}
    
    for i,frame in enumerate(amor_smalldt_clusters[1:,2]):
        f = i + 1
        ## Check ions are only in one cluster
        flat_ions = list(sum(frame, ()))
        n_repeats = len(flat_ions) - np.unique(flat_ions).shape[0]
        if n_repeats > 0:
            print(frame)
        for cluster in frame:
            found = False
            for c_ids in cluster_ids.keys():
                similarity = set(cluster_ids[c_ids]['o_cluster']) & set(cluster)
                
                if len(similarity) >= similatiry_cut:
                    cluster_ids[c_ids]['len'] += 1
                    cluster_ids[c_ids]['c_cluster'].append(cluster)
                    c_pos = average_pos(cluster,amor_smalldt_traj[f])
                    pos_vector = c_pos-cluster_ids[c_ids]['o_pos']
                    #sd_n = norm(pos_vector)
                    sd_r = norm(rotations.wrap_coords(pos_vector,cell))
                    #if abs(sd_n - sd_r) > 1.:
                    #    print(sd_n,sd_r)
                    cluster_ids[c_ids]['msd'].append(sd_r**2)
                    found = True
            if not found:
                new_ids = np.max(list(cluster_ids.keys())) + 1
                pos = average_pos(cluster,amor_smalldt_traj[f])
                cluster_ids[new_ids] = {'o_cluster':cluster, 'c_cluster':[cluster], 'len' : 1, 'o_pos':pos,'msd' : [0]}
    for c_ids in cluster_ids.keys():
        life_times.append(cluster_ids[c_ids]['len'])
        cluster_msd.append(cluster_ids[c_ids]['msd'])
    return cluster_ids, life_times, cluster_msd

#amor_clusters = count_Li_clusters(amor_traj, amor_in, 3.5, min_ions=6)
#cryst_clusters = count_Li_clusters(cryst_traj, cryst_in, 3.5, min_ions=6)

def decay_calc(life_times,sim,min_cut):
    from scipy.optimize import curve_fit
    
    decay = []
    population = len(life_times)
    for i in range(np.max(life_times)):
        if i == 0:
            decay.append([i,population])
        else:
            population -= life_times.count(i)
            decay.append([i,population])
    decay = np.array(decay)
    np.save('decay_min{}_sim{}.npy'.format(sim,min_cut, decay),decay)
    Ag = decay[0,1]
    A, b = curve_fit(lambda a,t,b: a*np.exp(-b*t),  decay[:,0],  decay[:,1],p0=[Ag,0.035])[0]
    
    print('Sim = {} Min = {}\nDifference in A0 = {}\nHalflife = {}\n'.format(sim,min_cut,Ag-A, (-1/b*np.log(0.5))))
     
    fig, ax = plt.subplots()
    
    ax.plot(decay[:,0],decay[:,1],label='Raw')
    ax.plot(decay[:,0], A*np.exp(-b*decay[:,0]),label='Fit')
    #ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Number of Clusters')


    plt.savefig('decaycurve_min{}_sim{}.png'.format(min_cut,sim))
    return


amor_smalldt_traj = ase.io.read('nvt.smalldt.xyz.xyz', index=':')
for i in range(7,11):
    for j in range((i-3),(i+1)):

        min_ions = i
        similarity = j
        
        with HiddenPrints():
            amor_smalldt_clusters = count_Li_clusters(amor_smalldt_traj, 'nvt.smalldt.xyz.xyz',
                                                  4.0, min_ions=min_ions)
        
        
        cluster_ids, life_times, cluster_msd = cluster_lifetime(amor_smalldt_clusters,amor_smalldt_traj,similarity) 
        decay_calc(life_times,similarity,min_ions)
        #fig, ax = plt.subplots()
        #
        #ax.hist(life_times, bins=25)
        #ax.set_xlabel('Lifetime of Cluster (fs)')
        #ax.set_ylabel('Counts')
        #
        ##plt.show()
        #plt.savefig('cluster_hist_min{}_sim{}.png'.format(min_ions,similarity))
