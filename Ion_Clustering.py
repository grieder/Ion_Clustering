import ase
import ase.io
from ase.neighborlist import neighbor_list as nl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import norm
import sys, os
from os import path


def markov_clustering(transition_matrix,
                      expansion = 2,
                      inflation = 2,
                      pruning_threshold = 0.00001,
                      iterlimit = 100):
    """Compute the Markov Clustering of a graph.
    See https://micans.org/mcl/.

    Because we're dealing with matrixes that are stochastic already,
    there's no need to add artificial loop values.

    Implementation inspired by https://github.com/GuyAllard/markov_clustering
    """

    assert transition_matrix.shape[0] == transition_matrix.shape[1]

    # Check for nonzero diagonal -- self loops needed to avoid div by zero and NaNs
    assert np.count_nonzero(transition_matrix.diagonal()) == len(transition_matrix)

    m1 = transition_matrix.copy()

    # Normalize (though it should be close already)
    m1 /= np.sum(m1, axis = 0)

    allcols = np.arange(m1.shape[1])

    converged = False
    for i in range(iterlimit):
        # -- Expansion
        m2 = np.linalg.matrix_power(m1, expansion)
        # -- Inflation
        np.power(m2, inflation, out = m2)
        m2 /= np.sum(m2, axis = 0)
        # -- Prune
        to_prune = m2 < pruning_threshold
        # Exclude the max of every column
        to_prune[np.argmax(m2, axis = 0), allcols] = False
        m2[to_prune] = 0.0
        # -- Check converged
        if np.allclose(m1, m2):
            converged = True
            break

        m1[:] = m2

    if not converged:
        raise ValueError("Markov Clustering couldn't converge in %i iterations" % iterlimit)

    # -- Get clusters
    attractors = m2.diagonal().nonzero()[0]

    clusters = set()

    for a in attractors:
        cluster = tuple(m2[a].nonzero()[0])
        clusters.add(cluster)

    return list(clusters)

def check_lattice(traj,test=None):

    if type(traj) == ase.atoms.Atoms:
        return all(traj.cell.diagonal() > 0.)
    else:
        try:
            return all(traj[0].cell.diagonal() > 0.)
        except:
            print("Trajectory must be Ase Atoms object or list of Ase Atoms objects")

def count_Li_clusters(traj, cutoff, min_ions = 2,lattice_parameters=None):

    """ 
    Cluster ions in a trajectory using Markov Clustering.
    Transition Matrix generated from neighboring ions
    
    Parameters:
        traj : list of ASE Atoms objects 
            The trajectory for analysis 
        
        cutoff : float
            Distance in angstroms for ion to be 
            considered linked for transition matrix
        
        min_ions : int
            The minimum ions to be considered a cluster
        
        lattice_parameters : array of shape (3,) or (3,3)
            If lattice parameters are not set in the ASE Atoms
            objects then provide the lattice parameters in Angstroms

    Returns : array shape(number of frames,3)
        The cluster information for each frame. Structured as:
            array([[frame number, number of clusters, cluster id's]...])
    """
    
    ## Make sure lattice parameters are set
    
    if lattice_parameters is None:
        if not check_lattice(traj):
            raise ValueError("\"lattice_parameters\" must be defined with shape (3,) or (3,3) ")

    else:
        for a in traj:
            a.set_cell(lattice_parameters)

    # iterate through frames and collect ion clusters at each frame
    n_clusters = []
    for frame,a in enumerate(traj):
        
        a.set_pbc([True,True,True])
        a.wrap()
        
        # Get list of neighboring Li within cutoff
        il,jl = nl('ij', a, {('Li','Li'):cutoff})

        # get minimum Li id to reset starting at 0
        if il.min() <= jl.min():
            index_offset = il.min()
        else:
            index_offset = jl.min()
        il -= index_offset
        jl -= index_offset

        # Get number of Li
        if il.max() >= jl.max():
            num_Li = il.max() + 1
        else:
            num_Li = jl.max() + 1

        # Start transition matrix with all zeros
        transition_matrix = np.zeros(shape=(num_Li,num_Li))
        
        # Add artificial loop value to diagonal
        np.fill_diagonal(transition_matrix, 1.0)

        # Fill transition matrix from neighborlist
        for i,j in zip(il,jl):
            transition_matrix[i,j] += 1.0

        # Perform Markov Clustering
        clusters_t = markov_clustering(transition_matrix,pruning_threshold = 0.00001)

        # Filter small clusters based on min_ions
        clusters = []
        for clust in clusters_t:
            new_cluster_list = []
            if len(clust) >= min_ions:
                for Li_ind in clust:
                    # add correct ion index for clusetrs
                    new_cluster_list.append(Li_ind+index_offset)
                clusters.append(tuple(new_cluster_list))

        n_clusters.append([frame,len(clusters), clusters])

    return np.asarray(n_clusters)

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''    
    setOfElems = set()
    elems = []
    for elem in listOfElems:
        if elem in setOfElems:
            elems.append(elem)
        else:
            setOfElems.add(elem)         
    return elems

def cluster_lifetime(clusters,similatiry_cut, min_ions,time_step=1.0,continuous=False):
    """
    Calculate the lifetime of clusters.
    
    Parameters:
        clusters : array shape(number of frames,3)
            The cluster information for each frame. Structured as:
            
                array([[frame number, number of clusters, cluster id's]...])
        
        similatiry_cut : int
            The number of ions in two clusters to be considers the same cluster
        
        timestep :  float
            The timestep of the trajectory used to make the clusters
    
    """ 
    def add_cluster(cluster, start):
        
        cluster_id = {'o_cluster':cluster, 
                      'c_cluster':[cluster], 
                      'len':1, 'start':start, 'end':start}
        return cluster_id
    
    def update_cluster(cluster_id, cluster):
        
        cluster_id['len'] += 1
        cluster_id['c_cluster'].append(cluster)
        cluster_id['end'] += 1
        
        return cluster_id
        
    cluster_ids = {}
    life_times_list = []
    
    # Get clusters from first frame
    frame_n = 0
    for i, cluster in enumerate(clusters[0,2]):
        if len(cluster) < min_ions:
            cluster_ids[i] = add_cluster(cluster,0)
    
    # Starting with the second frame calculate 
    # how many frames the cluster exists
    for i,frame in enumerate(clusters[1:,2]):
        f = i + 1
        
        # Check ions are only in one cluster
        # Should be the case based on the nature
        # of Markov Clustering
        flat_ions = list(sum(frame, ()))
        n_repeats = len(flat_ions) - np.unique(flat_ions).shape[0]
        if n_repeats > 0:
            elem = checkIfDuplicates(flat_ions)
            print('Same ions {} in two clusters in frame {}\n{}'.format(elem,f))
            
        # For each cluster in the frame 
        # check the similatiry of clusters 
        # to previous clusters 
        for cluster in frame:
            if len(cluster) < min_ions:
                found = True
                continue
            else:
                found = False
            current_ids = list(cluster_ids.keys())
            for c_ids in current_ids:
                similarity = set(cluster_ids[c_ids]['o_cluster']) & set(cluster)
                
                # If similare update lifetime info
                if len(similarity) >= similatiry_cut:
                    if continuous:
                        if i == cluster_ids[c_ids]['end']:
                            cluster_ids[c_ids] = update_cluster(cluster_ids[c_ids],
                                                            cluster)
                            found = True
                            break
                        else:
                            new_ids = np.max(list(cluster_ids.keys())) + 1
                            cluster_ids[new_ids] = add_cluster(cluster,f)
                            found = True
                    else:
                        cluster_ids[c_ids] = update_cluster(cluster_ids[c_ids],
                                                            cluster)
                        found = True
                        break
            # If not similar to other cluseter add new cluster to dict
            if not found:
                new_ids = np.max(list(cluster_ids.keys())) + 1
                cluster_ids[new_ids] = add_cluster(cluster,f)
                
    # Make list of lifetimes for ease later
    # and change 'len' from number of number of frames to time
    for c_ids in cluster_ids.keys():
        cluster_ids[c_ids]['len'] *= time_step
        life_times_list.append(cluster_ids[c_ids]['len'])
        
    return cluster_ids, life_times_list

## Choose number of ions to be considered a cluster
min_ions = 2

## Choose The number of ions in two clusters to be considers the same cluster 
similarity = 3

## Choose distance cutoff used to make the transition matrix for Markov Clustering
distance_cutoff = 4.0

## Add lattice parameters. Not need if lattice parameters are set in the trajectory
lat_par = None
#lat_par = [31.386271, 31.386271, 31.386271]

## Load trajectory
#traj_file_name = 'XDATCAR'
traj_file_name = 'nvt_smalldt.xyz'
traj = ase.io.read(traj_file_name,index='-2000:')

np_clusters = 'clusters_min{}_dist{}.npy'.format(min_ions,distance_cutoff)

if path.exists(np_clusters):
    print('Loaded')
    clusters = np.load(np_clusters, allow_pickle=True)
else:
    print('Calculated')
    clusters = count_Li_clusters(traj, distance_cutoff,
                                 min_ions=min_ions,
                                 lattice_parameters = lat_par)
    np.save(np_clusters, clusters)

## Set timestep if not 1.0fs and set continuous to False 
## if you want lifetimes to include clusters that break up and reform
continuous = True
cluster_ids, life_times = cluster_lifetime(clusters, similarity, 8 
                                           time_step=1.0, continuous=continuous)
if continuous:
    name = 'Continuous'
else:
    name = 'Intermittent'

fig, ax = plt.subplots()
ax.hist(life_times, bins=25, label=name)
ax.set_xlabel('Lifetime of Cluster (fs)')
ax.set_ylabel('Counts')
ax.legend()
#ax.set_ylim(0,50)
plt.savefig('cluster_hist_min{}_sim{}_dist{}.png'.format(min_ions,similarity,distance_cutoff))

