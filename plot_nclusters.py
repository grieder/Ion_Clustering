import numpy as np
import matplotlib.pyplot as plt


np_clusters = 'clusters_min{}_dist{}.npy'.format(2,4.0)
raw_clusters = np.load(np_clusters, allow_pickle=True)
legend_label = 'Melt 3000K'

def filter_clusters(clusters, min_ions=3):
    n_clusters = [] 
    for frame in clusters:
        t_clusters = []
        for cluster in frame[2]:
            if len(cluster) >= min_ions:
                t_clusters.append(cluster)
        n_clusters.append([frame[0],len(t_clusters),t_clusters])
    return np.asarray(n_clusters)

fig,axs = plt.subplots(2,3, figsize=(12,6))

for ax,cut in zip(axs.flatten(),[5,6,7,8,9,10]):
   clusters = filter_clusters(raw_clusters, min_ions=cut)
   
   ax.plot(clusters[:,0], clusters[:,1], label=legend_label)
   
   ax.set_xlabel('Frames')
   ax.set_ylabel(r'Number of Markov Clusters')
   ax.set_title(r'Clusters >= {} Li'.format(cut))
   ax.legend()
    
plt.tight_layout()
plt.savefig('number_cluster_overtime.png')
plt.show()

