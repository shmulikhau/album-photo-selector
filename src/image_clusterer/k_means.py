import torch
from src.image_clusterer.clusterer import Clusterer

RANDOM_SCALE: float = 4.

class KMeansClusterer(Clusterer):
    def make_cluster(self):
        return super().make_cluster()
    
    def _select_random_kernels(self, n_kernels):
        n_vectors, _ = self.data.shape
        a = torch.rand(1, n_kernels).to('cuda' if self.gpu else 'cpu').pow(RANDOM_SCALE)
        a[:,0] = a[:,0] * (n_vectors - n_kernels)
        a[:,0] = a[:,0].floor()
        for i in range(1, n_kernels):
            a[:,i] = a[:,i-1] + 1 + (a[:,i] * (n_vectors - a[:,i-1] - n_kernels + i))
            a[:,i] = a[:,i].floor()
        return a[0,:]
