import torch
import torch.nn.functional as F
import os
from src.image_clusterer.clusterer import Clusterer

RANDOM_SCALE: float = 4.
MAX_CYCLES = int(os.getenv("K_MEANS_MAX_CYCLES", "100"))


class KMeansClusterer(Clusterer):
    def make_cluster(self, n_kernels):
        i_kernels = self._select_random_kernels(n_kernels)
        kernels = self.data[i_kernels]
        for _ in range(MAX_CYCLES):
            weights = self.data @ kernels.T
            self.vectors2kernels = weights.argmax(dim=-1)
            weights = F.one_hot(self.vectors2kernels).to('cuda' if self.gpu else 'cpu')
            kernels_tmp = (weights[None,...] * self.data[...,None]).sum(dim=-2)
            if kernels_tmp.equal(kernels):
                break
    
    def _select_random_kernels(self, n_kernels):
        n_vectors, _ = self.data.shape
        a = torch.rand(1, n_kernels).to('cuda' if self.gpu else 'cpu').pow(RANDOM_SCALE)
        a[:,0] = a[:,0] * (n_vectors - n_kernels)
        a[:,0] = a[:,0].floor()
        for i in range(1, n_kernels):
            a[:,i] = a[:,i-1] + 1 + (a[:,i] * (n_vectors - a[:,i-1] - n_kernels + i))
            a[:,i] = a[:,i].floor()
        return a[0,:]
