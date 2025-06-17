import torch
import torch.nn.functional as F
import os
from src.image_clusterer.clusterer import Clusterer

RANDOM_SCALE: float = 4.
MAX_CYCLES = int(os.getenv("K_MEANS_MAX_CYCLES", "100"))


class KMeansClusterer(Clusterer):
    def make_cluster(self, n_kernels):
        if n_kernels > len(self.data):
            raise Exception(message=f"n_kernels '{n_kernels}' bigger than amount of vectors '{len(self.data)}'")
        i_kernels = self._select_random_kernels(n_kernels).type(torch.LongTensor)
        kernels = self.data[i_kernels]
        for _ in range(MAX_CYCLES):
            weights = self.data @ kernels.T
            self.vectors2kernels = weights.argmax(dim=-1)
            weights = F.one_hot(self.vectors2kernels).to('cuda' if self.gpu else 'cpu')
            kernels_tmp = (weights.T[...,None] * self.data[None,...]).sum(dim=-2)
            kernels_tmp /= kernels_tmp.norm(dim=-1).unsqueeze(-1)
            if (kernels_tmp * kernels).sum() > len(kernels) - 1e-4:
                break
            kernels = kernels_tmp
            print(f"finished cycle {_}")
    
    def _select_random_kernels(self, n_kernels):
        n_vectors, _ = self.data.shape
        a = torch.rand(1, n_kernels).to('cuda' if self.gpu else 'cpu').pow(RANDOM_SCALE)
        a[:,0] = a[:,0] * (n_vectors - n_kernels)
        a[:,0] = a[:,0].floor()
        for i in range(1, n_kernels):
            a[:,i] = a[:,i-1] + 1 + (a[:,i] * (n_vectors - a[:,i-1] - n_kernels + i))
            a[:,i] = a[:,i].floor()
        return a[0,:]
