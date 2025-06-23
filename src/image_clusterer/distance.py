import torch
from timeit_decorator import timeit
from src.image_clusterer.clusterer import Clusterer

class DistanceClusterer(Clusterer):

    @timeit(log_level=None)
    def make_cluster(self, threshould: float):
        global data; global indecies; global cluster_count
        data = self.data.detach().clone()
        indecies = torch.arange(0, len(data))
        cluster_count = 0
        self.vectors2kernels = torch.zeros_like(indecies).to('cuda' if self.gpu else 'cpu')

        def clustering(vector):
            global data; global indecies; global cluster_count
            scores = vector @ data.T
            M = scores > threshould
            cluster = data[M]
            cluster_indicies = indecies[M]
            self.vectors2kernels[cluster_indicies] = cluster_count
            data = data[~M]
            indecies = indecies[~M]
            for index, feature in zip(cluster_indicies, cluster):
                clustering(feature)
        
        while len(data):
            self.vectors2kernels[indecies[0]] = cluster_count
            vector_cluster = data[0]
            mask = torch.ones_like(indecies).to('cuda' if self.gpu else 'cpu')
            mask[0] = 0
            data = data[mask==1]
            indecies = indecies[mask==1]
            clustering(vector_cluster)
            cluster_count += 1
