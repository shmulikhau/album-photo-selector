from src.image_clusterer.clusterer import Clusterer

class KMeansClusterer(Clusterer):
    def make_cluster(self, gpu: bool = False):
        return super().make_cluster()