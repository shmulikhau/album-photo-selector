import torch
import numpy as np

class Clusterer:
    def __init__(self, embedding_dim, gpu: bool = False):
        self.gpu = gpu
        self.data = torch.zeros(0, embedding_dim).to('cuda' if self.gpu else 'cpu')
        self.vectors2kernels = None

    def push(self, embedding):
        self.push_batch(torch.tensor([embedding]).to('cuda' if self.gpu else 'cpu'))

    def push_batch(self, batch_embedding):
        self.data = torch.cat((self.data, batch_embedding), dim=0)

    def make_cluster(self, n_kernels):
        raise NotImplementedError()