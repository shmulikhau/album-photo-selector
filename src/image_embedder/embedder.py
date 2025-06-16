import numpy as np

class ImageEmbedder:

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self.model = self.load_model()

    def load_model(self):
        raise NotImplementedError()

    def get_image_embeddings(self, image_batch) -> np.ndarray:
        raise NotImplementedError()
    
    def get_text_embeddings(self, text) -> np.ndarray:
        raise NotImplementedError()