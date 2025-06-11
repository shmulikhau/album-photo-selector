import numpy as np

class ImageEmbedder:
    def get_embeddings(self, image_batch) -> np.ndarray:
        raise NotImplementedError()