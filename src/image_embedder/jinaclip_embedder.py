from src.image_embedder.embedder import ImageEmbedder

class JinaClip(ImageEmbedder):
    def __init__(self, gpu: bool = False):
        pass

    def get_embeddings(self, image_batch):
        return super().get_embeddings(image_batch)