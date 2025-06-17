from PIL import Image
import numpy as np
from transformers import AutoModel
from src.image_embedder.embedder import ImageEmbedder


class JinaClipEmbedder(ImageEmbedder):

    def load_model(self):
        return AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)\
            .to('cuda' if self.gpu else 'cpu')

    def get_image_embeddings(self, image_batch: list) -> np.ndarray:
        embeddings = self.model.encode_image(image_batch, batch_size=len(image_batch))
        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings
    
    def get_text_embeddings(self, text) -> np.ndarray:
        embeddings = self.model.encode_text(text)
        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings
