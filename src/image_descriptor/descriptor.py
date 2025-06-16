
class Descriptor:

    def __init__(self, gpu: bool = False):
        self.model = self.load_model()
        self.short_description_prompt = """
You are an intelligent agent tasked with analyzing a set of images and providing a short, 2-4 word description of the common denominator shared among them. Your description should capture the key visual themes or elements that appear in all the images.
When given a list of images, your response should:
1. Focus on key visual patterns or themes such as colors, objects, settings, or emotions.
2. Keep your description brief and to the point—no longer than 2-4 words.
3. If the images have no clear common denominator, provide a response such as "No common denominator" or "Varied elements."
"""

    def load_model(self):
        raise NotImplementedError()

    def get_description(self, image_paths: list[str]) -> str:
        raise NotImplementedError()