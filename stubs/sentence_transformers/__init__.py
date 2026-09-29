import numpy as np

class SentenceTransformer:
    def __init__(self, model_name_or_path, *args, **kwargs):
        print(f"[sentence_transformers-stub] Model loaded: {model_name_or_path}")
        self.model_name = model_name_or_path

    def encode(self, sentences, *args, **kwargs):
        if isinstance(sentences, str):
            return np.zeros(384, dtype=np.float32)
        else:
            return np.zeros((len(sentences), 384), dtype=np.float32)

    def to(self, device):
        pass
