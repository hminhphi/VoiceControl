"""
Stub: transformers
PC-only. Cung cấp AutoModel / AutoTokenizer giả lập dùng numpy embeddings.
car_control dùng khi ENABLE_EMBEDDING_MATCH=true (mặc định false trên PC).
"""
import numpy as np

_RNG = np.random.default_rng(42)


class _FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt", **kw):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        # Return fake token tensors
        from torch import Tensor
        return {
            "input_ids": Tensor(np.zeros((n, 8), dtype=np.float32)),
            "attention_mask": Tensor(np.ones((n, 8), dtype=np.float32)),
            "token_type_ids": Tensor(np.zeros((n, 8), dtype=np.float32)),
        }

    def __repr__(self): return "FakeTokenizer()"


class _FakeOutput:
    def __init__(self, n, seq=8, hidden=256):
        from torch import Tensor
        self.last_hidden_state = Tensor(
            _RNG.standard_normal((n, seq, hidden)).astype(np.float32)
        )


class _FakeModel:
    def eval(self): return self
    def __call__(self, **kw):
        mask = kw.get("attention_mask")
        if mask is not None:
            n = mask._data.shape[0] if hasattr(mask, "_data") else 1
        else:
            n = 1
        return _FakeOutput(n)

    def __repr__(self): return "FakeModel()"


class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_path, **kw):
        print(f"[transformers-stub] AutoTokenizer.from_pretrained({model_path!r}) → FakeTokenizer")
        return _FakeTokenizer()


class AutoModel:
    @classmethod
    def from_pretrained(cls, model_path, **kw):
        print(f"[transformers-stub] AutoModel.from_pretrained({model_path!r}) → FakeModel")
        return _FakeModel()


class BertModel(_FakeModel):
    pass

class RobertaModel(_FakeModel):
    pass

class pipeline:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return []
