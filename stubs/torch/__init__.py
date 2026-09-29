"""
Stub: torch
PC-only stub để car_control agent_executor hoạt động mà không cần CUDA.
Trên Jetson, real torch được load từ venv nên stub này bị bỏ qua.
"""
import math
import numpy as np

# ── nn.functional ──────────────────────────────────────────────────────────

class _F:
    @staticmethod
    def normalize(tensor, p=2, dim=1):
        norms = np.linalg.norm(tensor._data, axis=dim, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return Tensor(tensor._data / norms)

    @staticmethod
    def softmax(tensor, dim=-1):
        data = tensor._data
        e = np.exp(data - data.max(axis=dim, keepdims=True))
        return Tensor(e / e.sum(axis=dim, keepdims=True))


# ── nn ─────────────────────────────────────────────────────────────────────

class _NN:
    functional = _F()

    class Module:
        def eval(self): return self
        def __call__(self, *a, **kw): return self.forward(*a, **kw)
        def forward(self, *a, **kw): raise NotImplementedError

    class Linear(Module):
        def __init__(self, in_f, out_f, bias=True):
            self.weight = None
        def forward(self, x): return x


nn = _NN()


# ── Tensor ─────────────────────────────────────────────────────────────────

class Tensor:
    def __init__(self, data):
        if isinstance(data, Tensor):
            data = data._data
        self._data = np.array(data, dtype=np.float32)

    # shape / ndim
    @property
    def shape(self): return self._data.shape

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self._data, axis=dim))

    def squeeze(self, dim=None):
        return Tensor(np.squeeze(self._data, axis=dim) if dim is not None else np.squeeze(self._data))

    def sum(self, dim=None, keepdim=False):
        kw = {"keepdims": keepdim}
        return Tensor(self._data.sum(axis=dim, **kw) if dim is not None else self._data.sum(**kw))

    def clamp(self, min=None, max=None):
        return Tensor(np.clip(self._data, a_min=min, a_max=max))

    def item(self):
        return float(self._data.flat[0])

    def numpy(self):
        return self._data.copy()

    def float(self): return self

    def __mul__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data * o)

    def __truediv__(self, other):
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data / o)

    def __repr__(self):
        return f"Tensor({self._data})"


def matmul(a, b):
    ad = a._data if isinstance(a, Tensor) else np.array(a)
    bd = b._data if isinstance(b, Tensor) else np.array(b)
    return Tensor(np.matmul(ad, bd))


def max(input, dim=None):
    if dim is None:
        return Tensor(np.array(input._data.max()))
    values = input._data.max(axis=dim)
    indices = input._data.argmax(axis=dim)
    class _MaxResult:
        def __init__(self, v, i):
            self.values = Tensor(v)
            self.indices = Tensor(i.astype(np.int64))
        def __iter__(self):
            return iter((self.values, self.indices))
    return _MaxResult(values, indices)


def no_grad():
    """Context manager no-op."""
    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Ctx()


def zeros(*shape, **kw):
    return Tensor(np.zeros(shape, dtype=np.float32))

def ones(*shape, **kw):
    return Tensor(np.ones(shape, dtype=np.float32))

def tensor(data, **kw):
    return Tensor(np.array(data, dtype=np.float32))

def cat(tensors, dim=0):
    arrays = [t._data if isinstance(t, Tensor) else np.array(t) for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim))

# ── device / cuda ──────────────────────────────────────────────────────────

class _Device:
    def __init__(self, name): self.name = name
    def __str__(self): return self.name

def device(s): return _Device(s)

class _CUDA:
    is_available = staticmethod(lambda: False)

cuda = _CUDA()

# ── dtype stubs ────────────────────────────────────────────────────────────
float32 = np.float32
int64 = np.int64
long = np.int64
