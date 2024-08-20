import dataclasses
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

_TORCH_AVAILABLE = True
try:
    import torch
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    FloatArray: TypeAlias = torch.FloatTensor | np.ndarray[np.float32]
    IntArray: TypeAlias = torch.LongTensor | np.ndarray[np.int64]
else:
    FloatArray: TypeAlias = np.ndarray[np.float32]
    IntArray: TypeAlias = np.ndarray[np.int64]


@dataclass
class BaseValueContainer:

    def as_tuple(self):
        return dataclasses.astuple(self)

    def as_dict(self):
        return dataclasses.asdict(self)

    def as_torch(self):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("`torch` is not installed")
        data = self.as_dict()
        kwargs = {}
        for name, value in self.as_dict().items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                kwargs[name] = torch.as_tensor(value)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)

    def as_numpy(self):
        data = self.as_dict()
        kwargs = {}
        for name, value in self.as_dict().items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                if _TORCH_AVAILABLE:
                    value = value.detach().cpu() if isinstance(value, torch.Tensor) else value
                kwargs[name] = np.asarray(value)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)

    def to(self, device: str):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("`torch` is not installed")
        kwargs = {}
        for name, value in self.as_dict().items():
            if isinstance(value, torch.Tensor):
                kwargs[name] = value.to(device)
            else:
                kwargs[name] = value
        cls = type(self)
        return cls(**kwargs)


@dataclass(kw_only=True)
class InferenceInputs(BaseValueContainer):
    clean_text: str
    x: IntArray
    x_lengths: IntArray
    sids: IntArray|None = None
    lids: IntArray|None = None
    d_factor: float = 1.0
    p_factor: float = 1.0
    e_factor: float = 1.0


@dataclass(kw_only=True)
class InferenceOutputs(BaseValueContainer):
    wav: FloatArray
    wav_lengths: FloatArray
    latency: int
    rtf: float
    durations: FloatArray|None = None
    pitch: FloatArray|None = None
    energy: FloatArray|None = None
    am_rtf: float|None = None
    v_rtf: float|None = None
