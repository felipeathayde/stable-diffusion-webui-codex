import json
import logging
import os

from collections.abc import MutableMapping
from typing import Dict, Tuple

import safetensors.torch
import torch
from safetensors.torch import safe_open

from apps.backend import gguf
from apps.backend.runtime.misc import checkpoint_pickle
from apps.backend.runtime.ops.operations_gguf import ParameterGGUF

_log = logging.getLogger("backend.runtime.utils")


# Avoid importing the whole `runtime.ops` package here (it imports `utils`,
# creating a circular import). Import the required symbol directly from the
# submodule that defines it.


class KeyPrefixView(MutableMapping):
    """Lightweight mapping view that exposes `base` keys with a fixed prefix.

    - Does not materialize tensor values; delegates to `base[key_without_prefix]` on access.
    - Deletions and sets propagate to the underlying mapping.
    - Useful to avoid rebuilding huge state_dicts on CPU.
    """

    def __init__(self, base: MutableMapping, prefix: str):
        self._base = base
        self._prefix = prefix

    def _strip(self, k: str) -> str:
        if not k.startswith(self._prefix):
            raise KeyError(k)
        return k[len(self._prefix):]

    def __getitem__(self, k: str):
        return self._base[self._strip(k)]

    def __setitem__(self, k: str, v):
        self._base[self._strip(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._strip(k)]

    def __iter__(self):
        for k in self._base.keys():
            yield f"{self._prefix}{k}"

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            # Fallback: iterate
            return sum(1 for _ in self.__iter__())


class FilterPrefixView(MutableMapping):
    """View over keys under a given prefix, optionally re-prefixed lazily.

    - base: mapping with original keys (e.g., LazySafetensorsDict or KeyPrefixView)
    - prefix: filter only keys that start with this
    - new_prefix: keys presented by this view will start with new_prefix instead
    """

    def __init__(self, base: MutableMapping, prefix: str, new_prefix: str = ""):
        self._base = base
        self._prefix = prefix
        self._new_prefix = new_prefix

    def _to_base_key(self, k: str) -> str:
        # Map presented key 'k' back to the underlying base mapping key.
        if self._new_prefix:
            if k.startswith(self._new_prefix):
                return self._prefix + k[len(self._new_prefix):]
            # If caller already uses base prefix, pass-through
            if k.startswith(self._prefix):
                return k
            # Fallback: assume k is suffix; prepend prefix
            return self._prefix + k
        else:
            # Presented keys are suffix-only; base keys use prefix
            if k.startswith(self._prefix):
                return k
            return self._prefix + k

    def _present_key(self, base_key: str) -> str:
        if not self._prefix:
            if self._new_prefix:
                return f"{self._new_prefix}{base_key}"
            return base_key
        suffix = base_key[len(self._prefix):]
        if self._new_prefix:
            return f"{self._new_prefix}{suffix}"
        return suffix

    def __getitem__(self, k: str):
        return self._base[self._to_base_key(k)]

    def __setitem__(self, k: str, v):
        self._base[self._to_base_key(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._to_base_key(k)]

    def __iter__(self):
        for k in self._base.keys():
            if k.startswith(self._prefix):
                out = self._new_prefix + k[len(self._prefix):]
                yield out

    def __len__(self) -> int:
        c = 0
        for _ in self.__iter__():
            c += 1
        return c

    def materialize(self, *, return_mapping: bool = False):
        """Realise all tensors matching the prefix into a concrete dict.

        Prefers calling the underlying mapping's `materialize` helper when
        available so SafeTensors files are streamed with a single handle open.
        """
        materializer = getattr(self._base, "materialize", None)
        if callable(materializer):
            try:
                return materializer(prefix=self._prefix, new_prefix=self._new_prefix, return_mapping=return_mapping)
            except TypeError:
                result = materializer(prefix=self._prefix, new_prefix=self._new_prefix)
                if return_mapping:
                    raise
                return result

        out: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        for key in self._base.keys():
            if not key.startswith(self._prefix):
                continue
            presented = self._present_key(key)
            out[presented] = self._base[key]
            mapping[presented] = key
        if return_mapping:
            return out, mapping
        return out


class RemapKeysView(MutableMapping):
    """Present a remapped keyspace over an underlying mapping lazily.

    - base: original mapping (e.g., LazySafetensorsDict)
    - mapping: dict[new_key] -> old_key in the base mapping
    - Does not materialize any tensor unless __getitem__ is called.
    """

    def __init__(self, base: MutableMapping, mapping: dict[str, str]):
        self._base = base
        self._map = dict(mapping)

    def __getitem__(self, k: str):
        return self._base[self._map[k]]

    def __setitem__(self, k: str, v):
        self._map[k] = k
        self._base[k] = v

    def __delitem__(self, k: str):
        old = self._map.pop(k, None)
        if old is not None and old in self._base:
            del self._base[old]

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self):
        return len(self._map)

    def keys(self):
        return list(self._map.keys())

    def items(self):
        for k in self._map.keys():
            yield k, self._base[self._map[k]]

class CastOnGetView(MutableMapping):
    """Mapping view that casts tensor values on CPU to a target dtype on access.

    Useful to avoid fragile CPU bf16/fp16 ops during preprocessing. Only casts
    floating tensors matching `from_dtypes` and `device_type`.
    """

    def __init__(self, base: MutableMapping, *, device_type: str = "cpu", from_dtypes=None, to_dtype=None):
        import torch as _torch
        self._base = base
        self._device_type = device_type
        self._from = tuple(from_dtypes) if from_dtypes is not None else (_torch.bfloat16, _torch.float16)
        self._to = to_dtype or _torch.float32

    def __getitem__(self, k: str):
        import torch as _torch
        v = self._base[k]
        if isinstance(v, _torch.Tensor):
            try:
                if v.device.type == self._device_type and v.dtype in self._from:
                    return v.to(self._to)
            except Exception:
                return v
        return v

    def __setitem__(self, k: str, v):
        self._base[k] = v

    def __delitem__(self, k: str):
        del self._base[k]

    def __iter__(self):
        return iter(self._base.keys())

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            return sum(1 for _ in self.__iter__())


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, 'rt', encoding='utf-8') as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt, safe_load=True, device=None):
    """Load a checkpoint (safetensors/gguf/pickle) honoring an explicit device.

    - When ``device`` is None, use the current core initial load device from the
      memory manager to avoid accidental CPU pinning.
    - For safetensors, the returned mapping lazily loads tensors using
      ``safe_open(..., device=<device>)`` so values are produced directly on the
      requested device when possible.
    """
    from apps.backend.runtime.memory import memory_management as _mm  # local import avoids cycles

    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        try:
            device = _mm.core_initial_load_device(parameters=0, dtype=None)
        except Exception:
            device = torch.device("cpu")

    checkpoint_path = str(ckpt)
    suffix = os.path.splitext(checkpoint_path)[1].lower()

    if suffix == ".safetensors":
        # Always load tensors on CPU during state-dict preprocessing to avoid CUDA OOMs
        # and fragmentation when inspecting keys and remapping. Model weights will be
        # moved to the target device later by the memory manager.
        return LazySafetensorsDict(checkpoint_path, device="cpu")
    if suffix == ".gguf":
        return _load_gguf_state_dict(checkpoint_path)

    pl_sd = _load_pickled_checkpoint(checkpoint_path, device, safe_load)

    if "global_step" in pl_sd:
        _log.info("Global Step: %s", pl_sd['global_step'])

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def _load_gguf_state_dict(path):
    reader = gguf.GGUFReader(path)
    state_dict = {}
    for tensor in reader.tensors:
        state_dict[str(tensor.name)] = ParameterGGUF(tensor)
    return state_dict


class LazySafetensorsDict(MutableMapping):
    """Lazy, mutable mapping backed by a .safetensors file.

    - Keys come from the file; values are loaded on demand with safe_open.get_tensor.
    - Supports overlay writes and deletions without touching the underlying file.
    - Device: only CPU tensors are produced (parity with previous loader).
    
    Windows crash prevention: Once any tensor is accessed, the entire file is
    materialized into memory to avoid reopening the file repeatedly (which causes
    torch_cpu.dll crashes on Windows).
    """

    def __init__(self, filepath: str, device: str = "cpu"):
        self.filepath = filepath
        self.device = device or "cpu"
        self._overlay = {}          # in-memory writes/overrides
        self._deleted = set()       # keys logically removed
        self._keys_cache = None     # cached set of underlying keys
        self._materialized = None   # holds all tensors after first access
        self._materialized_triggered = False

    def _base_keys(self):
        if self._keys_cache is None:
            with safe_open(self.filepath, framework="pt", device=self.device) as f:
                self._keys_cache = set(f.keys())
        return self._keys_cache

    def _ensure_materialized(self):
        """Load all tensors from file once to avoid reopening repeatedly."""
        if self._materialized is None and not self._materialized_triggered:
            self._materialized_triggered = True
            self._materialized = {}
            try:
                with safe_open(self.filepath, framework="pt", device=self.device) as f:
                    self._keys_cache = set(f.keys())
                    for key in f.keys():
                        self._materialized[key] = f.get_tensor(key)
            except Exception:
                # If materialization fails, clear and fall back to per-key loading
                self._materialized = None
                self._materialized_triggered = False

    # Mapping protocol
    def __getitem__(self, key):
        if key in self._overlay:
            return self._overlay[key]
        if key in self._deleted:
            raise KeyError(key)
        
        # Materialize all tensors on first access to avoid repeated file opens
        self._ensure_materialized()
        
        if self._materialized is not None:
            if key in self._materialized:
                return self._materialized[key]
            raise KeyError(key)
        
        # Fallback for edge cases (should rarely happen)
        if key not in self._base_keys():
            raise KeyError(key)
        with safe_open(self.filepath, framework="pt", device=self.device) as f:
            t = f.get_tensor(key)
        return t

    def __setitem__(self, key, value):
        self._overlay[key] = value
        if self._keys_cache is None and key not in self._deleted:
            # do not expand base key set; overlay keys are separate
            pass
        if key in self._deleted:
            self._deleted.remove(key)

    def __delitem__(self, key):
        if key in self._overlay:
            del self._overlay[key]
        else:
            # mark as deleted logically
            self._deleted.add(key)

    def __iter__(self):
        base = (k for k in self._base_keys() if k not in self._deleted)
        # overlay can shadow base
        for k in base:
            if k not in self._overlay:
                yield k
        for k in self._overlay.keys():
            yield k

    def __len__(self):
        return len([k for k in self._base_keys() if k not in self._deleted and k not in self._overlay]) + len(self._overlay)

    # Convenience helpers
    def keys(self):
        return list(iter(self))

    def items(self):
        # Use materialization to avoid per-item file opens
        self._ensure_materialized()
        for k in self:
            yield k, self[k]

    def materialize(
        self,
        *,
        prefix: str = "",
        new_prefix: str = "",
        return_mapping: bool = False,
    ):
        """Eagerly load tensors matching `prefix`, optionally re-prefixing keys."""

        def _translate(key: str) -> str:
            suffix = key[len(prefix):] if prefix and key.startswith(prefix) else key
            if new_prefix:
                return f"{new_prefix}{suffix}"
            if prefix:
                return suffix
            return key

        result: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        with safe_open(self.filepath, framework="pt", device=self.device) as handle:
            for key in handle.keys():
                if prefix and not key.startswith(prefix):
                    continue
                if key in self._deleted:
                    continue
                if key in self._overlay:
                    continue
                presented = _translate(key)
                result[presented] = handle.get_tensor(key)
                mapping[presented] = key

        for key, value in self._overlay.items():
            if prefix and not key.startswith(prefix):
                continue
            if key in self._deleted:
                continue
            presented = _translate(key)
            result[presented] = value
            mapping[presented] = key

        if return_mapping:
            return result, mapping
        return result



def _load_pickled_checkpoint(path, device, safe_load):
    if safe_load:
        from apps.backend.runtime.models import safety as model_safety
        try:
            return model_safety.safe_torch_load(path, map_location=device)
        except model_safety.UnsafeCheckpointError:
            raise
    return torch.load(path, map_location=device, pickle_module=checkpoint_pickle)
def set_attr(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], torch.nn.Parameter(value, requires_grad=False))


def set_attr_raw(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    setattr(obj, attrs[-1], value)


def copy_to_param(obj, attr, value):
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, attrs[-1])
    prev.data.copy_(value)


def get_attr(obj, attr):
    attrs = attr.split(".")
    for name in attrs:
        obj = getattr(obj, name)
    return obj


def get_attr_with_parent(obj, attr):
    attrs = attr.split(".")
    parent = obj
    name = None
    for name in attrs:
        parent = obj
        obj = getattr(obj, name)
    return parent, name, obj


def calculate_parameters(sd, prefix=""):
    params = 0
    for k in sd.keys():
        if k.startswith(prefix):
            params += sd[k].nelement()
    return params


def tensor2parameter(x):
    if isinstance(x, torch.nn.Parameter):
        return x
    else:
        return torch.nn.Parameter(x, requires_grad=False)


def fp16_fix(x):
    # An interesting trick to avoid fp16 overflow
    # Legacy issue reference: https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1114
    # Related: https://github.com/comfyanonymous/ComfyUI/blob/f1d6cef71c70719cc3ed45a2455a4e5ac910cd5e/comfy/ldm/flux/layers.py#L180

    if x.dtype in [torch.float16]:
        return x.clip(-32768.0, 32768.0)
    return x


def dtype_to_element_size(dtype):
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def nested_compute_size(obj, element_size):
    module_mem = 0

    if isinstance(obj, dict):
        for key in obj:
            module_mem += nested_compute_size(obj[key], element_size)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for i in range(len(obj)):
            module_mem += nested_compute_size(obj[i], element_size)
    elif isinstance(obj, torch.Tensor):
        module_mem += obj.nelement() * element_size

    return module_mem


def nested_move_to_device(obj, **kwargs):
    if isinstance(obj, dict):
        for key in obj:
            obj[key] = nested_move_to_device(obj[key], **kwargs)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = nested_move_to_device(obj[i], **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(nested_move_to_device(i, **kwargs) for i in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(**kwargs)
    return obj


def get_state_dict_after_quant(model, prefix=''):
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m.weight, 'bnb_quantized'):
            if not m.weight.bnb_quantized:
                original_device = m.weight.device
                m.cuda()
                m.to(original_device)

    sd = model.state_dict()
    sd = {(prefix + k): v.clone() for k, v in sd.items()}
    return sd


def beautiful_print_gguf_state_dict_statics(state_dict):
    from gguf.constants import GGMLQuantizationType
    type_counts = {}
    for k, v in state_dict.items():
        gguf_cls = getattr(v, 'gguf_cls', None)
        if gguf_cls is not None:
            type_name = gguf_cls.__name__
            if type_name in type_counts:
                type_counts[type_name] += 1
            else:
                type_counts[type_name] = 1
    _log.info('GGUF state dict: %s', type_counts)
    return
