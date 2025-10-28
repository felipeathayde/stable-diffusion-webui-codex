from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple


@dataclass
class SignalBundle:
    state_dict: Mapping[str, Any]
    keys: Tuple[str, ...]
    shapes: Mapping[str, Tuple[int, ...]]

    def shape(self, key: str) -> Tuple[int, ...] | None:
        return self.shapes.get(key)

    def has_prefix(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.keys)


def build_bundle(state_dict: Mapping[str, Any]) -> SignalBundle:
    keys = tuple(state_dict.keys())
    shapes: dict[str, Tuple[int, ...]] = {}
    for k, v in state_dict.items():
        shape = getattr(v, "shape", None)
        if shape is None and hasattr(v, "size"):
            try:
                shape = tuple(int(x) for x in v.size())  # type: ignore[arg-type]
            except Exception:
                shape = None
        if shape is not None:
            shapes[k] = tuple(int(x) for x in shape)
    return SignalBundle(state_dict=state_dict, keys=keys, shapes=shapes)


def count_blocks(keys: Iterable[str], template: str) -> int:
    """Count sequential blocks in keys following a zero-based template.

    Example template: ``"model.diffusion_model.input_blocks.{}."``
    """
    count = 0
    while True:
        prefix = template.format(count)
        if not any(k.startswith(prefix) for k in keys):
            break
        count += 1
    return count


def has_all_keys(bundle: SignalBundle, *required: str) -> bool:
    return all(k in bundle.state_dict for k in required)


def get_tensor_dtype(tensor: Any) -> str | None:
    return getattr(getattr(tensor, "dtype", None), "name", None)
