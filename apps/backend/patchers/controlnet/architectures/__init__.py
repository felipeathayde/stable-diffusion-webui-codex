"""ControlNet architecture registry."""

from .factory import (
    ControlArchitectureRegistry,
    default_architecture_registry,
    resolve_control_module,
    create_control_module,
)
from .sd.control import ControlNet
from .sd.control_lite import ControlNetLite
from .sd.lora import ControlLora
from .sd.t2i_adapter import T2IAdapter

# Register default SD-family modules.
default_architecture_registry.register("controlnet", ControlNet)
default_architecture_registry.register("controlnet_lite", ControlNetLite)
default_architecture_registry.register("controlnet_lora", ControlLora)
default_architecture_registry.register("t2i_adapter", T2IAdapter)

__all__ = [
    "ControlArchitectureRegistry",
    "default_architecture_registry",
    "resolve_control_module",
    "create_control_module",
]
