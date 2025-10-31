"""SDXL-specific ControlNet placeholders."""

from ..sd.control import ControlNet  # reuse SD implementation until specialization lands
from ..sd.control_lite import ControlNetLite
from ..sd.lora import ControlLora
from ..sd.t2i_adapter import T2IAdapter

__all__ = ["ControlNet", "ControlNetLite", "ControlLora", "T2IAdapter"]
