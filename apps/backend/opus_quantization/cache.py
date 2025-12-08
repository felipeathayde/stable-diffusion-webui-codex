# OpusQuantization - Thread-safe device cache

from __future__ import annotations

import threading
from typing import Callable, Dict, Any

import torch

__all__ = ["DeviceCache", "get_device_cache"]


class DeviceCache:
    """
    Thread-safe per-device tensor cache.
    
    Used to store lookup tables and other device-specific tensors
    without modifying global state during inference.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[torch.device, Dict[str, torch.Tensor]] = {}
    
    def get_or_create(
        self,
        device: torch.device,
        key: str,
        factory: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """
        Get a cached tensor for the given device, or create it using factory.
        
        Thread-safe: multiple threads can safely call this concurrently.
        """
        # Normalize device
        if isinstance(device, str):
            device = torch.device(device)
        
        # Fast path: check without lock
        device_cache = self._cache.get(device)
        if device_cache is not None:
            tensor = device_cache.get(key)
            if tensor is not None:
                return tensor
        
        # Slow path: create with lock
        with self._lock:
            if device not in self._cache:
                self._cache[device] = {}
            
            if key not in self._cache[device]:
                # Create the tensor and move to device
                tensor = factory()
                if tensor.device != device:
                    tensor = tensor.to(device)
                self._cache[device][key] = tensor
            
            return self._cache[device][key]
    
    def clear(self, device: torch.device = None) -> None:
        """Clear cache for a specific device, or all devices if None."""
        with self._lock:
            if device is None:
                self._cache.clear()
            elif device in self._cache:
                del self._cache[device]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "devices": list(str(d) for d in self._cache.keys()),
                "keys_per_device": {
                    str(d): list(keys.keys()) 
                    for d, keys in self._cache.items()
                },
            }


# Global cache instance
_GLOBAL_CACHE = DeviceCache()


def get_device_cache() -> DeviceCache:
    """Get the global device cache instance."""
    return _GLOBAL_CACHE
