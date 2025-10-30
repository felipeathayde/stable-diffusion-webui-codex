from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import torch

from .config import ControlGraph, ControlNode

logger = logging.getLogger("backend.runtime.controlnet")


class ControlComposite:
    """Composite wrapper that exposes legacy ControlNet interfaces while using ControlGraph."""

    def __init__(self, graph: ControlGraph):
        self.graph = graph
        self.head = None
        self._update_links()

    def _update_links(self) -> None:
        previous = None
        for node in self.graph.nodes:
            control = node.control
            control.set_previous_controlnet(previous)
            previous = control
        self.head = previous

    def prepare(self, model, percent_to_sigma) -> None:
        logger.debug("Preparing %d control nodes", len(self.graph.nodes))
        self._update_links()
        for node in self.graph.nodes:
            node.control.pre_run(model, percent_to_sigma)

    def cleanup(self) -> None:
        for node in self.graph.nodes:
            node.control.cleanup()
        self.head = None

    def inference_memory_requirements(self, dtype: torch.dtype) -> int:
        total = 0
        for node in self.graph.nodes:
            total += node.control.inference_memory_requirements(dtype)
        return total

    def get_models(self) -> List[object]:
        models: List[object] = []
        for node in self.graph.nodes:
            models.extend(node.control.get_models())
        return models

    def set_transformer_options(self, transformer_options):
        for node in self.graph.nodes:
            node.control.transformer_options = transformer_options

    def get_control(self, x_noisy, t, cond, batched_number):
        if self.head is None:
            return None
        return self.head.get_control(x_noisy, t, cond, batched_number)

    def __bool__(self):
        return bool(self.graph.nodes)


def build_composite(nodes: Iterable[ControlNode]) -> Optional[ControlComposite]:
    nodes_list = list(nodes)
    if not nodes_list:
        return None
    graph = ControlGraph(nodes=nodes_list)
    return ControlComposite(graph)
