"""Forward-hook helpers for collecting model activations."""

from __future__ import annotations

from collections import OrderedDict
from typing import OrderedDict as OrderedDictType

import torch
from torch import nn


class ActivationRecorder:
    """Record named module outputs during a forward pass."""

    def __init__(self) -> None:
        self.activations: OrderedDictType[str, torch.Tensor] = OrderedDict()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def watch(self, name: str, module: nn.Module) -> None:
        handle = module.register_forward_hook(self._make_hook(name))
        self.handles.append(handle)

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
                self.activations[name] = output[0].detach().cpu()
            else:
                raise TypeError(f"cannot record non-tensor activation for {name}")

        return hook

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

