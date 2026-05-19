"""Inference-only stubs for RIFE training losses.

The released RIFE_HDv3.py instantiates EPE() and SOBEL() in ``Model.__init__``
even when used only for inference. These classes pull in training-time logic
we don't ship. We provide no-op stand-ins so import + construction succeed;
inference calls only use ``self.flownet``.
"""
import torch
import torch.nn as nn


class EPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - infer never calls
        raise NotImplementedError("EPE is a training-only stub at inference time")


class SOBEL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernelX = torch.tensor([[1.0, 0.0, -1.0],
                                [2.0, 0.0, -2.0],
                                [1.0, 0.0, -1.0]]).reshape(1, 1, 3, 3)
        kernelY = kernelX.transpose(2, 3)
        self.register_buffer("kernelX", kernelX)
        self.register_buffer("kernelY", kernelY)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("SOBEL is a training-only stub at inference time")


class VGGPerceptualLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("VGGPerceptualLoss training-only stub")
