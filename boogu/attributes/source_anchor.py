"""Opt-in source-latent anchoring for experimental image editing.

State projection and velocity mixing are separate research modes. Neither
guarantees preservation of decoded RGB pixels or object identity.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SourceAnchorRequest:
    """Source-aligned spatial anchoring strengths in ``[0, 1]``.

    A value of zero leaves a latent position free; one follows the VAE source
    trajectory. Intermediate values are soft anchors, not attribute-specific
    guarantees. The caller must supply the map in source-canvas coordinates.
    """

    strength: torch.Tensor
    mode: str

    def __post_init__(self):
        if self.mode not in ("projection", "velocity"):
            raise ValueError("source anchor mode must be projection or velocity")
        if (
            not isinstance(self.strength, torch.Tensor)
            or self.strength.ndim != 4
            or self.strength.shape[1] != 1
            or min(self.strength.shape[-2:]) < 1
            or not torch.isfinite(self.strength).all()
            or (self.strength < 0).any()
            or (self.strength > 1).any()
        ):
            raise ValueError("source anchor strength must be finite Bx1xHxW in [0,1]")

    @classmethod
    def from_edit(
        cls, edit: torch.Tensor, *, edit_strength: float = 0.0,
        mode: str,
    ) -> "SourceAnchorRequest":
        """Anchor unchanged positions fully and edited positions softly."""
        if (
            not isinstance(edit, torch.Tensor)
            or edit.ndim != 4
            or edit.shape[1] != 1
            or not torch.isfinite(edit).all()
            or (edit < 0).any()
            or (edit > 1).any()
        ):
            raise ValueError("edit must be a finite Bx1xHxW map in [0,1]")
        if (
            isinstance(edit_strength, bool)
            or not isinstance(edit_strength, (int, float))
            or not math.isfinite(edit_strength)
            or not 0 <= edit_strength <= 1
        ):
            raise ValueError("edit_strength must be finite and in [0,1]")
        return cls(1 - edit.float() * (1 - edit_strength), mode=mode)


class SourceLatentAnchor:
    def __init__(
        self,
        request: SourceAnchorRequest,
        source_latent: torch.Tensor,
        initial_noise: torch.Tensor,
    ):
        if not isinstance(request, SourceAnchorRequest):
            raise TypeError("source anchor requires a SourceAnchorRequest")
        if (
            not isinstance(source_latent, torch.Tensor)
            or not isinstance(initial_noise, torch.Tensor)
            or source_latent.ndim != 4
            or source_latent.shape != initial_noise.shape
            or not torch.isfinite(source_latent).all()
            or not torch.isfinite(initial_noise).all()
        ):
            raise ValueError("source latent and initial noise must be finite BCHW peers")
        strength = request.strength
        if (
            not isinstance(strength, torch.Tensor)
            or strength.ndim != 4
            or strength.shape[:2] != (source_latent.shape[0], 1)
            or min(strength.shape[-2:]) < 1
            or not torch.isfinite(strength).all()
            or (strength < 0).any()
            or (strength > 1).any()
        ):
            raise ValueError("source anchor strength must be finite BCHW, Bx1, in [0,1]")
        self.source = source_latent.detach().float().clone()
        self.noise = initial_noise.detach().float().clone()
        self.strength = F.interpolate(
            strength.detach().to(device=source_latent.device, dtype=torch.float32),
            size=source_latent.shape[-2:],
            mode="area",
        ).clone()
        self.mode = request.mode

    def control_velocity(self, model_velocity: torch.Tensor) -> torch.Tensor:
        """Mix one model update with source flow, without accumulating projections."""
        if self.mode != "velocity":
            raise ValueError("velocity control requires velocity anchor mode")
        if (
            not isinstance(model_velocity, torch.Tensor)
            or model_velocity.shape != self.source.shape
            or model_velocity.device != self.source.device
            or not torch.isfinite(model_velocity).all()
        ):
            raise ValueError("model velocity must be a finite same-shape tensor")
        source_velocity = self.source - self.noise
        result = model_velocity.float() + self.strength * (
            source_velocity - model_velocity.float()
        )
        return result.to(dtype=model_velocity.dtype)

    def project_hard_protection(
        self, current: torch.Tensor, time: float
    ) -> torch.Tensor:
        """Correct BF16 drift only where full source protection was requested."""
        if self.mode != "velocity":
            raise ValueError("hard protection requires velocity anchor mode")
        if (
            not isinstance(current, torch.Tensor)
            or current.shape != self.source.shape
            or current.device != self.source.device
            or not torch.isfinite(current).all()
        ):
            raise ValueError("current latent must be a finite same-shape tensor")
        time = float(time)
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("source anchor time must be finite and in [0,1]")
        target = ((1 - time) * self.noise + time * self.source).to(current.dtype)
        return torch.where(self.strength == 1, target, current)

    def project(self, current: torch.Tensor, time: float) -> torch.Tensor:
        if self.mode != "projection":
            raise ValueError("state projection requires projection anchor mode")
        if (
            not isinstance(current, torch.Tensor)
            or current.shape != self.source.shape
            or current.device != self.source.device
            or not torch.isfinite(current).all()
        ):
            raise ValueError("current latent must be a finite same-shape tensor")
        time = float(time)
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("source anchor time must be finite and in [0,1]")
        target = (1 - time) * self.noise + time * self.source
        result = current.float() + self.strength * (target - current.float())
        return result.to(dtype=current.dtype)
