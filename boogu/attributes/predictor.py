"""Static, instruction-conditioned per-object attribute-role prediction.

The shared network accepts one object per batch item. For O objects, repeat source
and instruction inputs, flatten B*O, then reshape outputs to B,O,5,3,H,W (roles),
B,O,1,H,W (support), and B,O,5,H,W (applicability). Run once per edit. These outputs
are attribute supervision, independent of the legacy spatial PRI representation.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

ATTRIBUTE_ORDER = ("color", "structure", "texture", "glyph", "pixel")
ROLE_ORDER = ("K", "M", "U")
ARCHITECTURE = "attribute-role-21ch-stem1x1-128x3-crossattn4-v2"


@dataclass(frozen=True)
class AttributeRolePrediction:
    role_logits: torch.Tensor  # B,5,3,H,W; K (keep), M (modify), U (unspecified)
    support_logits: torch.Tensor  # B,1,H,W
    applicability_logits: torch.Tensor  # B,5,H,W
    valid_mask: torch.Tensor  # B,1,H,W, boolean
    role_temperatures: torch.Tensor  # five positive calibration temperatures

    @property
    def role_probabilities(self) -> torch.Tensor:
        probabilities = (
            self.role_logits.float()
            / self.role_temperatures.float().reshape(1, 5, 1, 1, 1)
        ).softmax(2)
        unknown = torch.zeros_like(probabilities)
        unknown[:, :, 2] = 1
        return torch.where(self.valid_mask.unsqueeze(2), probabilities, unknown)

    @property
    def object_support(self) -> torch.Tensor:
        return self.support_logits.float().sigmoid() * self.valid_mask

    @property
    def applicability(self) -> torch.Tensor:
        return self.applicability_logits.float().sigmoid() * self.valid_mask


class _SpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False)
        self.pointwise = nn.Conv2d(128, 128, 1, bias=False)
        self.norm = nn.GroupNorm(8, 128)
        self.activation = nn.SiLU()

    def forward(self, features):
        return features + self.activation(
            self.norm(self.pointwise(self.depthwise(features)))
        )


class AttributeRolePredictor(nn.Module):
    """21 spatial channels; three 128-wide scales and four-head token attention.

    Latents must already be on the source grid (16 channels). RGB and object hint
    are resized to that grid. Token embeddings are B,T,4096; token_mask is B,T,
    boolean True for usable tokens. valid_mask is boolean B,1,H,W. RGB follows the
    training dataset's fixed normalization; this module does not infer it.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(21, 128, 1)
        self.scales = nn.ModuleList(
            nn.Sequential(_SpatialBlock(), _SpatialBlock()) for _ in range(3)
        )
        self.downsample = nn.ModuleList(
            nn.Conv2d(128, 128, 3, stride=2, padding=1) for _ in range(2)
        )
        self.token_projection = nn.Linear(4096, 128)
        self.query_norm = nn.LayerNorm(128)
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128)
        )
        self.fusion = nn.ModuleList(
            nn.Sequential(nn.Conv2d(256, 128, 1), nn.SiLU()) for _ in range(2)
        )
        self.role_head = nn.Conv2d(128, 15, 1)
        self.support_head = nn.Conv2d(128, 1, 1)
        self.applicability_head = nn.Conv2d(128, 5, 1)
        self.register_buffer("role_temperatures", torch.ones(5))

    def set_role_temperatures(self, values) -> None:
        """Install held-out calibration values; this alone is not calibration evidence."""
        values = torch.as_tensor(
            values, device=self.role_temperatures.device, dtype=torch.float32
        )
        if (
            values.shape != (5,)
            or not torch.isfinite(values).all()
            or (values <= 0).any()
        ):
            raise ValueError("role temperatures must be five finite positive numbers")
        with torch.no_grad():
            self.role_temperatures.copy_(values)

    def forward(
        self,
        source_latent,
        source_rgb,
        object_hint,
        valid_mask,
        instruction_tokens,
        token_mask,
    ) -> AttributeRolePrediction:
        spatial = (
            ("source_latent", source_latent, 16),
            ("source_rgb", source_rgb, 3),
            ("object_hint", object_hint, 1),
        )
        for name, value, channels in spatial:
            if value.ndim != 4 or value.shape[1] != channels or min(value.shape) < 1:
                raise ValueError(
                    f"{name} must have shape B,{channels},H,W with nonempty dimensions"
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating point values")
        b, _, h, w = source_latent.shape
        if any(value.shape[0] != b for _, value, _ in spatial):
            raise ValueError("spatial batch sizes must match")
        if valid_mask.dtype != torch.bool or valid_mask.shape != (b, 1, h, w):
            raise ValueError("valid_mask must be boolean with shape B,1,H,W")
        if (object_hint < 0).any() or (object_hint > 1).any():
            raise ValueError("object_hint must lie in [0,1]")
        if (
            instruction_tokens.ndim != 3
            or instruction_tokens.shape[0] != b
            or instruction_tokens.shape[2] != 4096
        ):
            raise ValueError("instruction_tokens must have shape B,T,4096")
        if (
            token_mask.dtype != torch.bool
            or token_mask.shape != instruction_tokens.shape[:2]
        ):
            raise ValueError("token_mask must be boolean with shape B,T")
        if not token_mask.any(dim=1).all():
            raise ValueError("every instruction needs at least one valid token")
        if (
            not instruction_tokens.is_floating_point()
            or not torch.isfinite(instruction_tokens[token_mask]).all()
        ):
            raise ValueError(
                "valid instruction tokens must be finite floating point values"
            )
        device, dtype = self.stem.weight.device, self.stem.weight.dtype
        if any(
            value.device != device
            for value in (
                source_latent,
                source_rgb,
                object_hint,
                valid_mask,
                instruction_tokens,
                token_mask,
            )
        ):
            raise ValueError("all inputs and the predictor must share a device")
        rgb = F.interpolate(
            source_rgb.to(dtype), (h, w), mode="bilinear", align_corners=False
        )
        hint = F.interpolate(object_hint.to(dtype), (h, w), mode="nearest")
        features = torch.cat(
            (
                source_latent.to(dtype) * valid_mask,
                rgb * valid_mask,
                hint * valid_mask,
                valid_mask.to(dtype),
            ),
            dim=1,
        )
        features, skips = self.stem(features), []
        for index, scale in enumerate(self.scales):
            features = scale(features)
            if index < 2:
                skips.append(features)
                features = self.downsample[index](features)
        shape = features.shape
        queries = features.flatten(2).transpose(1, 2)
        tokens = self.token_projection(
            instruction_tokens.masked_fill(~token_mask[..., None], 0).to(dtype)
        )
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            tokens,
            tokens,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        queries = queries + attended
        queries = queries + self.feed_forward(queries)
        features = queries.transpose(1, 2).reshape(shape)
        for fusion, skip in zip(self.fusion, reversed(skips)):
            features = F.interpolate(
                features, skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = fusion(torch.cat((features, skip), dim=1))
        return AttributeRolePrediction(
            self.role_head(features).reshape(b, 5, 3, h, w),
            self.support_head(features),
            self.applicability_head(features),
            valid_mask,
            self.role_temperatures.clone(),
        )
