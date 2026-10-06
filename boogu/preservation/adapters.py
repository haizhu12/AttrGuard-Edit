"""Reversible source-conditioned residuals attached to native single-stream layers."""

import weakref
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .conditioning import ATTRIBUTE_NAMES, SourceCondition, _finite_float

_MISSING = object()
_OWNER = "_source_preservation_owner"


def _same_metadata(actual, expected):
    """Compare JSON-like metadata without Python's bool/int coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_metadata(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_metadata(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


class _ResidualBank(nn.Module):
    def __init__(self, hidden_size, source_size, rank):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.W_h = nn.Linear(hidden_size, rank)
        self.W_s = nn.Linear(source_size, rank)
        self.W_r = nn.Linear(15, rank)
        self.W_t = nn.Linear(1, rank)
        self.W_out = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.W_out.weight)
        nn.init.zeros_(self.W_out.bias)

    def _apply(self, fn, recurse=True):
        # Probe only an empty tensor: casting a parameter and casting it back
        # would permanently round FP32 master weights and optimizer gradients.
        def preserve(value):
            probe = fn(torch.empty(0, device=value.device, dtype=value.dtype))
            return value.to(device=probe.device, dtype=torch.float32)

        return super()._apply(preserve, recurse=recurse)

    def forward(self, generated, source, roles, time):
        with torch.autocast(device_type=generated.device.type, enabled=False):
            generated = generated.float()
            hidden = self.W_h(self.norm(generated)) + self.W_s(source.float())
            hidden = hidden + self.W_r(roles.float()) + self.W_t(time.float())
            return self.W_out(F.silu(hidden))


@dataclass(frozen=True)
class _Invocation:
    source: torch.Tensor
    roles: torch.Tensor
    time: torch.Tensor
    start: int
    count: int


@dataclass
class _BackwardRecord:
    done: bool = False


class SourceConditionedTransformer(nn.Module):
    """Freeze a native Boogu transformer and train small aligned residual banks.

    v1 supports B=1, one same-size reference, compact all-valid instruction
    tokens, and normalized time in [0,1]. Native files and layer lists are
    unchanged. Direct calls to ``base`` outside wrapper forwards bypass banks.

    ``close()`` restores checkpoint settings and removes only owned hooks; the
    base stays frozen. Finish backward before closing. A graph whose output
    backward has not run is guarded even when only its downstream loss remains;
    retained graphs / backward through an intermediate tensor must also be
    released by the caller before closing.
    Export only ``export_state()`` for portable adapter checkpoints.
    """

    def __init__(
        self,
        base,
        block_indices=(23, 27, 31),
        rank=32,
        checkpointing=True,
        use_source=True,
        use_roles=True,
    ):
        super().__init__()
        if hasattr(base, _OWNER):
            raise RuntimeError("source preservation is already installed on this base")
        self._check_supported(base)
        indices = tuple(block_indices)
        layers = getattr(base, "single_stream_layers", None)
        if layers is None or not indices or len(set(indices)) != len(indices):
            raise ValueError(
                "block_indices must select distinct native single-stream layers"
            )
        if any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(layers)
            for i in indices
        ):
            raise ValueError(
                "block_indices are outside the native single-stream layers"
            )
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        for name, flag in (
            ("checkpointing", checkpointing),
            ("use_source", use_source),
            ("use_roles", use_roles),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"{name} must be boolean")
        self.base = base
        self.block_indices = indices
        self.rank = rank
        self.checkpointing = checkpointing
        self.use_source = use_source
        self.use_roles = use_roles
        self.hidden_size = int(base.config.hidden_size)
        self.in_channels = int(base.config.in_channels)
        self.patch_size = int(base.config.patch_size)
        self.banks = nn.ModuleDict(
            {
                str(i): _ResidualBank(
                    self.hidden_size, self.patch_size**2 * self.in_channels, rank
                )
                for i in indices
            }
        )
        first_parameter = next(base.parameters(), None)
        if first_parameter is not None:
            self.banks.to(device=first_parameter.device)
        self._context = ContextVar(f"source_preservation_{id(self)}", default=None)
        self._handles = []
        self._pending = []
        self._closed = False
        self._previous_flag = getattr(base, "gradient_checkpointing", _MISSING)
        self._previous_function = getattr(
            base, "_gradient_checkpointing_func", _MISSING
        )
        self._function_was_local = "_gradient_checkpointing_func" in base.__dict__
        previous_training = {module: module.training for module in base.modules()}
        previous_grad = {
            parameter: parameter.requires_grad for parameter in base.parameters()
        }
        try:
            base.requires_grad_(False)
            base.eval()
            base.gradient_checkpointing = checkpointing
            base._gradient_checkpointing_func = self._checkpoint
            setattr(base, _OWNER, self._context)
            for index in indices:
                self._handles.append(
                    layers[index].register_forward_hook(self._make_hook(str(index)))
                )
        except BaseException:
            self._uninstall()
            for parameter, requires_grad in previous_grad.items():
                parameter.requires_grad_(requires_grad)
            for module, training in previous_training.items():
                module.training = training
            raise

    @staticmethod
    def _check_supported(base):
        for module in base.modules():
            if any(
                getattr(module, flag, False)
                for flag in (
                    "enable_teacache",
                    "enable_taylorseer",
                    "enable_teacache_for_all_layers",
                    "enable_taylorseer_for_all_layers",
                )
            ):
                raise RuntimeError(
                    "TeaCache / TaylorSeer caching is unsupported by source preservation"
                )
            if any(
                getattr(module, name, None) is not None
                for name in (
                    "_hf_hook",
                    "_diffusers_hook",
                    "_group_offload_hook",
                    "_cache_context",
                )
            ):
                raise RuntimeError(
                    "offload / caching hooks are unsupported by source preservation"
                )
        if getattr(base, "hf_device_map", None):
            raise RuntimeError(
                "offload device maps are unsupported by source preservation"
            )

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def _make_hook(self, key):
        def inject(module, args, output):
            invocation = self._context.get()
            if invocation is None:
                return output
            if self._closed:
                raise RuntimeError(
                    "source preservation was closed before backward completed"
                )
            if not isinstance(output, torch.Tensor) or output.shape != (
                1,
                invocation.start + invocation.count,
                self.hidden_size,
            ):
                raise ValueError(
                    "native joint token layout does not match compact text + reference + generated grid"
                )
            start = invocation.start
            generated = output[:, start : start + invocation.count]
            residual = self.banks[key](
                generated, invocation.source, invocation.roles, invocation.time
            )
            return torch.cat(
                (output[:, :start], generated + residual.to(generated.dtype)), dim=1
            )

        return inject

    def _checkpoint(self, module, *args, **kwargs):
        # Capture the immutable invocation NOW, not during backward, when the
        # wrapper's dynamic scope has already reset to its previous context.
        invocation = self._context.get()

        def run(*inputs):
            token = self._context.set(invocation)
            try:
                return module(*inputs, **kwargs)
            finally:
                self._context.reset(token)

        # A frozen prefix creates no autograd graph and gains nothing from cp.
        # An adapted block must be checkpointed even with frozen input; every
        # subsequent native block must retain the adapted input's gradient.
        adapted = invocation is not None and any(
            module is self.base.single_stream_layers[i] for i in self.block_indices
        )
        needs_grad = any(
            isinstance(value, torch.Tensor) and value.requires_grad for value in args
        )
        if not torch.is_grad_enabled() or (not needs_grad and not adapted):
            return run(*args)
        return checkpoint(run, *args, use_reentrant=False)

    def _snapshot(self, condition, kwargs):
        if not isinstance(condition, SourceCondition):
            raise TypeError("preservation must be a SourceCondition")
        hidden = kwargs.get("hidden_states")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 4
            or hidden.shape[0] != 1
        ):
            raise ValueError("source preservation requires hidden_states [1,C,H,W]")
        if (
            tuple(hidden.shape) != condition.latent_shape
            or hidden.shape[1] != self.in_channels
        ):
            raise ValueError(
                "generated latent shape must match the source latent and native channels"
            )
        if condition.patch_size != self.patch_size or any(
            d % self.patch_size for d in hidden.shape[-2:]
        ):
            raise ValueError(
                "source and generated patch grids must match the native patch_size"
            )
        refs = kwargs.get("ref_image_hidden_states")
        if (
            not isinstance(refs, (list, tuple))
            or len(refs) != 1
            or not isinstance(refs[0], (list, tuple))
            or len(refs[0]) != 1
        ):
            raise ValueError("source preservation requires exactly one reference image")
        if not isinstance(refs[0][0], torch.Tensor) or tuple(refs[0][0].shape) != tuple(
            hidden.shape[1:]
        ):
            raise ValueError(
                "reference and generated latent shapes must match without resizing"
            )
        mask = kwargs.get("instruction_attention_mask")
        text = kwargs.get("instruction_hidden_states")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.ndim != 2
            or mask.shape[0] != 1
            or mask.shape[1] < 1
        ):
            raise ValueError("instruction_attention_mask must have shape [1,T]")
        if not torch.all(mask == 1).item():
            raise ValueError(
                "v1 requires compact all-valid instruction tokens; padded text is unsupported"
            )
        features = text if isinstance(text, (list, tuple)) else (text,)
        if not features or any(
            not isinstance(feature, torch.Tensor)
            or feature.ndim != 3
            or feature.shape[0] != 1
            or feature.shape[1] != mask.shape[1]
            for feature in features
        ):
            raise ValueError(
                "instruction embeddings must match the actual native text mask"
            )
        time = kwargs.get("timestep")
        _finite_float(time, "timestep")
        if time.numel() != 1 or ((time < 0) | (time > 1)).any().item():
            raise ValueError("timestep must contain one normalized value in [0,1]")
        count = (hidden.shape[-2] // self.patch_size) * (
            hidden.shape[-1] // self.patch_size
        )
        _finite_float(condition.source_tokens, "source_tokens")
        _finite_float(condition.roles, "roles")
        if condition.source_tokens.shape != (
            1,
            count,
            self.patch_size**2 * self.in_channels,
        ):
            raise ValueError("source_tokens do not match the native latent patch grid")
        if condition.roles.shape != (1, count, 15):
            raise ValueError("roles must have shape [1,N,15]")
        roles = (
            condition.roles.detach()
            .to(device=hidden.device, dtype=torch.float32)
            .clone()
        )
        if ((roles < 0) | (roles > 1)).any().item() or not torch.allclose(
            roles.reshape(1, count, 5, 3).sum(-1),
            torch.ones(1, count, 5, device=hidden.device),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("roles must contain K/M/U probabilities summing to one")
        source = (
            condition.source_tokens.detach()
            .to(device=hidden.device, dtype=torch.float32)
            .clone()
        )
        if not self.use_source:
            source.zero_()
        if not self.use_roles:
            roles.zero_()
            roles[..., 2::3] = 1
        return _Invocation(
            source,
            roles,
            time.detach()
            .to(device=hidden.device, dtype=torch.float32)
            .reshape(1, 1, 1)
            .clone(),
            int(mask.sum().item()) + count,
            count,
        )

    def forward(self, *, preservation=None, **native_kwargs):
        if self._closed:
            raise RuntimeError("source preservation wrapper is closed")
        if preservation is not None:
            self._check_supported(self.base)
        invocation = (
            None
            if preservation is None
            else self._snapshot(preservation, native_kwargs)
        )
        token = self._context.set(invocation)
        try:
            result = self.base(**native_kwargs)
        finally:
            self._context.reset(token)
        if invocation is not None:
            self._track_backward(result)
        return result

    def _track_backward(self, result):
        result = getattr(result, "sample", result)
        values = result if isinstance(result, (list, tuple)) else (result,)
        for value in values:
            if isinstance(value, torch.Tensor) and value.requires_grad:
                # The autograd hook retains the record for the lifetime of the
                # graph, even when Python no longer retains the output tensor.
                record = _BackwardRecord()
                self._pending.append(weakref.ref(record))

                def backward_started(gradient, record=record):
                    # Do not allow hooks to disappear during checkpoint replay.
                    torch.autograd.Variable._execution_engine.queue_callback(
                        lambda: setattr(record, "done", True)
                    )
                    return gradient

                value.register_hook(backward_started)
        self._pending = [r for r in self._pending if r() is not None and not r().done]

    def adapter_parameters(self):
        """Named trainable parameters only; suitable for optimizer construction."""
        return {
            f"banks.{name}": parameter
            for name, parameter in self.banks.named_parameters()
        }

    def _metadata(self):
        return {
            "schema": "boogu_source_preservation_v1",
            "block_indices": list(self.block_indices),
            "rank": self.rank,
            "hidden_size": self.hidden_size,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "attributes": list(ATTRIBUTE_NAMES),
            "role_order": ["keep", "modify", "unknown"],
            "use_source": self.use_source,
            "use_roles": self.use_roles,
        }

    def export_state(self):
        values = self.adapter_parameters()
        if any(
            p.dtype != torch.float32 or not torch.isfinite(p).all().item()
            for p in values.values()
        ):
            raise ValueError("adapter state must be finite FP32")
        return {
            "metadata": self._metadata(),
            "adapters": {name: p.detach().cpu().clone() for name, p in values.items()},
        }

    def load_state(self, state):
        """Validate the entire state before copying any parameter."""
        if (
            type(state) is not dict
            or set(state) != {"metadata", "adapters"}
            or not _same_metadata(state["metadata"], self._metadata())
        ):
            raise ValueError(
                "adapter metadata does not exactly match this configuration"
            )
        parameters = self.adapter_parameters()
        values = state["adapters"]
        if not isinstance(values, dict) or set(values) != set(parameters):
            raise ValueError("adapter tensor names do not exactly match")
        staged = {}
        for name, parameter in parameters.items():
            value = values[name]
            if (
                type(value) is not torch.Tensor
                or value.layout != torch.strided
                or value.is_meta
                or value.dtype != torch.float32
                or value.shape != parameter.shape
                or not torch.isfinite(value).all().item()
            ):
                raise ValueError(
                    f"adapter {name} must have the expected shape and finite FP32 values"
                )
            staged[name] = value.detach().to(device=parameter.device).clone()
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(staged[name])

    def _uninstall(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._previous_flag is _MISSING:
            self.base.__dict__.pop("gradient_checkpointing", None)
        else:
            self.base.gradient_checkpointing = self._previous_flag
        if self._function_was_local:
            self.base._gradient_checkpointing_func = self._previous_function
        else:
            self.base.__dict__.pop("_gradient_checkpointing_func", None)
        if getattr(self.base, _OWNER, None) is self._context:
            delattr(self.base, _OWNER)

    def close(self):
        if self._closed:
            return
        if self._context.get() is not None or any(
            r() is not None and not r().done for r in self._pending
        ):
            raise RuntimeError(
                "finish outstanding backward before closing source preservation"
            )
        self._uninstall()
        self._closed = True
