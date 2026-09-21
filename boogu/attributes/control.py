"""Finite direct-velocity correction; local proxies do not certify final quality."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .feedback import REJECTION_REASONS, LookaheadFeedback


def rms(x):
    return x.float().square().mean().sqrt()


def _inward_velocity(velocity, correction):
    """Cast without letting a coordinate exceed its continuous correction.

    The original velocity is representable in its own dtype. If nearest
    rounding overshoots the correction, its adjacent value toward that
    baseline is inside the interval. Actual tensor norms are still checked
    by the controller after this coordinate-wise rounding.
    """
    candidate = (velocity.detach().float() + correction).to(velocity.dtype)
    actual = candidate.float() - velocity.detach().float()
    overshoots = actual.abs() > correction.abs()
    return torch.where(
        overshoots, torch.nextafter(candidate, velocity.detach()), candidate
    )


@dataclass(frozen=True)
class ControlConfig:
    beta: float = 0.05
    kappa: float = 0.1
    local_kappa: float = 0.2
    start: float = 0.35
    end: float = 0.85
    interval: int = 2
    iterations: int = 2
    backtracks: int = 3
    step_size: float = 0.25
    regularization: float = 0.01
    edit_ratio: float = 0.05
    improvement_epsilon: float = 1e-8
    feedback: str = "local"
    lookahead_steps: int = 1
    proposal_mode: str = "quantized"

    def __post_init__(self):
        for name in (
            "beta",
            "kappa",
            "local_kappa",
            "start",
            "end",
            "step_size",
            "regularization",
            "edit_ratio",
            "improvement_epsilon",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid {name}")
        for name in ("interval", "iterations", "backtracks"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.start <= self.end <= 1:
            raise ValueError("invalid control time interval")
        if self.feedback not in ("local", "audit", "rollout"):
            raise ValueError("invalid feedback mode")
        if type(self.lookahead_steps) is not int or self.lookahead_steps not in (1, 2):
            raise ValueError("lookahead_steps must be 1 or 2")
        if self.proposal_mode not in ("legacy", "quantized"):
            raise ValueError("invalid proposal mode")


class BudgetController:
    def __init__(self, observer, decode, *, source_latent, initial_latent, config):
        if source_latent.shape != initial_latent.shape or initial_latent.shape[0] != 1:
            raise ValueError("control requires one same-canvas source latent")
        if (
            not torch.isfinite(source_latent).all()
            or not torch.isfinite(initial_latent).all()
        ):
            raise ValueError("nonfinite initial/source latent")
        self.observer, self.decode, self.config = observer, decode, config
        self.initial_budget = config.beta * float(
            rms(source_latent.float() - initial_latent.float())
        )
        self.remaining = self.initial_budget
        self.spent = 0.0
        self.records = []

    def step(self, z, velocity, *, t, dt, index, lookahead=None):
        t, dt = float(t), float(dt)
        if (
            not math.isfinite(t)
            or not math.isfinite(dt)
            or not 0 <= t < 1
            or dt <= 0
            or t + dt > 1 + 1e-6
        ):
            raise ValueError("invalid true integration time")
        if (
            z.shape != velocity.shape
            or z.shape[0] != 1
            or not torch.isfinite(z).all()
            or not torch.isfinite(velocity).all()
        ):
            raise ValueError("invalid state or velocity")
        record = {
            "index": index,
            "t": t,
            "dt": dt,
            "accepted": 0,
            "debit": 0.0,
            "remaining": self.remaining,
            "reason": "inactive",
            "before": [],
            "after": [],
            "caps": [],
            "decode_calls": 0,
            "nonfinite_observations": 0,
            "feedback": self.config.feedback,
            "proposal_mode": self.config.proposal_mode,
            "lookahead_steps": self.config.lookahead_steps,
            "lookahead_decode_calls": 0,
            "extra_prediction_calls": 0,
            "lookahead_nonfinite_observations": 0,
            "lookahead_before": [],
            "lookahead_after": [],
            "lookahead_after_available": False,
            "lookahead_objective_before": None,
            "lookahead_objective_after": None,
            "lookahead_caps": [],
            "lookahead_time": None,
            "goal_available": callable(getattr(self.observer, "goal_errors", None)),
            "goal_before": [],
            "goal_after": [],
            "goal_numerical_epsilon": 1e-7,
            "candidates": [],
            "rejection_counts": dict.fromkeys(REJECTION_REASONS, 0),
        }
        self.records.append(record)
        cfg = self.config
        if cfg.proposal_mode == "quantized":
            record["rejection_counts"]["duplicate"] = 0
        if self.remaining <= 0 or not cfg.start <= t <= cfg.end or index % cfg.interval:
            return velocity
        if not self.observer.items or not self.observer.edit.any():
            record["reason"] = "unavailable_constraints"
            return velocity
        scale = float(rms(velocity))
        radius = min(cfg.kappa * scale, self.remaining / dt)
        local = cfg.local_kappa * scale
        if radius <= 0 or local <= 0:
            return velocity
        if cfg.feedback != "local" and not callable(lookahead):
            raise ValueError("active nonlocal feedback requires a lookahead callback")
        edit = F.interpolate(
            self.observer.edit.to(z.device), size=z.shape[-2:], mode="area"
        )
        edit_limit = cfg.edit_ratio * (1 - t) * scale
        denom = (z.shape[1] * edit.sum()).sqrt()
        record["edit_limit"] = edit_limit
        record["radius"] = radius
        record["local_radius"] = local
        baseline = velocity.detach().float()
        state = z.detach().float()
        feedback = None
        if cfg.feedback != "local":
            feedback = LookaheadFeedback(
                lookahead, self.observer, z, velocity, index, t, record
            )
            if feedback.baseline is None and cfg.feedback == "rollout":
                record["reason"] = "unavailable_lookahead_observation"
                return velocity

        def observe(q):
            record["decode_calls"] += 1
            endpoint = state + (1 - t) * (baseline + q)
            image = self.decode(endpoint)
            # A malformed decoder contract must still fail loudly. Numerical
            # unavailability is a separate, recoverable observation outcome.
            if image.shape != self.observer.source.shape:
                raise ValueError("decoded image does not match source coordinates")
            if not torch.isfinite(image).all():
                record["nonfinite_observations"] += 1
                record["rejection_counts"]["nonfinite"] += 1
                return None
            errors = self.observer.errors(image)
            if not torch.isfinite(errors).all():
                record["nonfinite_observations"] += 1
                record["rejection_counts"]["nonfinite"] += 1
                return None
            return errors

        def loss(errors, q):
            excess = (errors - self.observer.tolerances).clamp_min(0)
            keep = (
                self.observer.weights * excess.square()
            ).sum() / self.observer.weights.sum()
            # mean(q**2), not sqrt(mean(q**2))**2: the latter has NaN
            # autograd at the required q=0 starting point.
            return keep + cfg.regularization * q.float().square().mean() / (
                radius * radius
            )

        def guard(q):
            return float(((1 - t) * q * edit.sqrt()).norm() / denom)

        def feasible(q):
            q = q * torch.clamp(
                local / q.square().mean(1, keepdim=True).sqrt().clamp_min(1e-12), max=1
            )
            return q * torch.clamp(
                q.new_tensor(radius) / rms(q).clamp_min(1e-12), max=1
            )

        with torch.enable_grad():
            current = torch.zeros_like(state)
            before = observe(current)
            if before is None:
                record["reason"] = "unavailable_observation"
                return velocity
            before = before.detach()
            caps = torch.maximum(before, self.observer.tolerances)
            record.update(
                before=before.tolist(),
                after=before.tolist(),
                caps=caps.tolist(),
                edit_displacement=0.0,
            )
            current_objective = float(loss(before, current))
            if current_objective == 0:
                record["reason"] = "within_tolerance"
                return velocity
            final_velocity = velocity
            seen_velocities = (
                [velocity.detach()] if cfg.proposal_mode == "quantized" else []
            )
            for iteration in range(cfg.iterations):
                q = current.detach().requires_grad_(True)
                errors = observe(q)
                if errors is None:
                    break
                objective = loss(errors, q)
                if not torch.isfinite(objective):
                    record["rejection_counts"]["nonfinite"] += 1
                    break
                g = torch.autograd.grad(objective, q)[0]
                if not torch.isfinite(g).all() or float(rms(g)) == 0:
                    record["rejection_counts"]["gradient"] += 1
                    if not torch.isfinite(g).all():
                        record["rejection_counts"]["nonfinite"] += 1
                    break
                accepted = False
                if cfg.proposal_mode == "quantized":
                    # Project once, then backtrack toward this bounded target
                    # from the last accepted correction. Scaling a large raw
                    # gradient before projection can repeat a saturated point.
                    target = feasible(
                        current
                        - cfg.step_size * radius * g.detach() / rms(g).clamp_min(1e-12)
                    )
                for h in range(cfg.backtracks):
                    if cfg.proposal_mode == "quantized":
                        proposed = current + (target - current) / (2**h)
                        candidate_velocity = _inward_velocity(velocity, proposed)
                    else:
                        proposed = feasible(
                            q.detach()
                            - cfg.step_size
                            * radius
                            / (2**h)
                            * g.detach()
                            / rms(g).clamp_min(1e-12)
                        )
                        candidate_velocity = (baseline + proposed).to(velocity.dtype)
                    actual = candidate_velocity.float() - baseline
                    debit = dt * float(rms(actual))
                    displacement = guard(actual)
                    candidate = {
                        "iteration": iteration,
                        "backtrack": h,
                        "local_after": [],
                        "local_objective": None,
                        "lookahead_after": [],
                        "lookahead_objective": None,
                        "first_state_delta_rms": None,
                        "predicted_direct_delta_rms": debit,
                        "edit_displacement": displacement,
                        "local_passed": False,
                        "lookahead_passed": None,
                        "accepted": False,
                        "reasons": [],
                    }
                    record["candidates"].append(candidate)
                    if cfg.proposal_mode == "quantized":
                        if any(
                            torch.equal(candidate_velocity, old)
                            for old in seen_velocities
                        ):
                            candidate["reasons"].append("duplicate")
                            record["rejection_counts"]["duplicate"] += 1
                            continue
                        seen_velocities.append(candidate_velocity.detach())
                    checks = {
                        "nonfinite": not torch.isfinite(actual).all()
                        or not math.isfinite(debit)
                        or not math.isfinite(displacement),
                        "global": float(rms(actual)) > radius + 1e-8,
                        "local": float(actual.square().mean(1).sqrt().max())
                        > local + 1e-8,
                        "budget": debit > self.remaining,
                        "edit": displacement > edit_limit + 1e-8,
                    }
                    for reason, failed in checks.items():
                        if failed:
                            candidate["reasons"].append(reason)
                            record["rejection_counts"][reason] += 1
                    if candidate["reasons"]:
                        continue
                    with torch.no_grad():
                        errors = observe(actual)
                        if errors is None:
                            candidate["reasons"].append("nonfinite")
                            continue
                        value = float(loss(errors, actual))
                    candidate.update(local_after=errors.tolist(), local_objective=value)
                    if not math.isfinite(value):
                        candidate["reasons"].append("nonfinite")
                    if not torch.all(errors <= caps + 1e-7):
                        candidate["reasons"].append("local_caps")
                    if not value < current_objective - cfg.improvement_epsilon:
                        candidate["reasons"].append("local_objective")
                    for reason in candidate["reasons"]:
                        record["rejection_counts"][reason] += 1
                    if not candidate["reasons"]:
                        candidate["local_passed"] = True
                        observed = None
                        if feedback is not None:
                            observed, passed = feedback.compare(
                                candidate_velocity, candidate
                            )
                            candidate["lookahead_passed"] = passed
                            if cfg.feedback == "rollout" and not passed:
                                continue
                        current = actual.detach()
                        final_velocity = candidate_velocity.detach()
                        current_objective = value
                        accepted = True
                        record["accepted"] += 1
                        candidate["accepted"] = True
                        if feedback is not None:
                            feedback.accept(observed)
                        record.update(
                            after=errors.tolist(), edit_displacement=displacement
                        )
                        break
                if not accepted:
                    break
        debit = dt * float(rms(current))
        self.remaining -= debit
        self.spent += debit
        record.update(
            debit=debit,
            remaining=self.remaining,
            reason="accepted" if record["accepted"] else "no_feasible_descent",
        )
        return final_velocity
