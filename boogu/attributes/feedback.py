"""Same-time observations for bounded, inference-only feedback acceptance.

The callback owns real scheduler/CFG execution. It must be pure and apply the
candidate velocity only on its first step. This module never differentiates
through, or optimizes, the lookahead trajectory.
"""

import math

import torch

REJECTION_REASONS = (
    "gradient",
    "nonfinite",
    "global",
    "local",
    "edit",
    "budget",
    "local_caps",
    "local_objective",
    "lookahead_caps",
    "lookahead_objective",
    "lookahead_goal",
    "zero_realized",
)


class LookaheadFeedback:
    def __init__(self, callback, observer, state, velocity, index, t, record):
        self.callback, self.observer = callback, observer
        self.state, self.index, self.t, self.record = state, index, t, record
        self.goal = getattr(observer, "goal_errors", None)
        if self.goal is not None and not callable(self.goal):
            raise ValueError("goal_errors must be callable when provided")
        record["goal_available"] = self.goal is not None
        self.baseline = self.observe(velocity)
        self.current = self.baseline
        if self.baseline is not None:
            self.caps = torch.maximum(self.baseline["errors"], observer.tolerances)
            record.update(
                lookahead_before=self.baseline["errors"].tolist(),
                lookahead_after=self.baseline["errors"].tolist(),
                lookahead_after_available=True,
                lookahead_caps=self.caps.tolist(),
                lookahead_time=self.baseline["time"],
                lookahead_objective_before=self.baseline["objective"],
                lookahead_objective_after=self.baseline["objective"],
                goal_before=self.baseline["goals"].tolist(),
                goal_after=self.baseline["goals"].tolist(),
            )

    def observe(self, velocity):
        with torch.no_grad():
            result = self.callback(self.state, velocity, self.index)
            if not isinstance(result, dict):
                raise TypeError("lookahead must return an observation dictionary")
            required = {"image", "first_state", "state", "time", "prediction_calls"}
            if not required <= result.keys():
                raise ValueError("lookahead observation is missing required fields")
            calls = result["prediction_calls"]
            if type(calls) is not int or calls < 0:
                raise ValueError("invalid lookahead prediction_calls")
            self.record["extra_prediction_calls"] += calls
            # The callback contract decodes exactly one final observation image.
            self.record["lookahead_decode_calls"] += 1
            image = result["image"]
            if (
                not isinstance(image, torch.Tensor)
                or image.shape != self.observer.source.shape
            ):
                raise ValueError("lookahead image does not match source coordinates")
            for name in ("first_state", "state"):
                value = result[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != self.state.shape
                ):
                    raise ValueError(
                        f"lookahead {name} does not match latent coordinates"
                    )
            time = float(result["time"])
            if not math.isfinite(time) or any(
                not torch.isfinite(result[name]).all()
                for name in ("image", "first_state", "state")
            ):
                return self.unavailable()
            if not self.t < time <= 1:
                raise ValueError("invalid lookahead observation time")
            errors = self.observer.errors(image)
            if (
                not isinstance(errors, torch.Tensor)
                or errors.ndim != 1
                or errors.shape != self.observer.tolerances.shape
            ):
                raise ValueError("lookahead keep errors must match observer items")
            goals = image.new_empty(0) if self.goal is None else self.goal(image)
            if not isinstance(goals, torch.Tensor) or goals.ndim != 1:
                raise ValueError("goal_errors must return a one-dimensional tensor")
            excess = (errors - self.observer.tolerances).clamp_min(0)
            objective = float(
                (self.observer.weights * excess.square()).sum()
                / self.observer.weights.sum()
            )
            if (
                not torch.isfinite(errors).all()
                or not torch.isfinite(goals).all()
                or not math.isfinite(objective)
            ):
                return self.unavailable()
            return {
                "errors": errors.detach().clone(),
                "goals": goals.detach().clone(),
                "first_state": result["first_state"].detach().clone(),
                "time": time,
                "objective": objective,
            }

    def unavailable(self):
        self.record["lookahead_nonfinite_observations"] += 1
        self.record["rejection_counts"]["nonfinite"] += 1

    def compare(self, velocity, candidate):
        observed = self.observe(velocity)
        if observed is None:
            candidate["reasons"].append("nonfinite")
            return None, False
        candidate.update(
            lookahead_after=observed["errors"].tolist(),
            lookahead_objective=observed["objective"],
            lookahead_time=observed["time"],
            goal_after=observed["goals"].tolist(),
            lookahead_current_available=self.current is not None,
        )
        if self.baseline is None:
            candidate["reasons"].append("unavailable_lookahead_baseline")
            return observed, False
        if not math.isclose(
            observed["time"], self.baseline["time"], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("lookahead branches must be evaluated at the same time")
        if observed["goals"].shape != self.baseline["goals"].shape:
            raise ValueError("lookahead goal_errors shape changed between branches")
        delta = observed["first_state"].float() - self.baseline["first_state"].float()
        delta_rms = float(delta.square().mean().sqrt())
        candidate["first_state_delta_rms"] = delta_rms
        reasons = []
        if not torch.all(observed["errors"] <= self.caps + 1e-7):
            reasons.append("lookahead_caps")
        if (
            self.current is None
            or not observed["objective"] < self.current["objective"]
        ):
            reasons.append("lookahead_objective")
        # Semantic goal quality is separate from the edit displacement guard.
        if self.goal is not None and not torch.all(
            observed["goals"] <= self.baseline["goals"] + 1e-7
        ):
            reasons.append("lookahead_goal")
        if delta_rms == 0:
            reasons.append("zero_realized")
        if not math.isfinite(delta_rms):
            reasons.append("nonfinite")
        candidate["reasons"].extend(reasons)
        for reason in reasons:
            self.record["rejection_counts"][reason] += 1
        return observed, not reasons

    def accept(self, observed):
        self.current = observed
        if observed is not None:
            self.record.update(
                lookahead_after=observed["errors"].tolist(),
                lookahead_after_available=True,
                lookahead_objective_after=observed["objective"],
                goal_after=observed["goals"].tolist(),
            )
        else:
            # Audit still accepts the legacy action when its observation fails.
            # Its final errors are unknown, rather than those of an older action.
            self.record.update(
                lookahead_after=[],
                lookahead_after_available=False,
                lookahead_objective_after=None,
                goal_after=[],
            )
