"""Same-state, real-Euler lookahead; only the final observation is extrapolated.

The proposal is executed for one actual step. Later steps recompute the field,
without differentiating through the model or mutating the live scheduler.
"""

import copy

import torch


class EulerLookahead:
    def __init__(self, scheduler, predict, decode, dtype, *, horizon=1):
        if type(horizon) is not int or horizon not in (1, 2):
            raise ValueError("lookahead horizon must be 1 or 2")
        self.scheduler = scheduler
        self.predict, self.decode = predict, decode
        self.dtype, self.horizon = dtype, horizon

    @torch.no_grad()
    def __call__(self, state, velocity, index):
        times = self.scheduler._timesteps
        count = len(times) - 1
        if type(index) is not int or not 0 <= index < count:
            raise ValueError("invalid lookahead index")
        # step() only changes _step_index. Its timeline/config are read-only.
        scheduler = copy.copy(self.scheduler)
        scheduler._step_index = index
        current = state.detach().clone()
        field = velocity.detach().clone()
        first = None
        predictions = 0
        end = min(index + self.horizon, count)
        for j in range(index, end):
            if j != index:
                field = self.predict(current, j)
                predictions += 1
            current = scheduler.step(field, times[j], current, return_dict=False)[0].to(
                self.dtype
            )
            if first is None:
                first = current.detach().clone()
        endpoint = current.float()
        if end < count:
            field = self.predict(current, end)
            predictions += 1
            endpoint = endpoint + (1 - float(times[end])) * field.float()
        return {
            "image": self.decode(endpoint),
            "first_state": first,
            "state": current,
            "time": float(times[end]),
            "prediction_calls": predictions,
        }
