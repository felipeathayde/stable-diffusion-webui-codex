"""Condition helpers for diffusion sampling.

Invariants (enforced):
- For dict-based conditioning (SDXL/SD3+), keys must include 'crossattn' (B,S,C) and 'vector' (B,V).
- For tensor-based conditioning (SD15/SD20 legacy), input must be (B,S,C) and is treated as cross-attn only.
- No silent coercions: violations raise ValueError with the root cause.
"""

import math
import logging
import torch


def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    if tensor.shape[0] < batch_size:
        reps = [math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1)
        return tensor.repeat(reps)[:batch_size]
    return tensor


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Condition:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class ConditionCrossAttn(Condition):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4:
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)
            out.append(c)
        return torch.cat(out)


class ConditionConstant(Condition):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


def compile_conditions(cond):
    if cond is None:
        return None

    if isinstance(cond, torch.Tensor):
        # Legacy path: only cross-attn provided.
        if cond.ndim != 3:
            raise ValueError(f"cross-attn tensor must be 3D (B,S,C); got shape={tuple(cond.shape)}")
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond),
            ),
        )
        return [result]

    # Dict-based path: require keys and shapes
    if not isinstance(cond, dict):
        raise TypeError(f"conditioning must be Tensor or dict; got {type(cond).__name__}")
    if 'crossattn' not in cond:
        raise ValueError("conditioning dict missing required key 'crossattn'")
    if 'vector' not in cond:
        raise ValueError("conditioning dict missing required key 'vector' (pooled/global embedding)")

    cross_attn = cond['crossattn']
    pooled_output = cond['vector']

    if not isinstance(cross_attn, torch.Tensor) or cross_attn.ndim != 3:
        raise ValueError(f"'crossattn' must be a 3D tensor (B,S,C); got {type(cross_attn).__name__} shape={getattr(cross_attn,'shape',None)}")
    if not isinstance(pooled_output, torch.Tensor) or pooled_output.ndim != 2:
        raise ValueError(f"'vector' must be a 2D tensor (B,V); got {type(pooled_output).__name__} shape={getattr(pooled_output,'shape',None)}")

    result = dict(
        cross_attn=cross_attn,
        pooled_output=pooled_output,
        model_conds=dict(
            c_crossattn=ConditionCrossAttn(cross_attn),
            y=Condition(pooled_output),
        ),
    )

    if 'guidance' in cond:
        result['model_conds']['guidance'] = Condition(cond['guidance'])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "compiled conditions: cross_attn=%s pooled=%s",
            tuple(cross_attn.shape), tuple(pooled_output.shape)
        )
    return [result]


def compile_weighted_conditions(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = compile_conditions(feed)
        h[0]['strength'] = current_weight
        results += h

    return results
logger = logging.getLogger("backend.runtime.sampling.condition")
