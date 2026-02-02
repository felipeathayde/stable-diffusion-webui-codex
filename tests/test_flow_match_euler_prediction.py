import math

import pytest


torch = pytest.importorskip("torch")


from apps.backend.runtime.sampling_adapters.prediction import FlowMatchEulerPrediction  # noqa: E402


def _base_sigmas(pseudo_timestep_range: int) -> "torch.Tensor":
    return torch.arange(1, pseudo_timestep_range + 1, dtype=torch.float32) / float(pseudo_timestep_range)


def test_flow_match_dynamic_exponential_uses_exp_mu():
    pseudo = 512
    pred = FlowMatchEulerPrediction(
        seq_len=4096,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
        pseudo_timestep_range=pseudo,
        shift=None,
        time_shift_type="exponential",
    )

    assert pred.mu is not None
    assert pred.shift == pytest.approx(math.exp(pred.mu))

    base = _base_sigmas(pseudo)
    alpha = float(pred.shift)
    expected = alpha * base / (1.0 + (alpha - 1.0) * base)
    torch.testing.assert_close(pred.sigmas, expected)

    # Regression guard: the historical bug used mu directly as alpha (no exp).
    wrong_alpha = float(pred.mu)
    wrong = wrong_alpha * base / (1.0 + (wrong_alpha - 1.0) * base)
    assert not torch.allclose(pred.sigmas, wrong)


@pytest.mark.parametrize("alpha", [1.0, 3.0, 6.0])
def test_flow_match_fixed_shift_matches_formula(alpha: float):
    pseudo = 256
    pred = FlowMatchEulerPrediction(pseudo_timestep_range=pseudo, shift=alpha)

    assert pred.mu is None
    assert pred.shift == pytest.approx(alpha)

    base = _base_sigmas(pseudo)
    expected = alpha * base / (1.0 + (alpha - 1.0) * base)
    torch.testing.assert_close(pred.sigmas, expected)


def test_flow_match_invalid_time_shift_type_raises():
    with pytest.raises(ValueError, match="time_shift_type"):
        FlowMatchEulerPrediction(shift=None, time_shift_type="nope")


def test_flow_match_shift_must_be_positive():
    with pytest.raises(ValueError, match="shift must be > 0"):
        FlowMatchEulerPrediction(shift=0.0)

