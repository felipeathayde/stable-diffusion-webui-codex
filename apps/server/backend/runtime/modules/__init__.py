"""Additional model helpers (k-diffusion, predictors)."""

from .k_model import KModel
from .k_prediction import (
    AbstractPrediction,
    Prediction,
    PredictionFlux,
    k_prediction_from_diffusers_scheduler,
    rescale_zero_terminal_snr_sigmas,
)

# Backward compatibility: some code may expect FlowMatchEulerPrediction
# which maps to our PredictionFlux implementation.
FlowMatchEulerPrediction = PredictionFlux
from . import k_diffusion_extra

__all__ = [
    "AbstractPrediction",
    "FlowMatchEulerPrediction",
    "PredictionFlux",
    "KModel",
    "Prediction",
    "k_diffusion_extra",
    "k_prediction_from_diffusers_scheduler",
    "rescale_zero_terminal_snr_sigmas",
]
