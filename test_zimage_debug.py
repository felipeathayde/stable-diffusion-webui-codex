"""Quick Z Image debug test - minimal generation to check logs."""
import sys
sys.path.insert(0, "C:\\Users\\lucas\\OneDrive\\Documentos\\stable-diffusion-webui-codex")

import torch
from apps.backend.runtime.models.loader import load_model_checkpoint
from apps.backend.engines.zimage.zimage import ZImageEngine
from apps.backend.runtime.models.types import CheckpointInfo
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Load Z Image model
checkpoint_path = "C:\\Users\\lucas\\OneDrive\\Documentos\\stable-diffusion-webui-codex\\models\\zimage\\z_image_turbo-Q8_0.gguf"

print("=" * 80)
print("Z IMAGE DEBUG TEST - Checking timestep→model_output chain")
print("=" * 80)

# This is just to check model loading and forward pass
# We'll rely on the actual backend for full integration
print("\nLogs will show:")
print("1. timestep (sigma) input")
print("2. t_inv (1-sigma) after inversion")  
print("3. model output BEFORE negation")
print("4. model output AFTER negation")
print("\nRun an actual generation from the UI to see these logs.")
print("=" * 80)
