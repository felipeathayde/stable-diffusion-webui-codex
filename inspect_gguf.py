#!/usr/bin/env python
"""Quick inspection of GGUF file contents."""
import sys
sys.path.insert(0, '.')

from apps.backend.quantization.gguf import GGUFReader

path = sys.argv[1] if len(sys.argv) > 1 else 'models/flux/flux1-dev-Q2_K.gguf'
print(f"Loading: {path}")

reader = GGUFReader(path)
print(f"\n=== GGUF Metadata ===")
for key, field in list(reader.fields.items())[:20]:
    if field.types and field.parts:
        try:
            val = field.parts[field.data[0]] if field.data[0] >= 0 else "<complex>"
            if hasattr(val, 'tobytes'):
                val = val.tobytes().decode('utf-8', errors='replace')[:50]
            print(f"  {key}: {val}")
        except:
            print(f"  {key}: <error reading>")

print(f"\n=== GGUF Tensors ===")
print(f"Total tensors: {len(reader.tensors)}")
print(f"\nFirst 50 tensor names:")
for i, t in enumerate(reader.tensors[:50]):
    print(f"  {i}: {t.name}")

# Check for specific patterns
tensor_names = [t.name for t in reader.tensors]
t5_keys = [k for k in tensor_names if 'encoder.block' in k or 'shared.weight' in k]
clip_keys = [k for k in tensor_names if 'text_model' in k or 'logit_scale' in k]
flux_keys = [k for k in tensor_names if 'double_blocks' in k or 'single_blocks' in k]

print(f"\n=== Detected Patterns ===")
print(f"T5-like keys: {len(t5_keys)}")
print(f"CLIP-like keys: {len(clip_keys)}")
print(f"Flux transformer keys: {len(flux_keys)}")

if t5_keys:
    print(f"\nSample T5 keys: {t5_keys[:5]}")
if clip_keys:
    print(f"\nSample CLIP keys: {clip_keys[:5]}")
if flux_keys:
    print(f"\nSample Flux keys: {flux_keys[:5]}")
