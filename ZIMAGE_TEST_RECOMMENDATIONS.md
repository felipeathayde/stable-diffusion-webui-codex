# Z Image Debugging - Test Recommendations

## Finding Summary

### ✅ FIXES APPLIED
1. **Honor HF transformer config** (`apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/transformer/config.json`)
   - `rope_theta = 256.0`
   - `axes_dims = (32, 48, 48)` (RoPE axis split)
   - `t_scale = 1000.0` (timestep scaling)
2. **Honor HF scheduler config** (`apps/backend/huggingface/Alibaba-TongYi/Z-Image-Turbo/scheduler/scheduler_config.json`)
   - `shift = 3.0` (default)
3. **Fix VAE latent normalization**
   - Apply `process_out()` before VAE decode for Flow16 (Flux/Z-Image) latents, otherwise decodes can collapse to gray/noise.
4. **Flow16 VAE config parity (HF Flux/Z-Image)**
   - `use_quant_conv = false`, `use_post_quant_conv = false` (avoids misleading missing-key warnings like `quant_conv.*` with official VAEs)

### 📊 RESULTS FROM ZIMAGE_8

**Positive Changes:**
- norm(x) now DECREASES: 478→360→366→456 (before: was increasing 480→736)
- Model output is reasonable: norm=620-740
- No more divergence in early steps

**Remaining Issue:**
- If output still looks noisy, inspect the sigma ladder near the tail (large last-step drops can destabilize very low-step runs).
  - Diffusers `ZImagePipeline` recommends `num_inference_steps=9` for Turbo (≈8 effective updates; last `dt=0`).
  - Quick sanity: `python tools/diagnostics/inspect_flow_sigma_schedule.py --steps 9 --scheduler simple --mu 3 --pseudo-timestep-range 1000` should end with `..., 0.5, 0.3, 0, 0`.
  - If your log shows `norm(eps)=0` on the very last step (`sigma=0 -> 0`), that’s expected (dt=0). What matters is whether `norm(x)` stabilizes instead of blowing up near the tail.

### 🧪 RECOMMENDED TESTS

**Test 1: More Steps (Recommended)**
- Use 20 steps instead of 9
- This will create smoother sigma transitions
- Should eliminate the large jump problem

**Test 2: Adjust Shift (Alternative)**
- Try a smaller `shift` only if you have evidence the default (3.0) is wrong for your specific checkpoint.
- Default for Z Image Turbo is `shift=3.0` per HF scheduler config.

**Test 3: Different Scheduler (If above fail)**
- Try "normal" scheduler instead of "simple"
- May provide better sigma spacing

### 📝 HOW TO TEST

**Option A: Via UI**
1. Keep same prompt and settings
2. Change steps from 9 to 20
3. Generate zimage_9

**Option B: Test Script**
Run from root directory:
```powershell
# Test with 20 steps
python -c "
import sys
sys.path.insert(0, '.')
# Your test code here with steps=20
"
```

**Option C: Sigma Ladder Sanity (no models required)**
```powershell
python tools/diagnostics/inspect_flow_sigma_schedule.py --steps 9 --scheduler simple --mu 3 --pseudo-timestep-range 1000
```

### 🎯 EXPECTED OUTCOME

With 20 steps, the sigma schedule will be:
```
sigma: 1.0 → 0.95 → 0.90 → ... → 0.05 → 0.0
```

This should:
1. Maintain smooth norm(x) decrease throughout
2. Avoid the large jump at the end
3. Produce cleaner final image

### 📌 NEXT STEPS

1. **Generate zimage_9 with 20 steps**
2. Check if norm(x) decreases smoothly to ~100-200
3. Verify image has structure (not pure noise)

If zimage_9 still shows noise, the issue may be elsewhere (but logs suggest it should work now).
