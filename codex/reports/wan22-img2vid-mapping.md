WAN 2.2 Image-to-Video (Img2Vid) — Inputs Mapping
Date: 2025-10-22

Sources
- .refs/wan2-2workflow.json (ComfyUI workflow)
- .refs/ComfyUI-master (reference APIs/node names)

I2V channel composition (design)
- The UNet I/O for I2V expects an input with 36 channels at the patch embedding:
  - 16 latent (VAE) + 4 temporal mask + 16 image features.
- Runtime behavior (strict): when the loaded GGUF expects 36 and your VAE yields 16, the backend assembles this layout automatically for img2vid. If the expectation differs (e.g., not 16+4+16), the runtime raises an explicit error.

Primary nodes to mirror
- WanImageToVideo (type: "WanImageToVideo").
- VAELoader / CLIP loaders associated to WAN GGUF.
- KSampler(s) (seed and steps where applicable).
- RIFE VFI (optional interpolation).
- Video assemble/write (filename prefix, fps, resolution).

Proposed UI fields (Img2Vid tab)
- Initial image: file (reuses InitialImageCard). Required.
- Resolution: width, height.
- Duration/FPS:
  - fps (integer)
  - num_frames or seconds (choose one; map to workflow count)
- Seed: integer (or -1 random).
- Model variant (WAN): enum [HighNoise, LowNoise] — maps to GGUF path choice.
- Lora (Lightning): toggle + weight (0..1) — optional.
- CFG / denoise:
  - if workflow exposes, include CFG scale and steps; else hide.
- Output:
  - filename prefix
  - container/codec (mp4 default)
  - save workflow image (optional)
- Interpolation (RIFE): optional group
  - enabled (bool)
  - factor or target_fps

Backend payload (draft)
```
POST /img2vid
{
  "__strict_version": 1,
  "img2vid_init_image": "dataurl",
  "img2vid_width": 768,
  "img2vid_height": 768,
  "img2vid_fps": 24,
  "img2vid_frames": 48,
  "img2vid_seed": -1,
  "wan_variant": "HighNoise|LowNoise",
  "wan_lora": { "enabled": true, "weight": 0.6 },
  "video_filename_prefix": "wan22",
  "video_format": "mp4",
  "interpolation": { "enabled": false, "factor": 2 }
}
```

Notes
- GGUF model names in workflow: `Wan2.2-I2V-A14B-HighNoise-*.gguf` and `LowNoise`. Keep a server-side resolver for friendly labels.
- RIFE node present; expose only toggles required.
- Keep parity with existing task/event streaming.
