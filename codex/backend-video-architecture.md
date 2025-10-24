# Backend Video Architecture — Per‑Task Runtimes and Generic GGUF (WAN22)

Date: 2025-10-24

Overview
- Engines implement tasks by file under `apps/server/backend/engines/diffusion/`:
  - `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`.
- Model engines (e.g., WAN 2.2) are thin and delegate to those per‑task modules.
- GGUF path is model‑agnostic: use `apps.server.backend.gguf` + `runtime/ops` + PyTorch SDPA.

WAN22 specifics
- Engines: `wan22_14b`, `wan22_5b` (diffusion directory). Semantic key ‘wan22’ maps to them.
- Common dataclasses in `engines/diffusion/wan22_common.py`:
  - `EngineOpts`, `WanComponents`, `WanStageOptions`.
- Generic GGUF runtime in `runtime/nn/wan22.py`:
  - `derive_spec_from_state`, `WanUNetGGUF.forward` (SA/CA/FFN via SDPA, modulação de tempo 6×C), skeletons `run_txt2vid`/`run_img2vid`.
- Per‑task runtimes:
  - `txt2vid.py`, `img2vid.py` — Diffusers‑first, delegam para GGUF quando não há pipeline Diffusers; VFI/export hooks.
- Schedulers:
  - `engines/util/schedulers.py` com `SamplerKind` e `apply_sampler_scheduler` (mapeia strings para Diffusers scheduler com avisos).

UI & Registration
- UI blocks/presets: `apps/interface/blocks.json` (+ `apps/interface/blocks.d`). Backend serve em `/api/ui/blocks`.
- Registration: `engines/registration.py`. `register_default_engines()` importa daqui.

Policy (Do/Don’t)
- Do: PyTorch‑first para atenção/ops; SDPA como padrão. Sem kernels custom quando existir operador equivalente no PyTorch.
- Do: Dataclasses para contratos (params/options/components) e enums para mapeamentos.
- Don’t: Novos `engines/video/wan/*` ou pacotes `wan_gguf*`.
- Don’t: Misturar lógica de múltiplas tarefas em um único arquivo com condicionais profundas.

Next Steps
- GGUF operational: patch embed/unembed 3D, encode/decodificação VAE por quadro, loop Euler(Simple) + CFG; seed do Low com último frame do High.
- Seleção explícita de variante (14B/5B) via preset/modelo no `run_api.py`.
- Remover legados (`engines/video/wan/*`, `wan_gguf*`) quando GGUF estiver funcional.

