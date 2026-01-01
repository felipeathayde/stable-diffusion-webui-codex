# apps/backend/interfaces Overview
<!-- tags: backend, api, validation -->
Date: 2025-12-05
Owner: Backend API Maintainers
Last Review: 2025-12-30
Status: Active

## Purpose
- Defines API-facing schemas and adapters that expose backend capabilities to the Codex frontend and external clients.

## Subdirectories
- `api/` — FastAPI endpoint implementations and adapters.
- `schemas/` — Pydantic/dataclass schemas describing request/response payloads.

## Notes
- Keep schemas in sync with the frontend API client (`apps/interface/src/api`).
- Avoid embedding business logic here—delegate to services/use cases and focus on validation and serialization.
- API workers should reuse a single `InferenceOrchestrator` instance per process to preserve engine caches/VRAM across requests. See `api/run_api.py` (`_ORCH` singleton).
- 2025-11-14: `/api/txt2img` enforces the semantic contract (e.g., `prompt`, `negative_prompt`, `width`, `extras.highres`) but still tolerates compatibility keys (`codex_engine`, `codex_diffusion_device`, `sd_model_checkpoint`) while downstream clients migrate; prompts may be empty to support negative-only runs.
- 2025-11-21: SPA static mount now registers after all `/api/*` routes to prevent POSTs from being intercepted by the UI fallback; invalid txt2/img2/video payloads raise HTTP errors instead of returning 200 with a background error.
- 2025-11-21: Module-level `app` remains available for ASGI servers, but the preferred entrypoint is the uvicorn factory `apps.backend.interfaces.api.run_api:create_api_app`. Factory and direct `:app` both build the same FastAPI instance.
- 2025-11-14: `create_api_app(argv, env)` is the canonical FastAPI factory; when launching uvicorn manually use `--factory apps.backend.interfaces.api.run_api:create_api_app` so the runtime bootstraps before serving (the TUI/launcher already calls it).
- 2025-12-03: `/api/txt2img` extras now accept `highres.refiner` (enable/steps/cfg/seed/model/vae) alongside the global `extras.refiner`, raising HTTP 400 on malformed nested refiner configs.
- 2025-12-03: `/api/tasks/{task_id}/cancel` allows best-effort cancellation (immediate vs after_current flag); workers abort event streaming with `error: cancelled` when `mode=immediate`.
- 2025-12-03: `/api/options` now accepts `codex_{core,te,vae}_{device,dtype}` to set per-role backend/dtype via memory manager; device choices auto/cuda/cpu/mps/xpu/directml, dtype auto/fp16/bf16/fp32.
- 2025-12-05: `/api/txt2img` extras agora aceitam um objeto opcional `text_encoder_override` (family + label + components[]) validado como JSON; quando presente, o worker de txt2img o encaminha como `engine_options.text_encoder_override` para o orchestrator/engines, que por sua vez repassam o override ao `runtime.models.loader` (via `TextEncoderOverrideConfig`). A partir de 2025-12-06, `components[]` também aceita entradas no formato `alias=/abs/path/to/weights.safetensors` para overrides por arquivo (ex.: Flux), que o loader interpreta como `explicit_paths` sem depender de labels de `/api/text-encoders`.
- 2025-12-05: `/api/options` e `/api/txt2img` expõem flags `codex_smart_offload`/`codex_smart_fallback`/`codex_smart_cache` (checkboxes na UI → `smart_offload`/`smart_fallback`/`smart_cache` no payload) para controlar descarregamento entre estágios, fallback para CPU em caso de OOM e caches de condicionamento SDXL; quando um job inclui `smart_cache`, o valor por-job prevalece sobre o snapshot global, permitindo rodar jobs mistos em uma mesma sessão.
- 2025-12-05: `/api/engines/capabilities` passa a incluir um bloco opcional `smart_cache` com contadores de hits/misses agregados (por bucket) para diagnóstico de caching de SDXL no runtime.
- 2025-12-05: `/api/memory` agora usa `apps.backend.runtime.memory.memory_management.memory_snapshot()` para expor um snapshot estruturado de VRAM/CPU (backend, dispositivo primário, probe, budgets, stats do torch e modelos carregados); clientes que só leem `total_vram_mb` continuam atendidos, mas novas UIs devem consumir o snapshot completo.
- 2025-12-06: `_bootstrap_runtime` agora pré-calcula o inventário de modelos (`apps.backend.inventory.cache.refresh()`) durante o bootstrap do backend, de forma que `/api/models/inventory` esteja quente quando a UI abrir o QuickSettings; a rota continua expondo `?refresh=true` e `POST /api/models/inventory/refresh` para rescans explícitos.
- 2025-12-06: `settings_schema.json` e `settings_registry.py` incluem chaves `codex_flux_core_streaming_*` (enabled/policy/blocks_per_segment/window_size/auto_threshold_mb) sob a seção SDXL, pensadas para controlar streaming do core Flux via `/api/options`. Por enquanto, apenas o backend consome esses valores convertendo-os em `engine_options` para o engine Flux; a UI pode optar por expô-los como controles avançados em uma fase posterior.
- 2025-12-14: `/api/txt2vid` e `/api/img2vid` populam `steps` em `Txt2VidRequest/Img2VidRequest` e o plano de vídeo (`build_video_plan`) lê `guidance_scale` (alinhamento de contrato com o runtime).
- 2025-12-16: Added `/api/vid2vid` (multipart: `video` upload + JSON `payload`) and `/api/output/{rel_path}` for root-scoped serving of exported videos. Path-based vid2vid inputs are allowed but restricted to the backend working directory to avoid permission surprises; upload is recommended.
- 2025-12-16: `/api/vid2vid` now supports `vid2vid_method="wan_animate"` with extra multipart inputs (`reference_image`, preprocessed `pose_video`/`face_video`, and optional `background_video`/`mask_video` for replacement mode).
- 2025-12-19: `/api/tools/convert-gguf` expanded quantization menu and now accepts `tensor_type_overrides` (regex → quant per tensor) for mixed schemes and advanced tuning.
- 2025-12-29: `/api/paths` now resolves `apps/paths.json` via `CODEX_ROOT` (required), keeping QuickSettings path-based filtering stable across launchers and CWD changes.
- 2025-12-29: `run_api.py` no longer uses `os.getcwd()` for repo files (settings/blocks/tabs/workflows/presets/tmp); it uses the resolved project root so the backend behaves the same no matter where it’s launched from.
- 2025-12-29: `/api/models/inventory` and `/api/text-encoders` now return repo-relative `path` values (when under `CODEX_ROOT`) to avoid leaking absolute host paths to the UI; WAN video APIs resolve repo-relative `wan_*` paths back to absolute under `CODEX_ROOT` so runtime never depends on CWD.
- 2025-12-29: API port-guard (`port_free`) now checks IPv4 + IPv6 loopback/wildcard (0.0.0.0/127.0.0.1/::/::1) to avoid “localhost” split-brain where an IPv6-only listener exists but the guard only tested IPv4.
- 2025-12-30: Suppressed uvicorn access-log spam for `/api/tools/convert-gguf/{job_id}` polling; opt out via `CODEX_UVICORN_ACCESS_LOG_TOOLS=1`.
- 2025-12-31: `/api/img2img` now accepts `img2img_extras` (incl. `text_encoder_override` + `tenc_sha`), enforces `tenc_sha` for `.gguf`, and forwards request-level `engine_options` into the orchestrator (parity with `/api/txt2img`, needed for Flux/Kontext GGUF runs).
- 2025-12-31: `/api/img2img` now infers missing `img2img_width/img2img_height` from the init image (snapped to multiples of 8) and provides Kontext defaults when `img2img_steps/img2img_cfg_scale/img2img_distilled_cfg_scale` are omitted (`28/1.0/2.5`).
