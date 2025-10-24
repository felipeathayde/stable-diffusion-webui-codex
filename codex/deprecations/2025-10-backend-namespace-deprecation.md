# Deprecation: `backend.*` → `apps.server.backend.*`

Date: 2025-10-24

Scope
- All active code under this repository must import backend APIs via the façade `apps.server.backend.*`.
- Legacy files under `legacy/` may keep historical imports for reference only (read‑only).

Rationale
- We consolidated the backend into a single coherent package under `apps/server/backend` with a public façade (`apps.server.backend`).
- A dynamic import redirector is installed in `backend/__init__.py` to keep old imports working temporarily while we migrate downstream consumers.
- This yields clearer ownership, fewer circular imports, and a strictly versioned surface.

Timeline
- 2025-10-24: Deprecation notice published.
- 2025-10-24: Removal executed — legacy `backend/` and `backend_ext/` packages deleted; HF assets moved into `apps/server/backend/huggingface/`.

What to do (migration checklist)
- Replace `from backend import X` with `from apps.server.backend import X` when `X` is exported by the façade. Prefer façade‑level imports over deep subpackages.
- For modules not exported at façade root, import the intended subpackage under `apps.server.backend`, e.g. runtime ops and text processing.
- Run a quick scan locally: `rg -n "(^|\s)from\s+backend\b|(^|\s)import\s+backend\b" -S` and fix occurrences outside `legacy/`.
- Sanity‑check after edits: `python -m compileall apps/server/backend`.

Canonical mappings (prefix rules)
- The redirector maps `backend.*` to `apps.server.backend.*` with a few legacy group renames:
  - `backend.diffusion_engine` → `apps.server.backend.engines.diffusion`
  - `backend.text_processing` → `apps.server.backend.runtime.text_processing`
  - `backend.sampling` → `apps.server.backend.runtime.sampling`
  - `backend.misc` → `apps.server.backend.runtime.misc`
  - `backend.modules` → `apps.server.backend.runtime.modules`
  - `backend.nn` → `apps.server.backend.runtime.nn`
  - `backend.video.interpolation` → `apps.server.backend.video.interpolation`
  - Singles: `backend.operations(_bnb|_gguf)` → `apps.server.backend.runtime.ops.operations(_bnb|_gguf)`
  - Singles: `backend.{utils,stream,state_dict,loader,logging_utils,torch_trace,memory_management}` → their counterparts under `apps.server.backend.runtime.*`

Façade exports (preferred imports)
- Services: `ImageService`, `MediaService`, `OptionsService`, `ProgressService`, `SamplerService`.
- Engines: `register_default_engines`, errors (`EngineLoadError`, `EngineExecutionError`).
- Core: `InferenceOrchestrator`, `EngineRegistry`, request dataclasses (`Txt2ImgRequest`, `Img2ImgRequest`, `Txt2VidRequest`, `Img2VidRequest`, events).
- Runtime: `attention`, `logging`, `memory_management`, `models`, `nn`, `ops`, `shared`, `stream`, `text_processing`, `utils`.

Examples
```py
# BEFORE
from backend import utils, shared
from backend.operations import apply_model
from backend.text_processing import ClassicTextProcessingEngine

# AFTER (façade‑first)
from apps.server.backend import utils, shared
from apps.server.backend.ops import operations as ops
from apps.server.backend import text_processing

ops.apply_model(...)
engine = text_processing.ClassicTextProcessingEngine(...)
```

Observability
- No redirector remains. If you still import `backend.*`, Python will raise `ModuleNotFoundError`. Migrate to the façade imports shown above.

References
- Shim inventory: `apps/server/backend/SHIM_INVENTORY.md` (authoritative list of remaining shims and removals).
- Handoff context: `.sangoi/handoffs/2025-10-24-backend-cleanup-shims-gguf-handoff.md`.

Notes
- Do not modify files under `legacy/`.
- External extensions should migrate to the façade within the window above. If a missing façade export is blocking your migration, open an issue or PR to add a stable export rather than importing deep internals.
