
# Codex Workspace Code Style Guide
Date: 2025-10-31
Owner: Backend Runtime Maintainers
Status: Active

## Purpose
- Document the coding guidelines that govern all refactors and net-new implementations in this workspace.
- Capture expectations for architecture, documentation, error handling, logging, and dependency hygiene.
- Provide actionable examples so contributors can align with Codex standards quickly.

## Core Principles
1. **Codex-first design**  
   - Rebuild modules with clear dataclasses/enums, no opaque globals, and explicit validation.  
   - Avoid “drop-in” ports. Understand the behaviour and reauthor in Codex style.
2. **Modularity & separation**  
   - Keep architecture-specific code in dedicated packages (`architectures/`, `preprocessors/`, etc.).  
   - Hold shared utilities in base modules; only import across packages via explicit registries/factories.
3. **Explicit errors**  
   - No silent fallbacks. Raise `NotImplementedError` or well-typed exceptions describing cause + remediation.  
   - Never add catch-all try/except that masks the real failure. Error messages must include actionable hints.
4. **Telemetry & validation**  
   - Log at `DEBUG` for shapes/device/dtype and at `WARNING`+ for user-impacting issues.  
   - Validate tensors (shape, dtype, range) at entry points before heavy compute.  
   - Do not print; use structured logging.
5. **Dependency hygiene**  
   - Ban references to legacy repo modules (`modules.*`, `devices`, etc.).  
   - Encapsulate optional dependencies with guarded imports and clear errors.
6. **Zero tolerance for shims / copy-paste**  
   - Do not introduce shims or adapters to mask legacy behaviour; fix the root cause instead of patching symptoms.  
   - Never copy code from `.refs/**` or external sources verbatim; study the behaviour (baseline: Hugging Face Diffusers) and reimplement in Codex style.  
   - Exception: mathematically delicate kernels (e.g., GGUF quantization) where bit-exact reproduction is required—copy only with license headers and document the justification.

## Directory Layout
- `apps/backend/patchers/controlnet/architectures/` – architecture-specific modules (SD, SDXL, Flux, Chroma). Include `AGENTS.md` per subfolder.
- `apps/backend/runtime/controlnet/preprocessors/` – preprocessors grouped by domain (`edges.py`, `depth.py`). Models live under `models/<name>.py`.
- `.sangoi/backend/runtime/controlnet-parity.md` – authoritative status tracker; update per feature.

## Patterns & Examples
### Dataclasses & Enums
```python
from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class ZoeDepthConfig:
    weights_path: str = "zoedepth/ZoeD_M12_N.pt"
    device: torch.device | None = None
    dtype: torch.dtype = torch.float32

class ControlWeightMode(Enum):
    STATIC = "static"
    SCHEDULED = "scheduled"
```
- Prefer enums over raw strings for mode selection; export via `__all__` when reused.

### Lazy Loaders
```python
@lru_cache(maxsize=1)
def load_zoe_model(config: ZoeDepthConfig) -> nn.Module:
    if ZoeDepth is None:
        raise RuntimeError("zoedepth package is required")
    conf = get_config("zoedepth", "infer")
    model = ZoeDepth.build_from_config(conf)
    state = safe_torch_load(config.weights_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.to(device=config.device or torch.device("cpu"), dtype=config.dtype)
    model.eval()
    return model
```

### Preprocessor Contract
```python
def preprocess_leres(image: torch.Tensor, **kwargs) -> PreprocessorResult:
    image = _ensure_image_batch(image)
    config = LeReSConfig(**kwargs)
    model = load_leres_model(config)
    outputs = []
    with torch.no_grad():
        for sample in image:
            depth = model(sample.unsqueeze(0))
            depth = depth - depth.amin(dim=(-2, -1), keepdim=True)
            depth = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            outputs.append(depth)
    depth_map = torch.cat(outputs, dim=0)
    return PreprocessorResult(
        image=depth_map,
        metadata={"preprocessor": "depth_leres", "weights_path": config.weights_path},
    )
```
- Always clamp/normalize results; return metadata with provenance.

### Logging
```python
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("Linked control chain: %s", [node.config.name for node in self.graph.nodes])
```
- Use module-level loggers; avoid `print`.

### Imports
- Only import from `apps.*` inside the repo.  
- Optional heavy deps (`transformers`, `zoedepth`) must be wrapped in try/except ImportError with explicit error messages.

## Documentation Standard
- Each new directory must include `AGENTS.md` describing purpose/owner/status.  
- Update parity doc, changelog, and task log for every change.

## Checklist Before Landing
1. Run `ast.parse` on touched files.  
2. Update AGENTS/parity/changelog/task log.  
3. Verify optional deps are guarded.  
4. Confirm no legacy imports.  
5. Ensure logging/exception messages are actionable.

_This guide is the reference for future Codex refactors—keep it current._
