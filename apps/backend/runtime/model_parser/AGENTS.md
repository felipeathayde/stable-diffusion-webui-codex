# AGENT — Model Parser
<!-- tags: runtime, model-parser -->
Status: Active

## Mandate
- Parse checkpoint state dicts without `huggingface_guess`.
- Split, convert, and validate components using registry `ModelSignature` metadata.
- Produce structured `CodexEstimatedConfig` objects for loaders and adapters.

## Key files
- `__init__.py` — public parser entrypoint.
- `builders.py` — component registration and estimated-config assembly.
- `plan.py` — execution engine for declarative parser plans.
- `quantization.py` — quantization detection/validation helpers.
- `converters/` — shared component converters.
- `families/` — family-specific parser planners.

## Expectations
- Keep GGUF plans aligned with canonical keyspace resolvers in `apps/backend/runtime/state_dict/**`.
- `quantization.py` must detect GGUF/NF4/FP4 and fail loud on unsupported packed artifacts.
- When parser modules change, run `uv run python -m py_compile ...` for the touched parser files and record manual validation steps.
