# AGENT — apps/backend/patchers/controlnet/architectures/sdxl
Date: 2025-10-31  
Owner: Backend Runtime Maintainers  
Last Review: 2025-10-31  
Status: Draft

## Purpose
- Placeholder aliases for SDXL ControlNet until SDXL-specific behaviour diverges from the SD implementation.

## Notes
- Re-exports the SD modules today (`ControlNet`, `ControlLora`, `ControlNetLite`, `T2IAdapter`).
- Update this package when SDXL requires dedicated logic (e.g., different hint scaling or adapter layouts).
