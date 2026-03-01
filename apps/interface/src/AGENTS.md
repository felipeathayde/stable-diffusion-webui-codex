<!-- tags: frontend, interface-src, overview -->
# apps/interface/src Overview
Date: 2025-10-28
Last Review: 2026-02-28
Status: Active

## Purpose
- Houses the Vue 3 application source code (components, state stores, API client, global styles).

## Subdirectories
- `api/` — Typed client wrappers and DTOs used to call the backend.
- `components/` — Reusable UI components.
- `stores/` — Pinia stores handling shared state.
- `styles/` — Scoped CSS modules imported via Tailwind tokens.
- `views/` — Page-level views mapped in the router.

## Key Files
- `App.vue` — Root component.
- `main.ts` — Application entrypoint (mounts Vue, installs plugins).
- `router.ts` — Route definitions for major views.
- `styles.css` — Global style entry (Tailwind + tokens).

## Notes
- Follow the frontend guidelines in `.sangoi/frontend/guidelines/` when adding new modules.
- Keep API types and schemas synchronized with `.sangoi/backend/interfaces/`.
- 2026-02-28: Source area follows root testing policy: manual validation by default; automated/unit tests are not maintained unless explicitly requested by the repo owner.
- 2025-12-04: Legacy `/txt2vid` and `/img2vid` SPA routes were removed; WAN22 video workflows now enter exclusively via model tabs (`/models/:tabId` with `type === 'wan'`) and backend video endpoints remain available for those tabs only.
- 2025-12-17: Added a shared `stores/workflows.ts` and guided-gen UI primitives (`styles/components/guided-gen.css`) to support WAN guided generation and reactive snapshots under `/workflows`.
- 2025-12-23: Added shared slider primitives under `components/ui/` (SliderField + NumberStepperInput) with matching styles under `styles/components/`.
- 2025-12-29: `App.vue` derives `--sticky-offset` from the `.main-header` height (via `ResizeObserver`) so `RunCard` can stay sticky below the header.
- 2026-01-01: Image model tabs now include `clipSkip` in their per-tab params and send `clip_skip`/`img2img_clip_skip` to the backend (no prompt-tag injection needed).
- 2026-01-03: Added standardized file header blocks to `App.vue`, `main.ts`, and `router.ts` (doc-only change; part of rollout).
- 2026-02-28: WAN video tabs now use explicit stage LoRA arrays (`wan_high/wan_low.loras[]`, each entry `{sha, weight?}`), aligning with backend fail-loud stage validation.
- 2026-01-23: WAN video width/height now snap to multiples of 16 (rounded up; Diffusers parity) to avoid backend validation errors and patch-grid drift.
- 2026-02-16: WAN LightX2V I2V 14B now keeps stage `flowShift` in tab state and auto-manages `flowShift=5.0` from QuickSettings mode/toggle wiring for distill parity.
- 2026-02-06: Added `src/env.d.ts` and enabled Vue SFC typechecking via `vue-tsc` (ensures `.vue` imports are typechecked; prevents silent TS drift).
- 2026-02-06: Added hard-fatal bootstrap orchestration (`stores/bootstrap.ts`) with root App loader/fatal retry gating; App now blocks partial UI until required startup dependencies load.
- 2026-02-21: `App.vue` bootstrap/fatal screen styles were moved from local `<style scoped>` into shared stylesheet `styles/components/bootstrap-screen.css` (imported via `styles.css`).
- 2026-02-08: SDXL swap-model UX now uses explicit step-pointer semantics across src (`swapAtStep` in stores/UI, serialized as `switch_at_step` in API payloads) to avoid confusion with literal SDXL refiner “step count”.
- 2026-02-08: `ImageModelTab.vue` now composes `components/Img2ImgInpaintParamsCard.vue` for detailed init-image/inpaint controls; shared normalization moved to `utils/image_params.ts`.
- 2026-02-08: Hires controls now follow the Basic Parameters row organization, are hidden in img2img mode by policy, and txt2img hires prompt overrides now fallback to base prompts when blank.
- 2026-02-17: Top navigation removed `models`/`xyz`, added `/gallery` placeholder route/tab, and moved XYZ workflow into an embeddable card used inside image-tab Generation Parameters (with `/xyz` kept as compatibility wrapper).
- 2026-02-17: Footer branding now links to the repository/commit and `@lucas_sangoi` profile; Run progress UI is standardized via shared `components/results/RunProgressStatus.vue` across generation surfaces.
