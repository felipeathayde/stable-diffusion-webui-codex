# UI Consistency Report
- generated_at: 2026-02-21T18:16:01.261Z
- source_root: apps/interface/src
- css_root: apps/interface/src/styles
- strict_mode: enabled

## Static Inline Styles (`style="..."`)
- none

## Dynamic `:style` / `v-bind:style` Bindings
- total: 7
- allowlisted: 7
- disallowed: 0
- [allowlisted] apps/interface/src/components/results/ResultsCard.vue:44 — expr=`props.bodyStyle` — `<div class="panel-body" :class="props.bodyClass" :style="props.bodyStyle">`
- [allowlisted] apps/interface/src/components/results/RunCard.vue:63 — expr=`batchMenuStyle` — `:style="batchMenuStyle"`
- [allowlisted] apps/interface/src/components/ui/ImageZoomOverlay.vue:31 — expr=`zoomStyle` — `:style="zoomStyle"`
- [allowlisted] apps/interface/src/components/ui/InpaintMaskEditorOverlay.vue:44 — expr=`contentTransformStyle` — `:style="contentTransformStyle"`
- [allowlisted] apps/interface/src/components/ui/InpaintMaskEditorOverlay.vue:64 — expr=`brushCursorStyle` — `:style="brushCursorStyle"`
- [allowlisted] apps/interface/src/views/ImageModelTab.vue:368 — expr=`previewStyle` — `:style="previewStyle"`
- [allowlisted] apps/interface/src/views/WANTab.vue:634 — expr=`guidedTooltipStyle` — `:style="guidedTooltipStyle"`

## Scoped `<style>` Blocks
- none

## Duplicated Selectors Across Files
- none

## Docs/Toolchain Drift
- [DRIFT] tailwind.config.ts reference — expected: apps/interface/tailwind.config.ts exists
- [DRIFT] .legacy/root-archive reference — expected: .legacy/root-archive exists
- [DRIFT] build:ts script reference — expected: package.json contains scripts.build:ts
- [DRIFT] watch:ts script reference — expected: package.json contains scripts.watch:ts

## Backend Malformed-File Smoke Command Set (documented, not executed by this script)
- `cp apps/interface/tabs.json /tmp/tabs.json.bak && printf "{\"oops\":1}" > apps/interface/tabs.json`
- `cp apps/interface/workflows.json /tmp/workflows.json.bak && printf "{\"oops\":1}" > apps/interface/workflows.json`
- `cp apps/interface/presets.json /tmp/presets.json.bak && printf "{\"oops\":1}" > apps/interface/presets.json`
- `curl -i http://127.0.0.1:7850/api/ui/tabs` (expected fail-loud)
- `curl -i http://127.0.0.1:7850/api/ui/workflows` (expected fail-loud)
- `curl -i http://127.0.0.1:7850/api/ui/presets` (expected fail-loud)
- restore backups from `/tmp/*.bak` after smoke run
