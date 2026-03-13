# UI Consistency Report
- generated_at: 2026-03-13T03:44:41.333Z
- source_root: apps/interface/src
- css_root: apps/interface/src/styles
- strict_mode: enabled

## Static Inline Styles (`style="..."`)
- none

## Dynamic `:style` / `v-bind:style` Bindings
- total: 9
- allowlisted: 7
- disallowed: 2
- [disallowed] apps/interface/src/components/Img2ImgInpaintParamsCard.vue:67 — expr=`{ '--img2img-mask-src': `url('${maskImageData}')` }` — `:style="{ '--img2img-mask-src': `url('${maskImageData}')` }"`
- [allowlisted] apps/interface/src/components/results/ResultsCard.vue:45 — expr=`props.bodyStyle` — `<div class="panel-body" :class="props.bodyClass" :style="props.bodyStyle">`
- [allowlisted] apps/interface/src/components/results/RunCard.vue:72 — expr=`batchMenuStyle` — `:style="batchMenuStyle"`
- [allowlisted] apps/interface/src/components/ui/ImageZoomOverlay.vue:30 — expr=`zoomStyle` — `<div class="image-zoom-canvas" :style="zoomStyle" @click.stop>`
- [disallowed] apps/interface/src/components/ui/ImageZoomOverlay.vue:41 — expr=`frameGuideRectStyle` — `:style="frameGuideRectStyle"`
- [allowlisted] apps/interface/src/components/ui/InpaintMaskEditorOverlay.vue:46 — expr=`contentTransformStyle` — `:style="contentTransformStyle"`
- [allowlisted] apps/interface/src/components/ui/InpaintMaskEditorOverlay.vue:66 — expr=`brushCursorStyle` — `:style="brushCursorStyle"`
- [allowlisted] apps/interface/src/components/ui/VideoZoomOverlay.vue:28 — expr=`zoomStyle` — `<div class="video-zoom-canvas" :style="zoomStyle" @click.stop>`
- [allowlisted] apps/interface/src/views/WANTab.vue:718 — expr=`guidedTooltipStyle` — `:style="guidedTooltipStyle"`

## Scoped `<style>` Blocks
- none

## Duplicated Selectors Across Files
- none

## Docs/Toolchain Drift
- no tracked reference checks were detected

## Backend Malformed-File Smoke Command Set (documented, not executed by this script)
- `cp apps/interface/tabs.json /tmp/tabs.json.bak && printf "{\"oops\":1}" > apps/interface/tabs.json`
- `cp apps/interface/workflows.json /tmp/workflows.json.bak && printf "{\"oops\":1}" > apps/interface/workflows.json`
- `cp apps/interface/presets.json /tmp/presets.json.bak && printf "{\"oops\":1}" > apps/interface/presets.json`
- `curl -i http://127.0.0.1:7850/api/ui/tabs` (expected fail-loud)
- `curl -i http://127.0.0.1:7850/api/ui/workflows` (expected fail-loud)
- `curl -i http://127.0.0.1:7850/api/ui/presets` (expected fail-loud)
- restore backups from `/tmp/*.bak` after smoke run
