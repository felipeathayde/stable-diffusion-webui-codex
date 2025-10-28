"""Legacy txt2img wrapper removed in Codex backend."""

raise RuntimeError(
    "modules.txt2img has been removed. Call apps.backend.use_cases.txt2img.generate_txt2img"
    " with CodexProcessingTxt2Img via the backend services."
)
