"""Legacy img2img wrapper removed in Codex backend."""

raise RuntimeError(
    "modules.img2img has been removed. Use apps.backend.use_cases.img2img.run_img2img"
    " with CodexProcessingImg2Img via the backend services."
)
