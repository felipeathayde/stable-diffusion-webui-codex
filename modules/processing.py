"""Legacy pipeline wrapper removed in Codex backend.

Importing this module is no longer supported. Use
`apps.backend.runtime.processing` dataclasses and helpers instead.
"""

raise RuntimeError(
    "modules.processing has been removed from Codex backend. "
    "Use apps.backend.runtime.processing.* for native pipelines."
)
