from __future__ import annotations


class ModelParserError(RuntimeError):
    """Raised when the Codex model parser fails to interpret a checkpoint."""

    def __init__(self, message: str, *, component: str | None = None):
        ctx = f" [component={component}]" if component else ""
        super().__init__(f"{message}{ctx}")
        self.component = component


class MissingComponentError(ModelParserError):
    def __init__(self, component: str, detail: str | None = None):
        msg = f"Required component '{component}' is missing"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg, component=component)


class ValidationError(ModelParserError):
    pass


class UnsupportedFamilyError(ModelParserError):
    def __init__(self, family: str):
        super().__init__(f"No parser plan for model family '{family}'", component="planner")
        self.family = family
