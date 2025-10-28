from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ScriptArg(BaseModel):
    label: Optional[str] = Field(default=None, description="UI label for the argument")
    value: Optional[Any] = Field(default=None, description="Default value")
    minimum: Optional[Any] = Field(default=None, description="Minimum allowed value")
    maximum: Optional[Any] = Field(default=None, description="Maximum allowed value")
    step: Optional[Any] = Field(default=None, description="Step size for sliders")
    choices: Optional[list[str]] = Field(default=None, description="Available choices")


class ScriptInfo(BaseModel):
    name: Optional[str] = Field(default=None, description="Script identifier (lowercase)")
    is_alwayson: Optional[bool] = Field(default=None, description="Whether script always runs")
    is_img2img: Optional[bool] = Field(default=None, description="True if script applies to img2img")
    args: list[ScriptArg] = Field(default_factory=list, description="Argument metadata")


__all__ = ["ScriptArg", "ScriptInfo"]
