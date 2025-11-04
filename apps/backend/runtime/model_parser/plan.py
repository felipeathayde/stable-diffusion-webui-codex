from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict

from apps.backend.runtime.models.state_dict import try_filter_state_dict
from apps.backend.runtime.trace import event as trace_event

from .errors import MissingComponentError
from .specs import ComponentState, ParserContext, ParserPlan


def _materialize_component(component: ComponentState) -> Dict[str, Any]:
    tensors = component.tensors
    materializer = getattr(tensors, "materialize", None)
    try:
        if callable(materializer):
            result = materializer()
            if not isinstance(result, dict):
                result = dict(result)
            trace_event(
                "parser_materialize",
                component=component.name,
                keys=len(result),
                strategy="lazy_materialize",
            )
            return result
        result = dict(tensors.items())
        trace_event(
            "parser_materialize",
            component=component.name,
            keys=len(result),
            strategy="dict_copy",
        )
        return result
    except Exception as exc:
        raise RuntimeError(f"Failed to materialize component '{component.name}'") from exc


def execute_plan(plan: ParserPlan, state_dict: MutableMapping[str, Any], *, signature) -> ParserContext:
    context = ParserContext(root_state=state_dict, signature=signature, plan=plan)

    # Split components first so converters can assume presence.
    for split in plan.splits:
        view = try_filter_state_dict(state_dict, split.prefixes, new_prefix=split.strip_prefix or "")
        length = len(view)
        if length == 0:
            if split.required:
                raise MissingComponentError(split.name, detail=f"prefixes {tuple(split.prefixes)} not found")
            continue
        trace_event("parser_split", component=split.name, count=length)
        try:
            # Probe a sample tensor to report dtype/device for this component lazily
            sample_key = None
            for k in view:
                sample_key = k
                break
            dtype = None
            device = None
            if sample_key is not None:
                try:
                    t = view[sample_key]
                    dtype = getattr(getattr(t, 'dtype', None), 'name', None)
                    device = getattr(getattr(t, 'device', None), 'type', None)
                except Exception:
                    pass
        except Exception:
            pass
        # Do NOT materialize the whole component here; keep the filtered mapping lazy.
        context.components[split.name] = ComponentState(name=split.name, tensors=view)

    # Apply converters sequentially.
    for converter in plan.converters:
        component = context.components.get(converter.component)
        if component is None:
            # Optional components may omit converters.
            continue
        trace_event("parser_convert_start", component=converter.component, function=converter.function.__name__)
        materialized = _materialize_component(component)
        updated = converter.function(materialized, context)
        if not isinstance(updated, dict):
            raise TypeError(f"Converter {converter.function.__name__} must return dict, got {type(updated)!r}")
        component.tensors = updated
        trace_event("parser_convert_done", component=converter.component, keys=len(component.tensors))

    # Run validations
    for validation in plan.validations:
        trace_event("parser_validate", name=validation.name)
        validation.function(context)

    return context
