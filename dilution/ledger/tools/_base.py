"""Primitives for declaring walker tools.

`Tool` and `ToolArg` are plain dataclasses — no Pydantic, no runtime
validation. The build_provider_schema() function produces the
provider-specific Tool/function object that the LLM sees.

What the JSON Schema enforces at decode time:
  - required: every arg with required=True must appear
  - type: number, integer, string, boolean, array
  - minLength / minimum: lower bounds for strings / numbers
  - pattern: regex (used for date format YYYY-MM-DD and id slug)
  - enum: closed vocabulary for string-typed args
  - additionalProperties=false: prevents the {"terms": {...}} nesting
    pathology by disallowing keys not in `properties`

What stays runtime-validated (per-tool builder in parse.py):
  - cross-field constraints (≥1 field set on amend; reverse-split
    post<pre; close with reason='superseded' requires replaced_by)
  - date parsing into datetime.date
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ArgType = Literal[
    "number", "integer", "string", "boolean",
    "date", "array",
]

ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
PROPOSED_ID_PATTERN = r"^[a-z][a-z0-9-]{2,40}$"


@dataclass(frozen=True, slots=True)
class ToolArg:
    name: str
    type: ArgType
    required: bool
    description: str
    # Constraints. None = unconstrained for that dimension.
    enum_values: tuple[str, ...] | None = None
    items_type: ArgType | None = None        # for type="array"
    items_enum: tuple[str, ...] | None = None
    min_length: int | None = None             # strings, arrays
    min_items: int | None = None
    min_value: float | None = None            # numbers, integers
    pattern: str | None = None                # strings
    default: Any = None


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    args: tuple[ToolArg, ...]
    mutation_kind: str                 # "create_instrument" | "amend_instrument" | "record_event" | "close_instrument" | "apply_split" | "note_no_event"
    instrument_type: str | None        # "atm" | "warrant" | ... | None
    event_kind: str | None             # "drawdown" | "exercise" | ... | None
    valid_forms: frozenset[str]        # SEC form symbols this tool may apply to


# ─── Provider schema codegen ──────────────────────────────────────────

def _arg_to_json_schema_property(a: ToolArg) -> dict:
    """Translate a ToolArg to a JSON Schema property object."""
    prop: dict[str, Any] = {"description": a.description}

    if a.type == "number":
        prop["type"] = "number"
        if a.min_value is not None:
            prop["minimum"] = a.min_value
    elif a.type == "integer":
        prop["type"] = "integer"
        if a.min_value is not None:
            prop["minimum"] = int(a.min_value)
    elif a.type == "boolean":
        prop["type"] = "boolean"
    elif a.type == "string":
        prop["type"] = "string"
        if a.enum_values is not None:
            prop["enum"] = list(a.enum_values)
        if a.min_length is not None:
            prop["minLength"] = a.min_length
        if a.pattern is not None:
            prop["pattern"] = a.pattern
    elif a.type == "date":
        # ISO YYYY-MM-DD strings. Pattern is the load-bearing constraint
        # the decoder honors at sample time; downstream parse converts
        # to datetime.date.
        prop["type"] = "string"
        prop["pattern"] = ISO_DATE_PATTERN
    elif a.type == "array":
        prop["type"] = "array"
        if a.min_items is not None:
            prop["minItems"] = a.min_items
        item_schema: dict[str, Any] = {}
        if a.items_type == "string":
            item_schema["type"] = "string"
            if a.items_enum is not None:
                item_schema["enum"] = list(a.items_enum)
        elif a.items_type == "number":
            item_schema["type"] = "number"
        elif a.items_type == "integer":
            item_schema["type"] = "integer"
        else:
            item_schema["type"] = "string"
        prop["items"] = item_schema
    else:
        raise ValueError(f"unknown ToolArg type {a.type!r}")
    return prop


def _tool_to_parameters_schema(t: Tool) -> dict:
    """Produce the JSON-Schema object that goes into the tool's
    `parameters` field. Same shape across xAI and OpenAI-compat."""
    properties: dict[str, dict] = {}
    required: list[str] = []
    for a in t.args:
        properties[a.name] = _arg_to_json_schema_property(a)
        if a.required:
            required.append(a.name)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def build_provider_schema(t: Tool, *, provider: str) -> Any:
    """Produce a provider-specific tool object suitable for passing as
    one element of the `tools=` list on the chat factory.

    - "xai":     returns an xai_sdk.chat.Tool protobuf
    - "openai" / "moonshot" / "gemini": returns a dict in the OpenAI
      function-calling shape ({"type": "function", "function": {...}})
    """
    params = _tool_to_parameters_schema(t)
    if provider == "xai":
        from xai_sdk.chat import tool as _xai_tool
        return _xai_tool(
            name=t.name,
            description=t.description,
            parameters=params,
        )
    if provider in ("openai", "moonshot", "gemini"):
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": params,
            },
        }
    raise ValueError(f"unknown provider {provider!r}")


__all__ = [
    "Tool", "ToolArg",
    "ISO_DATE_PATTERN", "PROPOSED_ID_PATTERN",
    "build_provider_schema",
]
