"""Unit tests for dilution/ledger/tools/_base.py.

This module is pure, deterministic schema codegen: plain frozen+slots
dataclasses (Tool / ToolArg) plus three functions that translate them to
provider-specific JSON-Schema tool objects. No DB, no network, no LLM at
runtime. The only I/O seam is a *lazy* ``from xai_sdk.chat import tool``
inside ``build_provider_schema`` for provider='xai', which we monkeypatch
for hermeticity.
"""

from __future__ import annotations

import pytest

from dilution.ledger.tools._base import (
    ISO_DATE_PATTERN,
    PROPOSED_ID_PATTERN,
    Tool,
    ToolArg,
    _arg_to_json_schema_property,
    _tool_to_parameters_schema,
    build_provider_schema,
)
from dilution.ledger.tools import ALL_TOOLS


# ── helpers ────────────────────────────────────────────────────────────

def make_arg(**kw) -> ToolArg:
    """Construct a ToolArg with sensible defaults, overriding via kwargs."""
    base = dict(name="p", type="string", required=False, description="d")
    base.update(kw)
    return ToolArg(**base)


def make_tool(args=(), **kw) -> Tool:
    """Construct a Tool with all 7 fields populated."""
    base = dict(
        name="my_tool",
        description="a tool",
        args=tuple(args),
        mutation_kind="record_event",
        instrument_type=None,
        event_kind=None,
        valid_forms=frozenset(),
    )
    base.update(kw)
    return Tool(**base)


# ── module-level constants ──────────────────────────────────────────────

class TestConstants:
    def test_iso_date_pattern(self):
        import re
        assert ISO_DATE_PATTERN == r"^\d{4}-\d{2}-\d{2}$"
        # behavioral: matches a valid ISO date, rejects a non-ISO string
        assert re.match(ISO_DATE_PATTERN, "2026-06-10")
        assert re.match(ISO_DATE_PATTERN, "6/10/2026") is None

    def test_proposed_id_pattern(self):
        import re
        assert PROPOSED_ID_PATTERN == r"^[a-z][a-z0-9-]{2,40}$"
        assert re.match(PROPOSED_ID_PATTERN, "atm-001")
        # must start with a lowercase letter
        assert re.match(PROPOSED_ID_PATTERN, "1abc") is None
        # too short (min 3 chars total: leading letter + 2 more)
        assert re.match(PROPOSED_ID_PATTERN, "ab") is None


# ── _arg_to_json_schema_property ────────────────────────────────────────

class TestArgToJsonSchemaProperty:

    # --- number -----------------------------------------------------------
    def test_number_no_min(self):
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=None))
        assert prop == {"description": "d", "type": "number"}
        assert "minimum" not in prop

    def test_number_min_zero_not_dropped(self):
        # zero is not None and must survive (regression to a falsy check
        # would silently drop it).
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=0.0))
        assert "minimum" in prop
        assert prop["minimum"] == 0.0

    def test_number_negative_min(self):
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=-1.5))
        assert prop["minimum"] == -1.5

    def test_number_positive_min_kept_as_float(self):
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=2.5))
        assert prop["minimum"] == 2.5
        assert isinstance(prop["minimum"], float)

    # --- integer ----------------------------------------------------------
    def test_integer_min_coerced_to_int(self):
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=3.0))
        assert prop["type"] == "integer"
        assert prop["minimum"] == 3
        assert isinstance(prop["minimum"], int)
        assert not isinstance(prop["minimum"], bool)

    def test_integer_min_truncates_not_rounds(self):
        # int(2.9) == 2 (truncation toward zero), NOT round to 3.
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=2.9))
        assert prop["minimum"] == 2

    def test_integer_min_truncates_negative(self):
        # int(-2.9) == -2 (truncation toward zero)
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=-2.9))
        assert prop["minimum"] == -2

    def test_integer_min_zero_not_dropped(self):
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=0.0))
        assert "minimum" in prop
        assert prop["minimum"] == 0

    def test_integer_min_bool_true_coerces_to_one(self):
        # bool is an int subclass; int(True) == 1. The integer branch's
        # int(...) cast normalizes it to a plain int 1 (not the bool True).
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=True))
        assert prop["minimum"] == 1
        assert prop["minimum"] is not True
        assert isinstance(prop["minimum"], int) and not isinstance(prop["minimum"], bool)

    def test_integer_no_min(self):
        prop = _arg_to_json_schema_property(make_arg(type="integer", min_value=None))
        assert prop == {"description": "d", "type": "integer"}

    # --- boolean ----------------------------------------------------------
    def test_boolean_basic(self):
        prop = _arg_to_json_schema_property(make_arg(type="boolean"))
        assert prop == {"description": "d", "type": "boolean"}

    def test_boolean_ignores_other_constraints(self):
        # boolean branch ignores min_value/enum/pattern/min_length entirely.
        prop = _arg_to_json_schema_property(make_arg(
            type="boolean", min_value=1.0, enum_values=("x",),
            pattern="^z$", min_length=4,
        ))
        assert prop == {"description": "d", "type": "boolean"}

    # --- string -----------------------------------------------------------
    def test_string_plain(self):
        prop = _arg_to_json_schema_property(make_arg(type="string"))
        assert prop == {"description": "d", "type": "string"}

    def test_string_enum_tuple_coerced_to_list(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="string", enum_values=("a", "b")))
        assert prop["enum"] == ["a", "b"]
        assert isinstance(prop["enum"], list)

    def test_string_min_length_zero_not_dropped(self):
        prop = _arg_to_json_schema_property(make_arg(type="string", min_length=0))
        assert "minLength" in prop
        assert prop["minLength"] == 0

    def test_string_min_length_positive(self):
        prop = _arg_to_json_schema_property(make_arg(type="string", min_length=5))
        assert prop["minLength"] == 5

    def test_string_pattern_copied_through(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="string", pattern=PROPOSED_ID_PATTERN))
        assert prop["pattern"] == PROPOSED_ID_PATTERN

    def test_string_all_constraints(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="string", enum_values=("a", "b"), min_length=2,
            pattern="^a"))
        assert prop == {
            "description": "d",
            "type": "string",
            "enum": ["a", "b"],
            "minLength": 2,
            "pattern": "^a",
        }

    def test_string_all_constraints_none(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="string", enum_values=None, min_length=None, pattern=None))
        assert prop == {"description": "d", "type": "string"}

    # --- date -------------------------------------------------------------
    def test_date_forces_iso_pattern(self):
        prop = _arg_to_json_schema_property(make_arg(type="date"))
        assert prop == {
            "description": "d",
            "type": "string",
            "pattern": ISO_DATE_PATTERN,
        }

    def test_date_overrides_user_pattern_and_min_length(self):
        # date type IGNORES a.pattern and a.min_length; always forces ISO.
        prop = _arg_to_json_schema_property(make_arg(
            type="date", pattern="BOGUS", min_length=99, enum_values=("x",)))
        assert prop["pattern"] == ISO_DATE_PATTERN
        assert "minLength" not in prop
        assert "enum" not in prop

    # --- array ------------------------------------------------------------
    def test_array_items_string_with_enum(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", items_enum=("x", "y")))
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "string", "enum": ["x", "y"]}

    def test_array_items_string_no_enum(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", items_enum=None))
        assert prop["items"] == {"type": "string"}

    def test_array_items_number_ignores_enum(self):
        # items_enum is only honored for string items; number items drop it.
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="number", items_enum=("x", "y")))
        assert prop["items"] == {"type": "number"}

    def test_array_items_integer(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="integer"))
        assert prop["items"] == {"type": "integer"}

    def test_array_items_type_none_defaults_to_string(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type=None))
        assert prop["items"] == {"type": "string"}

    def test_array_items_unknown_type_defaults_to_string(self):
        # any items_type not in {string, number, integer} -> defaults string.
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="boolean"))
        assert prop["items"] == {"type": "string"}

    def test_array_min_items_none(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", min_items=None))
        assert "minItems" not in prop

    def test_array_min_items_zero_not_dropped(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", min_items=0))
        assert "minItems" in prop
        assert prop["minItems"] == 0

    def test_array_min_items_positive(self):
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", min_items=2))
        assert prop["minItems"] == 2

    def test_array_min_items_with_default_string_items(self):
        # min_items present AND items_type omitted: minItems is set and items
        # still defaults to the string subschema (the two paths are independent).
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type=None, min_items=3))
        assert prop == {
            "description": "d",
            "type": "array",
            "minItems": 3,
            "items": {"type": "string"},
        }

    def test_array_string_items_ignore_min_length_and_pattern(self):
        # array branch only reads items_type/items_enum/min_items; a stray
        # min_length/pattern on the (array) arg must NOT leak into the schema.
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", min_length=9, pattern="^z$"))
        assert prop == {
            "description": "d",
            "type": "array",
            "items": {"type": "string"},
        }
        assert "minLength" not in prop and "pattern" not in prop

    def test_array_full_dict_min_items_and_enum_items(self):
        # full-dict equality combining minItems + enum'd string items +
        # description, to lock the exact array property shape.
        prop = _arg_to_json_schema_property(make_arg(
            type="array", items_type="string", items_enum=("x", "y"),
            min_items=1))
        assert prop == {
            "description": "d",
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": ["x", "y"]},
        }

    # --- number must NOT coerce to int (mirror of the integer branch) -----
    def test_number_whole_float_stays_float(self):
        # the number branch copies min_value verbatim; unlike integer it must
        # NOT call int(). A whole-number float stays a float (5.0, not 5).
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=5.0))
        assert prop["minimum"] == 5.0
        assert isinstance(prop["minimum"], float)
        assert not isinstance(prop["minimum"], int)

    def test_number_int_min_value_kept_verbatim_as_int(self):
        # the number branch does NOT cast — an int min_value passes through as
        # an int (5, not 5.0). Mirror of test_number_whole_float_stays_float:
        # together they prove "copy verbatim, no normalization" on the number
        # branch (in contrast to the integer branch which always int()'s).
        prop = _arg_to_json_schema_property(make_arg(type="number", min_value=5))
        assert prop["minimum"] == 5
        assert isinstance(prop["minimum"], int)

    # --- unknown type -----------------------------------------------------
    def test_unknown_type_raises_value_error(self):
        # ArgType is a typing.Literal (not enforced at runtime); frozen only
        # blocks reassignment, so constructing with an off-Literal value works.
        bad = make_arg(type="object")
        with pytest.raises(ValueError) as exc:
            _arg_to_json_schema_property(bad)
        assert "object" in str(exc.value)

    # --- description always present --------------------------------------
    @pytest.mark.parametrize("type_", [
        "number", "integer", "boolean", "string", "date", "array",
    ])
    def test_every_branch_includes_description(self, type_):
        # items_type given so the array branch builds a valid items subschema.
        prop = _arg_to_json_schema_property(make_arg(
            type=type_, description="hello", items_type="string"))
        assert prop["description"] == "hello"


# ── _tool_to_parameters_schema ──────────────────────────────────────────

class TestToolToParametersSchema:

    def test_empty_args_omits_required(self):
        schema = _tool_to_parameters_schema(make_tool(args=()))
        assert schema == {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        }
        # required omitted entirely (NOT == [])
        assert "required" not in schema

    def test_all_optional_omits_required(self):
        args = (
            make_arg(name="a", type="string", required=False),
            make_arg(name="b", type="number", required=False),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        assert "required" not in schema
        assert set(schema["properties"]) == {"a", "b"}

    def test_mix_required_only_lists_required_in_order(self):
        args = (
            make_arg(name="first", type="string", required=False),
            make_arg(name="second", type="string", required=True),
            make_arg(name="third", type="string", required=True),
            make_arg(name="fourth", type="string", required=False),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        # ordering follows declaration order; list equality, not set.
        assert schema["required"] == ["second", "third"]

    def test_required_preserves_args_order(self):
        args = (
            make_arg(name="zeta", type="string", required=True),
            make_arg(name="alpha", type="string", required=True),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        assert schema["required"] == ["zeta", "alpha"]

    def test_additional_properties_always_literal_false(self):
        schema = _tool_to_parameters_schema(make_tool(
            args=(make_arg(name="x", type="string", required=True),)))
        assert schema["additionalProperties"] is False

    def test_type_object_always_present(self):
        schema = _tool_to_parameters_schema(make_tool(args=()))
        assert schema["type"] == "object"

    def test_properties_delegate_to_arg_builder(self):
        arg = make_arg(name="amt", type="number", required=True, min_value=0.0)
        schema = _tool_to_parameters_schema(make_tool(args=(arg,)))
        assert schema["properties"]["amt"] == _arg_to_json_schema_property(arg)
        assert schema["properties"]["amt"]["minimum"] == 0.0

    def test_property_order_preserved(self):
        args = (
            make_arg(name="c", type="string"),
            make_arg(name="a", type="string"),
            make_arg(name="b", type="string"),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        assert list(schema["properties"].keys()) == ["c", "a", "b"]

    def test_duplicate_arg_names_last_wins_in_properties(self):
        # later arg overwrites earlier in the properties dict (dict semantics).
        args = (
            make_arg(name="dup", type="string"),
            make_arg(name="dup", type="number", min_value=1.0),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        # one key; value is the LATER (number) definition.
        assert list(schema["properties"].keys()) == ["dup"]
        assert schema["properties"]["dup"]["type"] == "number"
        assert schema["properties"]["dup"]["minimum"] == 1.0

    def test_duplicate_required_name_listed_twice(self):
        # current behavior: required is appended per arg, so a duplicated
        # required name appears twice (no dedup).
        args = (
            make_arg(name="dup", type="string", required=True),
            make_arg(name="dup", type="number", required=True),
        )
        schema = _tool_to_parameters_schema(make_tool(args=args))
        assert schema["required"] == ["dup", "dup"]

    def test_consecutive_calls_return_independent_properties(self):
        # no shared module-level mutable state: each call builds a fresh
        # properties dict, so mutating one schema cannot leak into another.
        t = make_tool(args=(make_arg(name="a", type="string"),))
        s1 = _tool_to_parameters_schema(t)
        s2 = _tool_to_parameters_schema(t)
        assert s1 is not s2
        assert s1["properties"] is not s2["properties"]
        s1["properties"]["a"]["mutated"] = True
        assert "mutated" not in s2["properties"]["a"]


# ── build_provider_schema ───────────────────────────────────────────────

class TestBuildProviderSchema:

    @pytest.mark.parametrize("provider", ["openai", "moonshot", "gemini"])
    def test_openai_compat_shape(self, provider):
        arg = make_arg(name="amt", type="number", required=True, min_value=0.0)
        t = make_tool(name="record_drawdown", description="record it",
                      args=(arg,))
        out = build_provider_schema(t, provider=provider)
        assert out == {
            "type": "function",
            "function": {
                "name": "record_drawdown",
                "description": "record it",
                "parameters": _tool_to_parameters_schema(t),
            },
        }

    def test_openai_inner_parameters_exact(self):
        arg = make_arg(name="x", type="string", required=True)
        t = make_tool(args=(arg,))
        out = build_provider_schema(t, provider="openai")
        assert out["function"]["parameters"] == _tool_to_parameters_schema(t)

    def test_name_description_passthrough_verbatim(self):
        # no escaping / trimming of name or description.
        t = make_tool(name="  spaced name  ",
                      description="line1\nline2 with <tags> & symbols")
        out = build_provider_schema(t, provider="openai")
        assert out["function"]["name"] == "  spaced name  "
        assert out["function"]["description"] == "line1\nline2 with <tags> & symbols"

    @pytest.mark.parametrize("provider", ["anthropic", "", "XAI", "OpenAI",
                                          "Xai", "openai "])
    def test_unknown_provider_raises_value_error(self, provider):
        t = make_tool(args=())
        with pytest.raises(ValueError) as exc:
            build_provider_schema(t, provider=provider)
        assert repr(provider) in str(exc.value)

    def test_provider_matching_is_case_sensitive(self):
        t = make_tool(args=())
        # exact-case 'openai' works...
        assert build_provider_schema(t, provider="openai")["type"] == "function"
        # ...but 'OpenAI' does not.
        with pytest.raises(ValueError):
            build_provider_schema(t, provider="OpenAI")

    def test_xai_calls_sdk_with_kwargs(self, monkeypatch):
        captured = {}

        def fake_tool(**kw):
            captured.update(kw)
            return ("XAI_TOOL_OBJ", kw)

        monkeypatch.setattr("xai_sdk.chat.tool", fake_tool)

        arg = make_arg(name="amt", type="number", required=True, min_value=0.0)
        t = make_tool(name="record_drawdown", description="desc", args=(arg,))
        out = build_provider_schema(t, provider="xai")

        # returns whatever the sdk returns verbatim (docstring claims a
        # protobuf, but the function literally returns the stub's value).
        assert out == ("XAI_TOOL_OBJ", captured)
        assert captured["name"] == "record_drawdown"
        assert captured["description"] == "desc"
        assert captured["parameters"] == _tool_to_parameters_schema(t)
        # called by keyword (name/description/parameters); no positional args.
        assert set(captured) == {"name", "description", "parameters"}

    def test_xai_and_openai_share_identical_parameters(self, monkeypatch):
        # cross-provider parity: the params handed to the xai SDK must be
        # structurally identical to the params embedded in the openai dict,
        # since both come from the single _tool_to_parameters_schema source.
        captured = {}
        monkeypatch.setattr(
            "xai_sdk.chat.tool",
            lambda **kw: captured.update(kw) or ("XAI", kw),
        )
        arg = make_arg(name="amt", type="number", required=True, min_value=0.0)
        t = make_tool(name="record_drawdown", description="desc", args=(arg,))

        build_provider_schema(t, provider="xai")
        openai_out = build_provider_schema(t, provider="openai")

        assert captured["parameters"] == openai_out["function"]["parameters"]

    def test_xai_import_is_lazy_non_xai_path_unaffected(self, monkeypatch):
        # Break the xai import entirely; the openai path must still work
        # because the import lives inside the xai branch only.
        import sys
        import builtins

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name.startswith("xai_sdk"):
                raise ImportError("xai_sdk is unavailable in this test")
            return real_import(name, *args, **kwargs)

        # ensure a fresh import attempt would go through our blocker
        monkeypatch.delitem(sys.modules, "xai_sdk", raising=False)
        monkeypatch.delitem(sys.modules, "xai_sdk.chat", raising=False)
        monkeypatch.setattr(builtins, "__import__", blocking_import)

        t = make_tool(args=(make_arg(name="x", type="string"),))
        # openai path: no xai import triggered.
        out = build_provider_schema(t, provider="openai")
        assert out["type"] == "function"

        # and the xai path indeed tries (and fails) to import -> proves lazy.
        with pytest.raises(ImportError):
            build_provider_schema(t, provider="xai")


# ── realistic round-trip over the real production tool registry ──────────
#
# The synthetic tests above exercise every code branch with constructed
# inputs. This class instead feeds the *actual* 26 production Tool objects
# (dilution.ledger.tools.ALL_TOOLS) through the codegen and asserts the
# JSON-Schema invariants that downstream providers depend on. This catches
# real-world arg shapes the synthetic tests never construct, and guards
# against a future tool definition that violates a schema contract.

class TestRealToolRegistry:

    def test_registry_is_nonempty(self):
        # guard: if the import silently yielded {} the per-tool tests below
        # would vacuously pass. Pin a real lower bound.
        assert len(ALL_TOOLS) >= 20
        assert all(isinstance(t, Tool) for t in ALL_TOOLS.values())

    @pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
    def test_openai_schema_wellformed_for_every_tool(self, tool_name):
        t = ALL_TOOLS[tool_name]
        out = build_provider_schema(t, provider="openai")
        assert out["type"] == "function"
        fn = out["function"]
        # name/description copied verbatim from the Tool dataclass.
        assert fn["name"] == t.name
        assert fn["description"] == t.description
        params = fn["parameters"]
        assert params["type"] == "object"
        # the docstring's central guarantee: nesting pathology is blocked.
        assert params["additionalProperties"] is False
        # inner parameters is exactly what the standalone builder produces.
        assert params == _tool_to_parameters_schema(t)

    @pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
    def test_required_matches_declaration_order_for_every_tool(self, tool_name):
        t = ALL_TOOLS[tool_name]
        params = _tool_to_parameters_schema(t)
        expected_required = [a.name for a in t.args if a.required]
        if expected_required:
            # exact list, in declaration order (not a set).
            assert params["required"] == expected_required
        else:
            # required omitted entirely when no arg is required.
            assert "required" not in params

    @pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
    def test_every_property_has_description_and_type(self, tool_name):
        t = ALL_TOOLS[tool_name]
        params = _tool_to_parameters_schema(t)
        # one property per declared arg, all keyed by arg name (no dups in
        # the real registry).
        arg_names = [a.name for a in t.args]
        assert list(params["properties"].keys()) == arg_names
        for pname, prop in params["properties"].items():
            assert "description" in prop and prop["description"], (tool_name, pname)
            assert "type" in prop, (tool_name, pname)
            if prop["type"] == "array":
                # array properties must carry an items subschema.
                assert "items" in prop and "type" in prop["items"]

    @pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
    def test_date_args_force_iso_pattern_for_every_tool(self, tool_name):
        t = ALL_TOOLS[tool_name]
        params = _tool_to_parameters_schema(t)
        for a in t.args:
            if a.type == "date":
                prop = params["properties"][a.name]
                assert prop["type"] == "string"
                assert prop["pattern"] == ISO_DATE_PATTERN

    @pytest.mark.parametrize("tool_name", sorted(ALL_TOOLS))
    def test_xai_and_openai_params_identical_for_every_tool(self, tool_name,
                                                            monkeypatch):
        # cross-provider parity on real tools: the params handed to the xai
        # SDK must equal the params embedded in the openai dict.
        captured = {}
        monkeypatch.setattr(
            "xai_sdk.chat.tool",
            lambda **kw: captured.update(kw) or ("XAI", kw),
        )
        t = ALL_TOOLS[tool_name]
        build_provider_schema(t, provider="xai")
        openai_out = build_provider_schema(t, provider="openai")
        assert captured["parameters"] == openai_out["function"]["parameters"]
        assert captured["name"] == t.name == openai_out["function"]["name"]
