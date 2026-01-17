# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for FormResources resolution methods and parse_assignment edge cases.

Covers:
- FormResources.resolve_gen_model (lines 60-74)
- FormResources.resolve_parse_model (lines 81-87)
- FormResources.resolve_tools (lines 103-114)
- parse_assignment edge cases (line 192)
- _parse_resources edge cases (line 225)
"""

from dataclasses import dataclass
from uuid import uuid4

import pytest

from lionpride.work.capabilities import (
    AmbiguousResourceError,
    CapabilityError,
    FormResources,
    parse_assignment,
)


@dataclass
class MockBranch:
    """Mock Branch for testing resource resolution."""

    resources: set
    name: str | None = None
    id: str | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())


class TestFormResourcesResolveGenModel:
    """Tests for FormResources.resolve_gen_model."""

    def test_model_specified_and_in_branch_resources(self):
        """Test successful resolution when model is in branch resources."""
        branch = MockBranch(resources={"gpt4", "gpt5"}, name="test")
        res = FormResources(api_gen="gpt4")
        assert res.resolve_gen_model(branch) == "gpt4"

    def test_api_fallback_when_api_gen_not_specified(self):
        """Test api fallback when api_gen not specified."""
        branch = MockBranch(resources={"gpt4", "gpt5"}, name="test")
        res = FormResources(api="gpt5")
        assert res.resolve_gen_model(branch) == "gpt5"

    def test_model_not_in_branch_resources_raises_capability_error(self):
        """Test error when model not in branch resources (lines 63-64)."""
        branch = MockBranch(resources={"gpt4"}, name="test-branch")
        res = FormResources(api_gen="gpt5")  # gpt5 not in resources

        with pytest.raises(CapabilityError, match="Model 'gpt5' not available"):
            res.resolve_gen_model(branch)

    def test_model_not_in_branch_uses_branch_name_in_error(self):
        """Test error message includes branch name."""
        branch = MockBranch(resources={"gpt4"}, name="my-branch")
        res = FormResources(api="nonexistent")

        with pytest.raises(CapabilityError, match="my-branch"):
            res.resolve_gen_model(branch)

    def test_model_not_in_branch_uses_id_prefix_when_no_name(self):
        """Test error message uses id[:8] when branch has no name."""
        branch = MockBranch(resources={"gpt4"}, name=None)
        res = FormResources(api_gen="missing")

        with pytest.raises(CapabilityError, match="not available in branch"):
            res.resolve_gen_model(branch)

    def test_auto_resolve_single_resource(self):
        """Test auto-resolution when branch has exactly one resource (line 68-69)."""
        branch = MockBranch(resources={"gpt4-only"}, name="test")
        res = FormResources()  # No api or api_gen specified

        assert res.resolve_gen_model(branch) == "gpt4-only"

    def test_zero_resources_raises_capability_error(self):
        """Test error when branch has zero resources (lines 70-71)."""
        branch = MockBranch(resources=set(), name="empty-branch")
        res = FormResources()  # No api or api_gen specified

        with pytest.raises(CapabilityError, match="No resources available"):
            res.resolve_gen_model(branch)

    def test_zero_resources_uses_branch_id_when_no_name(self):
        """Test zero resources error uses id[:8] when no name."""
        branch = MockBranch(resources=set(), name=None)
        res = FormResources()

        with pytest.raises(CapabilityError, match="No resources available"):
            res.resolve_gen_model(branch)

    def test_multiple_resources_without_api_raises_ambiguous_error(self):
        """Test ambiguous error when multiple resources and no api specified (lines 72-74)."""
        branch = MockBranch(resources={"gpt4", "gpt5", "claude"}, name="multi-branch")
        res = FormResources()  # No api or api_gen specified

        with pytest.raises(AmbiguousResourceError, match="Multiple resources"):
            res.resolve_gen_model(branch)

    def test_ambiguous_error_includes_branch_name(self):
        """Test ambiguous error message includes branch name."""
        branch = MockBranch(resources={"a", "b"}, name="ambiguous-branch")
        res = FormResources()

        with pytest.raises(AmbiguousResourceError, match="ambiguous-branch"):
            res.resolve_gen_model(branch)


class TestFormResourcesResolveParseModel:
    """Tests for FormResources.resolve_parse_model."""

    def test_api_parse_specified_and_in_branch(self):
        """Test successful resolution when api_parse is specified."""
        branch = MockBranch(resources={"gpt4-mini"}, name="test")
        res = FormResources(api_parse="gpt4-mini")
        assert res.resolve_parse_model(branch) == "gpt4-mini"

    def test_api_gen_fallback_when_api_parse_not_specified(self):
        """Test api_gen fallback when api_parse not specified."""
        branch = MockBranch(resources={"gpt5"}, name="test")
        res = FormResources(api_gen="gpt5")
        assert res.resolve_parse_model(branch) == "gpt5"

    def test_api_fallback_when_api_parse_and_api_gen_not_specified(self):
        """Test api fallback when api_parse and api_gen not specified."""
        branch = MockBranch(resources={"gpt4"}, name="test")
        res = FormResources(api="gpt4")
        assert res.resolve_parse_model(branch) == "gpt4"

    def test_model_not_in_branch_resources_raises_capability_error(self):
        """Test error when parse model not in branch resources (lines 84-85)."""
        branch = MockBranch(resources={"gpt4"}, name="test-branch")
        res = FormResources(api_parse="missing-model")

        with pytest.raises(CapabilityError, match="Model 'missing-model' not available"):
            res.resolve_parse_model(branch)

    def test_fallback_to_gen_model_resolution(self):
        """Test fallback to resolve_gen_model when no model specified (line 87)."""
        branch = MockBranch(resources={"single-model"}, name="test")
        res = FormResources()  # No api_parse, api_gen, or api

        # Should fallback to resolve_gen_model auto-resolution
        assert res.resolve_parse_model(branch) == "single-model"

    def test_fallback_to_gen_model_raises_if_ambiguous(self):
        """Test fallback raises ambiguous error for multiple resources."""
        branch = MockBranch(resources={"a", "b"}, name="test")
        res = FormResources()

        with pytest.raises(AmbiguousResourceError):
            res.resolve_parse_model(branch)


class TestFormResourcesResolveTools:
    """Tests for FormResources.resolve_tools."""

    def test_no_tools_returns_false(self):
        """Test tools=None returns False (lines 103-104)."""
        branch = MockBranch(resources={"tool1", "tool2"}, name="test")
        res = FormResources(tools=None)

        assert res.resolve_tools(branch) is False

    def test_wildcard_returns_true(self):
        """Test tools='*' returns True (lines 105-107)."""
        branch = MockBranch(resources={"tool1", "tool2"}, name="test")
        res = FormResources(tools="*")

        assert res.resolve_tools(branch) is True

    def test_specific_tools_subset_returns_list(self):
        """Test specific tools returns list when valid (lines 108-114)."""
        branch = MockBranch(resources={"search", "calendar", "email"}, name="test")
        res = FormResources(tools=frozenset({"search", "calendar"}))

        result = res.resolve_tools(branch)
        assert isinstance(result, list)
        assert set(result) == {"search", "calendar"}

    def test_tools_not_in_branch_raises_capability_error(self):
        """Test error when tools not in branch resources (lines 110-113)."""
        branch = MockBranch(resources={"search"}, name="test-branch")
        res = FormResources(tools=frozenset({"search", "missing_tool"}))

        with pytest.raises(CapabilityError, match=r"Tools.*not available"):
            res.resolve_tools(branch)

    def test_tools_error_includes_missing_tools(self):
        """Test error message includes missing tools."""
        branch = MockBranch(resources={"search"}, name="test")
        res = FormResources(tools=frozenset({"missing1", "missing2", "search"}))

        with pytest.raises(CapabilityError, match="missing"):
            res.resolve_tools(branch)

    def test_tools_error_includes_branch_name(self):
        """Test error message includes branch name."""
        branch = MockBranch(resources={"a"}, name="tools-branch")
        res = FormResources(tools=frozenset({"missing"}))

        with pytest.raises(CapabilityError, match="tools-branch"):
            res.resolve_tools(branch)

    def test_empty_tools_frozenset_returns_false(self):
        """Test empty frozenset is falsy and returns [] (not False)."""
        branch = MockBranch(resources={"tool1"}, name="test")
        # Empty frozenset is different from None
        res = FormResources(tools=frozenset())

        # Empty frozenset should return empty list (it's subset of everything)
        result = res.resolve_tools(branch)
        assert result == []


class TestParseAssignmentEdgeCases:
    """Tests for parse_assignment edge cases."""

    def test_colon_in_field_name_without_branch(self):
        """Test colon in field name when colon is after arrow."""
        parsed = parse_assignment("input -> field:with:colons")
        assert parsed.branch_name is None
        assert parsed.output_fields == ["field:with:colons"]

    def test_branch_with_colon_in_output(self):
        """Test branch prefix with colon in output field."""
        parsed = parse_assignment("branch: input -> output:name")
        assert parsed.branch_name == "branch"
        assert parsed.output_fields == ["output:name"]

    def test_multiple_colons_before_arrow(self):
        """Test multiple colons before arrow - first is branch."""
        # This creates 'branch' as branch name, 'x:y' as input
        parsed = parse_assignment("branch: x:y -> z")
        assert parsed.branch_name == "branch"
        assert parsed.input_fields == ["x:y"]
        assert parsed.output_fields == ["z"]

    def test_whitespace_only_assignment(self):
        """Test whitespace-only string raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_assignment("   ")

    def test_resources_only_whitespace_returns_defaults(self):
        """Test resources part with only whitespace returns default FormResources (line 225)."""
        # The flow part + | + whitespace
        parsed = parse_assignment("a -> b |   ")
        assert parsed.resources.api is None
        assert parsed.resources.api_gen is None
        assert parsed.resources.tools is None

    def test_complex_assignment_preserves_raw(self):
        """Test raw assignment is preserved."""
        original = "branch: a, b -> c | api:gpt4"
        parsed = parse_assignment(original)
        assert parsed.raw == original


class TestParseResourcesEdgeCases:
    """Tests for _parse_resources edge cases."""

    def test_api_with_special_characters(self):
        """Test api name with special characters."""
        parsed = parse_assignment("a -> b | api:gpt-4-turbo")
        assert parsed.resources.api == "gpt-4-turbo"

    def test_tool_with_underscore(self):
        """Test tool name with underscore."""
        parsed = parse_assignment("a -> b | tool:my_custom_tool")
        assert parsed.resources.tools == frozenset({"my_custom_tool"})

    def test_mixed_spacing_in_resources(self):
        """Test resources with irregular spacing."""
        parsed = parse_assignment("a -> b |  api:gpt4 ,  tool:search ")
        assert parsed.resources.api == "gpt4"
        assert parsed.resources.tools == frozenset({"search"})

    def test_all_api_roles_together(self):
        """Test all api roles specified together."""
        parsed = parse_assignment("a -> b | api:default, api_gen:gen, api_parse:parse")
        assert parsed.resources.api == "default"
        assert parsed.resources.api_gen == "gen"
        assert parsed.resources.api_parse == "parse"
