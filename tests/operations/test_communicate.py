# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for communicate.py with composed params."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from lionpride.session.messages import Message


class TestCommunicateCoverage:
    """Test communicate.py with composed params."""

    async def test_communicate_missing_generate_param(self, session_with_model):
        """Test missing generate param raises ValueError."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = CommunicateParams()  # No generate param

        with pytest.raises(ValueError, match="communicate requires 'generate' parameter"):
            await communicate(session, branch, params)

    async def test_communicate_missing_instruction(self, session_with_model):
        """Test missing instruction raises ValueError."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, model = session_with_model
        branch = session.create_branch(name="test")

        params = CommunicateParams(
            generate=GenerateParams(imodel=model),  # No instruction
        )

        with pytest.raises(
            ValueError, match=r"communicate requires 'generate\.instruction' parameter"
        ):
            await communicate(session, branch, params)

    async def test_communicate_missing_imodel(self, session_with_model):
        """Test missing imodel raises ValueError."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, _model = session_with_model
        branch = session.create_branch(name="test")

        params = CommunicateParams(
            generate=GenerateParams(instruction="Test"),  # No imodel
        )

        with pytest.raises(ValueError, match=r"communicate requires 'generate\.imodel' parameter"):
            await communicate(session, branch, params)

    async def test_communicate_resource_access_denied(self, session_with_model):
        """Test branch without resource access raises PermissionError."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, model = session_with_model
        # Create branch without access to the model
        branch = session.create_branch(name="restricted", resources=set())

        params = CommunicateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel=model,
            ),
        )

        with pytest.raises(PermissionError, match="cannot access model"):
            await communicate(session, branch, params)

    async def test_communicate_return_as_text(self, session_with_model):
        """Test return_as='text' returns string."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = CommunicateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel=model,
            ),
            return_as="text",
        )

        result = await communicate(session, branch, params)
        assert isinstance(result, str)

    async def test_communicate_return_as_raw(self, session_with_model):
        """Test return_as='raw' returns dict."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = CommunicateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel=model,
            ),
            return_as="raw",
        )

        result = await communicate(session, branch, params)
        assert isinstance(result, dict)

    async def test_communicate_return_as_message(self, session_with_model):
        """Test return_as='message' returns Message."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        params = CommunicateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel=model,
            ),
            return_as="message",
        )

        result = await communicate(session, branch, params)
        assert isinstance(result, Message)

    async def test_format_result_text_with_basemodel(self):
        """Test text format with BaseModel returns model_dump_json."""
        from lionpride.operations.operate.communicate import _format_result

        class TestModel(BaseModel):
            value: str

        validated = TestModel(value="test")
        result = _format_result(
            return_as="text",
            validated=validated,
            response_text="raw text",
            raw_response={},
            assistant_msg=MagicMock(),
        )

        assert isinstance(result, str)
        assert "test" in result

    async def test_format_result_model(self):
        """Test model format returns validated model."""
        from lionpride.operations.operate.communicate import _format_result

        class TestModel(BaseModel):
            value: str

        validated = TestModel(value="test")
        result = _format_result(
            return_as="model",
            validated=validated,
            response_text="raw text",
            raw_response={},
            assistant_msg=MagicMock(),
        )

        assert isinstance(result, TestModel)
        assert result.value == "test"

    async def test_format_result_invalid_return_as(self):
        """Test invalid return_as raises ValueError."""
        from lionpride.operations.operate.communicate import _format_result

        with pytest.raises(ValueError, match="Unsupported return_as"):
            _format_result(
                return_as="invalid_type",
                validated=None,
                response_text="text",
                raw_response={},
                assistant_msg=MagicMock(),
            )

    async def test_communicate_with_string_imodel_name(self, session_with_model):
        """Test imodel as string name resolution."""
        from lionpride.operations.operate.communicate import communicate
        from lionpride.operations.operate.types import CommunicateParams, GenerateParams

        session, _model = session_with_model
        branch = session.create_branch(name="test")
        branch.resources.add("mock_model")

        # Pass imodel as string name
        params = CommunicateParams(
            generate=GenerateParams(
                instruction="Test",
                imodel="mock_model",  # String name, not object
            ),
        )

        result = await communicate(session, branch, params)
        assert isinstance(result, str)
