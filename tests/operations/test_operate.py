"""Test refactored operate factory."""

import pytest
from pydantic import BaseModel, Field

from lionpride.operations.operate import operate
from lionpride.session import Session


class SimpleModel(BaseModel):
    """Simple test model."""

    title: str = Field(..., description="Title")
    value: int = Field(..., ge=0)


class TestOperateRefactor:
    """Test the refactored modular operate."""

    @pytest.mark.asyncio
    async def test_modular_operate_basic(self):
        """Test basic operation of modular operate."""
        from unittest.mock import AsyncMock, MagicMock

        session = Session()
        branch = session.create_branch()

        # Mock model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        # Mock response
        mock_execution = MagicMock()
        mock_execution.status.value = "completed"
        mock_execution.response.data = {"title": "Test", "value": 42}

        mock_calling = MagicMock()
        mock_calling.execution = mock_execution
        mock_model.invoke.return_value = mock_calling

        # Test parameters
        params = {
            "instruction": "Generate a test response",
            "imodel": mock_model,
            "response_model": SimpleModel,
            "model_kwargs": {"model_name": "test", "temperature": 0.5},
        }

        # Execute
        result = await operate(session, branch, params)

        # Verify
        assert isinstance(result, SimpleModel)
        assert result.title == "Test"
        assert result.value == 42

        # Check that model was invoked
        mock_model.invoke.assert_called_once()
        call_kwargs = mock_model.invoke.call_args.kwargs
        # model_name is passed via model_kwargs
        assert call_kwargs.get("model_name") == "test" or call_kwargs.get("model") == "test"
        assert "messages" in call_kwargs

    @pytest.mark.asyncio
    async def test_modular_with_operative(self):
        """Test modular operate with operative."""
        from unittest.mock import AsyncMock, MagicMock

        from lionpride.operations import create_operative_from_model

        session = Session()
        branch = session.create_branch()

        # Create operative
        operative = create_operative_from_model(SimpleModel)

        # Mock model
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        # Mock response
        mock_execution = MagicMock()
        mock_execution.status.value = "completed"
        mock_execution.response.data = '{"title": "Op Test", "value": 100}'

        mock_calling = MagicMock()
        mock_calling.execution = mock_execution
        mock_model.invoke.return_value = mock_calling

        # Test parameters
        params = {
            "instruction": "Test with operative",
            "imodel": mock_model,
            "operative": operative,
            "model_kwargs": {"model_name": "test"},
        }

        # Execute
        result = await operate(session, branch, params)

        # Verify
        assert result.title == "Op Test"
        assert result.value == 100

    @pytest.mark.asyncio
    async def test_modular_with_lndl(self):
        """Test modular operate with LNDL."""
        from unittest.mock import AsyncMock, MagicMock

        from lionpride.operations import create_operative_from_model

        session = Session()
        branch = session.create_branch()

        # Create operative
        operative = create_operative_from_model(SimpleModel)

        # Mock model with LNDL response
        mock_model = MagicMock()
        mock_model.name = "test_model"
        mock_model.invoke = AsyncMock()

        lndl_response = """<lvar SimpleModel.title t>LNDL Title</lvar>
<lvar SimpleModel.value v>99</lvar>
OUT{simplemodel: [t, v]}"""

        mock_execution = MagicMock()
        mock_execution.status.value = "completed"
        mock_execution.response.data = lndl_response

        mock_calling = MagicMock()
        mock_calling.execution = mock_execution
        mock_model.invoke.return_value = mock_calling

        # Test parameters
        params = {
            "instruction": "Test with LNDL",
            "imodel": mock_model,
            "operative": operative,
            "use_lndl": True,
            "model_kwargs": {"model_name": "test"},
        }

        # Execute
        result = await operate(session, branch, params)

        # Verify LNDL parsing worked
        assert result.title == "LNDL Title"
        assert result.value == 99

    def test_parameter_validation(self):
        """Test parameter validation."""
        import asyncio
        from unittest.mock import MagicMock

        session = Session()
        branch = session.create_branch()

        # Missing instruction
        with pytest.raises(ValueError, match="instruction"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"imodel": MagicMock(), "response_model": SimpleModel},
                )
            )

        # Missing imodel
        with pytest.raises(ValueError, match="imodel"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"instruction": "test", "response_model": SimpleModel},
                )
            )

        # Missing both response_model and operable
        with pytest.raises(ValueError, match=r"response_model.*operable"):
            asyncio.run(
                operate(
                    session,
                    branch,
                    {"instruction": "test", "imodel": MagicMock()},
                )
            )
