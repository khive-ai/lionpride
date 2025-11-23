"""Operate operation - modularized components.

Components:
- generate.py: Basic generation operation (stateless model invocation)
- factory.py: Operate factory orchestrating structured output
- react.py: ReAct loop for multi-step reasoning with actions
- message_prep.py: Message preparation and LNDL spec generation
- response_parser.py: Response parsing (JSON/LNDL) and tool execution
- operative.py: Operative validation framework
"""

from .factory import operate
from .generate import generate
from .operative import Operative, create_action_operative, create_operative_from_model
from .react import ReactResult, ReactStep, react

__all__ = (
    "Operative",
    "ReactResult",
    "ReactStep",
    "create_action_operative",
    "create_operative_from_model",
    "generate",
    "operate",
    "react",
)
