"""
Local model support for CPU-only environments.
Provides agent implementations using smaller, CPU-friendly models or dummy agents.
"""
from .runner import LocalAgent, DummyLocalAgent, TransformerAgent, create_agent, MultiAgentLocalRunner

__all__ = [
    "LocalAgent",
    "DummyLocalAgent",
    "TransformerAgent",
    "create_agent",
    "MultiAgentLocalRunner",
]
