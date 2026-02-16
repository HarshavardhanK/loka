"""
Base tool infrastructure for Loka agent.

Uses Pydantic v2 for tool parameter schemas and result validation.
Each tool subclass defines a ``Parameters`` inner model and implements
``execute()``; the JSON schema for the OpenAI function-calling format
is derived automatically from the Pydantic model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Result from a tool execution."""

    success: bool
    output: Any = None
    error: Optional[str] = None


class Tool(ABC):
    """
    Base class for Loka agent tools.

    Tools provide specific capabilities that the agent can invoke
    during multi-step reasoning.

    Subclasses should define:
      - ``name`` / ``description`` class attributes
      - A ``Parameters(BaseModel)`` inner class with typed fields
      - ``execute(**kwargs) -> ToolResult``
    """

    name: str = "base_tool"
    description: str = "Base tool class"

    # Override in subclasses with a Pydantic model
    Parameters: type = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult with success status and output.
        """
        pass

    def to_function_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI-style function schema.

        If the subclass defines a ``Parameters`` Pydantic model the schema
        is generated automatically; otherwise returns an empty ``parameters``
        block.
        """
        if self.Parameters is not None and issubclass(self.Parameters, BaseModel):
            params = self.Parameters.model_json_schema()
        else:
            params = {"type": "object", "properties": {}}

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
        }


class ToolRegistry:
    """
    Registry for managing available tools.

    The registry maintains a collection of tools that the agent
    can access during reasoning.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def register_class(self, tool_class: Type[Tool]) -> None:
        """Register a tool by its class."""
        tool = tool_class()
        self.register(tool)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get function schemas for all registered tools."""
        return [tool.to_function_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name.
            **kwargs: Tool parameters.

        Returns:
            ToolResult from the tool execution.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown tool: {name}",
            )

        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
            )


# Global registry
default_registry = ToolRegistry()
