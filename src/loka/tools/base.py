"""
Base tool infrastructure for Loka agent.

This module provides the base classes for defining and registering
tools that the Loka agent can use for agentic reasoning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


@dataclass
class ToolResult:
    """Result from a tool execution."""
    
    success: bool
    output: Any
    error: Optional[str] = None


class Tool(ABC):
    """
    Base class for Loka agent tools.
    
    Tools provide specific capabilities that the agent can invoke
    during multi-step reasoning.
    """
    
    name: str = "base_tool"
    description: str = "Base tool class"
    
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
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON schema for the tool's parameters.
        
        Returns:
            Dictionary describing the expected parameters.
        """
        pass
    
    def to_function_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI-style function schema.
        
        Returns:
            Function schema dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
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
