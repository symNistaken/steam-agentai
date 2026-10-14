"""
Core Tool Interface - Base classes for all tools.

Separated to avoid circular imports between agents and tools.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Optional, TypeVar, Generic
import logging

from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Type variables for generic tool input/output
TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)


class ToolCall(BaseModel):
    """Represents a tool invocation by an agent."""
    tool_name: str
    parameters: dict[str, Any] = {}
    result: Optional[Any] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


class BaseTool(ABC, Generic[TInput, TOutput]):
    """
    Abstract base class for all tools that agents can invoke.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._call_count = 0
        self._total_execution_time = 0.0
    
    @property
    @abstractmethod
    def input_schema(self) -> type[TInput]:
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> type[TOutput]:
        pass
    
    @abstractmethod
    async def execute(self, input_data: TInput) -> TOutput:
        pass
    
    async def __call__(self, input_data: TInput) -> ToolCall:
        start_time = datetime.utcnow()
        tool_call = ToolCall(
            tool_name=self.name,
            parameters=input_data.model_dump() if isinstance(input_data, BaseModel) else {},
        )
        
        try:
            result = await self.execute(input_data)
            tool_call.result = result.model_dump() if isinstance(result, BaseModel) else result
        except Exception as e:
            tool_call.error = str(e)
            logger.error(f"Tool {self.name} failed: {e}")
        finally:
            end_time = datetime.utcnow()
            tool_call.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            self._call_count += 1
            self._total_execution_time += tool_call.execution_time_ms or 0
            
        return tool_call
    
    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "call_count": self._call_count,
            "total_execution_time_ms": self._total_execution_time,
            "avg_execution_time_ms": (
                self._total_execution_time / self._call_count 
                if self._call_count > 0 else 0
            ),
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    _instance: Optional["ToolRegistry"] = None
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance
    
    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    
    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]
    
    def clear(self) -> None:
        self._tools.clear()
