"""
Base Agent and Tool Framework for the Agentic AI Architecture.

This module provides the abstract base classes and patterns that all
specific agents (ProfileAgent, RetrievalAgent, etc.) will inherit from.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
import logging

from pydantic import BaseModel

from schemas.models import (
    AgentType,
    AgentStatus,
    AgentPlan,
    AgentResponse,
)
from core.tool_interface import BaseTool, ToolRegistry, ToolCall


# Configure logging
logger = logging.getLogger(__name__)


# Re-export for convenience
__all__ = ["BaseAgent", "AgentContext", "AgentRegistry", "BaseTool", "ToolRegistry", "ToolCall"]


# ============================================================================
# AGENT CONTEXT
# ============================================================================

class AgentContext(BaseModel):
    """
    Shared context passed between agents during orchestration.
    
    This allows agents to share data and state without tight coupling.
    """
    
    # Request context
    steam_id: str
    request_id: str
    
    # Shared state (populated by agents)
    user_profile: Optional[dict[str, Any]] = None
    taste_profile: Optional[dict[str, Any]] = None
    candidate_games: list[dict[str, Any]] = []
    ranked_games: list[dict[str, Any]] = []
    matchmaking_results: Optional[dict[str, Any]] = None
    
    # Execution trace
    executed_plans: list[AgentPlan] = []
    errors: list[str] = []
    
    # Metadata
    created_at: datetime = datetime.utcnow()
    
    def add_error(self, error: str) -> None:
        """Add an error to the context."""
        self.errors.append(error)
        logger.error(f"Context error: {error}")
    
    def get_data(self, key: str) -> Optional[Any]:
        """Safely get data from context."""
        return getattr(self, key, None)
    
    def set_data(self, key: str, value: Any) -> None:
        """Set data in context if attribute exists."""
        if hasattr(self, key):
            setattr(self, key, value)


# ============================================================================
# BASE AGENT
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    Agents are autonomous units that:
    1. Receive context and objectives
    2. Plan their execution strategy
    3. Execute tools to gather data/perform actions
    4. Return structured responses
    
    Subclasses must implement:
    - agent_type: The type of this agent
    - plan(): Create an execution plan
    - execute(): Execute the plan using tools
    """
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass
    
    @property
    def available_tools(self) -> list[str]:
        """List tools this agent can use (override to restrict)."""
        return [t["name"] for t in self.tool_registry.list_tools()]
    
    @abstractmethod
    async def plan(self, context: AgentContext, objective: str) -> AgentPlan:
        """
        Create an execution plan for the given objective.
        
        Args:
            context: Shared context with data from previous agents
            objective: What this agent should accomplish
            
        Returns:
            AgentPlan with steps and tool calls to execute
        """
        pass
    
    @abstractmethod
    async def execute(self, context: AgentContext, plan: AgentPlan) -> AgentResponse:
        """
        Execute the plan and return results.
        
        Args:
            context: Shared context (will be mutated with results)
            plan: The plan to execute
            
        Returns:
            AgentResponse with success/failure and data
        """
        pass
    
    async def run(
        self, 
        context: AgentContext, 
        objective: str,
        next_agent: Optional[AgentType] = None,
    ) -> AgentResponse:
        """
        Full agent lifecycle: plan -> execute -> respond.
        
        This is the main entry point for running an agent.
        """
        self._logger.info(f"Agent {self.agent_type.value} starting with objective: {objective}")
        
        # Create plan
        plan = await self.plan(context, objective)
        plan.status = AgentStatus.RUNNING
        plan.started_at = datetime.utcnow()
        
        try:
            # Execute plan
            response = await self.execute(context, plan)
            
            # Update plan status
            plan.status = AgentStatus.COMPLETED
            plan.completed_at = datetime.utcnow()
            
            # Set next agent in chain
            response.next_agent = next_agent
            response.plan = plan
            
            # Add plan to context trace
            context.executed_plans.append(plan)
            
            self._logger.info(
                f"Agent {self.agent_type.value} completed in {plan.duration_ms:.2f}ms"
            )
            
        except Exception as e:
            plan.status = AgentStatus.FAILED
            plan.completed_at = datetime.utcnow()
            context.add_error(f"{self.agent_type.value}: {str(e)}")
            
            response = AgentResponse(
                agent_type=self.agent_type,
                success=False,
                error=str(e),
                plan=plan,
            )
            self._logger.error(f"Agent {self.agent_type.value} failed: {e}")
        
        return response
    
    async def invoke_tool(self, tool_name: str, input_data: BaseModel) -> ToolCall:
        """
        Invoke a tool by name with the given input.
        
        Args:
            tool_name: Name of the tool to invoke
            input_data: Input data for the tool
            
        Returns:
            ToolCall with result or error
        """
        tool = self.tool_registry.get(tool_name)
        if tool is None:
            return ToolCall(
                tool_name=tool_name,
                parameters=input_data.model_dump(),
                error=f"Tool '{tool_name}' not found",
            )
        
        return await tool(input_data)
    
    def _create_base_plan(self, objective: str) -> AgentPlan:
        """Helper to create a base plan with common fields."""
        return AgentPlan(
            agent_type=self.agent_type,
            objective=objective,
            status=AgentStatus.PENDING,
        )


# ============================================================================
# AGENT REGISTRY
# ============================================================================

class AgentRegistry:
    """
    Registry for managing agent instances.
    
    The orchestrator uses this to get agents by type.
    """
    
    _instance: Optional["AgentRegistry"] = None
    
    def __new__(cls) -> "AgentRegistry":
        """Singleton pattern for global agent registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
        return cls._instance
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        self._agents[agent.agent_type] = agent
        logger.info(f"Registered agent: {agent.agent_type.value}")
    
    def get(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Get an agent by type."""
        return self._agents.get(agent_type)
    
    def list_agents(self) -> list[AgentType]:
        """List all registered agent types."""
        return list(self._agents.keys())
    
    def clear(self) -> None:
        """Clear all registered agents (useful for testing)."""
        self._agents.clear()


# ============================================================================
# DECORATOR FOR TOOL REGISTRATION
# ============================================================================

def register_tool(name: str, description: str):
    """
    Decorator to easily register a function as a tool.
    
    Usage:
        @register_tool("fetch_user", "Fetch user from Steam API")
        async def fetch_user(input: FetchUserInput) -> FetchUserOutput:
            ...
    """
    def decorator(func: Callable):
        # Create a dynamic tool class
        class DynamicTool(BaseTool):
            def __init__(self):
                super().__init__(name, description)
                self._func = func
            
            @property
            def input_schema(self):
                # Get from type hints
                hints = func.__annotations__
                return hints.get(list(hints.keys())[0], BaseModel)
            
            @property
            def output_schema(self):
                hints = func.__annotations__
                return hints.get("return", BaseModel)
            
            async def execute(self, input_data):
                return await self._func(input_data)
        
        # Register the tool
        tool = DynamicTool()
        ToolRegistry().register(tool)
        
        return func
    
    return decorator
