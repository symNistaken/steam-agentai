"""
Orchestrator - The Brain of the Agentic AI Architecture.

The Orchestrator is responsible for:
1. Receiving recommendation requests
2. Planning the execution flow
3. Coordinating agent execution in sequence
4. Augmenting with RAG (Retrieval Augmented Generation)
5. Aggregating results with explainability
6. Handling errors and retries

This is the main workflow manager that ties all agents together.

ENHANCED WITH:
- RAG (Retrieval Augmented Generation) for context-aware recommendations
- XAI (Explainable AI) for interpretable decisions
- Deep Learning embeddings for semantic understanding
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from schemas.models import (
    AgentType,
    AgentStatus,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    CandidateGame,
    MatchmakingResult,
)
from agents.base import AgentContext, AgentRegistry, BaseAgent
from services.rag_service import RAGService
from core.explainability import get_recommendation_explainer


logger = logging.getLogger(__name__)


# ============================================================================
# ORCHESTRATOR CONFIGURATION
# ============================================================================

class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    def __init__(
        self,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        enable_parallel_execution: bool = False,
        enable_matchmaking: bool = True,
        enable_rag: bool = True,
        enable_xai: bool = True,
    ):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_parallel_execution = enable_parallel_execution
        self.enable_matchmaking = enable_matchmaking
        self.enable_rag = enable_rag
        self.enable_xai = enable_xai


# ============================================================================
# PIPELINE DEFINITION
# ============================================================================

class PipelineStep:
    """Defines a step in the recommendation pipeline."""
    
    def __init__(
        self,
        agent_type: AgentType,
        objective: str,
        depends_on: list[AgentType] = None,
        optional: bool = False,
    ):
        self.agent_type = agent_type
        self.objective = objective
        self.depends_on = depends_on or []
        self.optional = optional


# Default recommendation pipeline
DEFAULT_PIPELINE = [
    PipelineStep(
        agent_type=AgentType.PROFILE,
        objective="Build user profile and extract taste preferences from Steam data",
        depends_on=[],
    ),
    PipelineStep(
        agent_type=AgentType.RETRIEVAL,
        objective="Retrieve candidate games based on user taste profile",
        depends_on=[AgentType.PROFILE],
    ),
    PipelineStep(
        agent_type=AgentType.RANKER,
        objective="Rank and score candidates based on fit, novelty, and constraints",
        depends_on=[AgentType.RETRIEVAL],
    ),
    PipelineStep(
        agent_type=AgentType.MATCHMAKING,
        objective="Find compatible players for multiplayer recommendations",
        depends_on=[AgentType.PROFILE],
        optional=True,  # Only runs if requested
    ),
]


# ============================================================================
# ORCHESTRATOR IMPLEMENTATION
# ============================================================================

class Orchestrator:
    """
    The main workflow manager for the recommendation system.
    
    Coordinates agent execution, augments with RAG, and explains decisions.
    
    FEATURES:
    - Multi-agent orchestration
    - RAG (Retrieval Augmented Generation) context
    - XAI (Explainable AI) for each recommendation
    - Deep Learning embeddings for semantic matching
    
    Usage:
        orchestrator = Orchestrator()
        response = await orchestrator.run(request)
    """
    
    def __init__(
        self,
        config: OrchestratorConfig = None,
        pipeline: list[PipelineStep] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.pipeline = pipeline or DEFAULT_PIPELINE
        self.agent_registry = AgentRegistry()
        self.rag_service = RAGService() if self.config.enable_rag else None
        self.explainer = get_recommendation_explainer() if self.config.enable_xai else None
        self._logger = logging.getLogger(self.__class__.__name__)
    
    async def run(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Execute the full recommendation pipeline.
        
        This is the main entry point for generating recommendations.
        
        Args:
            request: The recommendation request parameters
            
        Returns:
            RecommendationResponse with recommendations and metadata
        """
        start_time = datetime.utcnow()
        request_id = str(uuid.uuid4())
        
        self._logger.info(f"Starting pipeline for request {request_id}")
        
        # Initialize shared context
        context = AgentContext(
            steam_id=request.steam_id,
            request_id=request_id,
        )
        
        # Initialize response
        response = RecommendationResponse(
            request=request,
            pipeline_steps=[],
        )
        
        try:
            # Execute pipeline steps
            await self._execute_pipeline(context, request, response)
            
            # Build final response from context
            response = self._build_response(context, request, response)
            
        except Exception as e:
            self._logger.error(f"Pipeline failed: {e}")
            context.add_error(str(e))
        
        # Calculate total duration
        end_time = datetime.utcnow()
        response.total_duration_ms = (end_time - start_time).total_seconds() * 1000
        response.generated_at = end_time
        
        self._logger.info(
            f"Pipeline completed in {response.total_duration_ms:.2f}ms "
            f"with {len(response.recommendations)} recommendations"
        )
        
        return response
    
    async def _execute_pipeline(
        self,
        context: AgentContext,
        request: RecommendationRequest,
        response: RecommendationResponse,
    ) -> None:
        """Execute each step in the pipeline sequentially."""
        
        completed_agents = set()
        
        for step in self.pipeline:
            # Skip optional steps if not requested
            if step.optional:
                if step.agent_type == AgentType.MATCHMAKING and not request.include_matchmaking:
                    self._logger.info(f"Skipping optional step: {step.agent_type.value}")
                    continue
            
            # Check dependencies
            if not all(dep in completed_agents for dep in step.depends_on):
                self._logger.warning(
                    f"Skipping {step.agent_type.value}: dependencies not met"
                )
                continue
            
            # Get agent from registry
            agent = self.agent_registry.get(step.agent_type)
            if agent is None:
                self._logger.warning(f"Agent not registered: {step.agent_type.value}")
                continue
            
            # Execute agent with retry logic
            agent_response = await self._execute_with_retry(
                agent=agent,
                context=context,
                objective=step.objective,
            )
            
            # Track completion
            if agent_response.success:
                completed_agents.add(step.agent_type)
                if agent_response.plan:
                    response.pipeline_steps.append(agent_response.plan)
            else:
                self._logger.error(
                    f"Agent {step.agent_type.value} failed: {agent_response.error}"
                )
                # Continue pipeline on non-critical failures
                if not step.optional:
                    context.add_error(f"{step.agent_type.value}: {agent_response.error}")
    
    async def _execute_with_retry(
        self,
        agent: BaseAgent,
        context: AgentContext,
        objective: str,
    ):
        """Execute an agent with retry logic."""
        
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Execute with timeout
                response = await asyncio.wait_for(
                    agent.run(context, objective),
                    timeout=self.config.timeout_seconds,
                )
                
                if response.success:
                    return response
                
                last_error = response.error
                self._logger.warning(
                    f"Attempt {attempt + 1} failed for {agent.agent_type.value}: {last_error}"
                )
                
            except asyncio.TimeoutError:
                last_error = "Timeout exceeded"
                self._logger.warning(
                    f"Attempt {attempt + 1} timed out for {agent.agent_type.value}"
                )
            except Exception as e:
                last_error = str(e)
                self._logger.error(
                    f"Attempt {attempt + 1} raised exception for {agent.agent_type.value}: {e}"
                )
        
        # All retries failed
        from schemas.models import AgentResponse
        return AgentResponse(
            agent_type=agent.agent_type,
            success=False,
            error=f"Failed after {self.config.max_retries} attempts: {last_error}",
        )
    
    def _build_response(
        self,
        context: AgentContext,
        request: RecommendationRequest,
        response: RecommendationResponse,
    ) -> RecommendationResponse:
        """Build final response from execution context, with RAG and XAI."""
        
        # Parse user profile from context
        if context.user_profile:
            response.user_profile = UserProfile(**context.user_profile)
        
        # Parse and sort recommendations
        if context.ranked_games:
            recommendations = [
                CandidateGame(**game) for game in context.ranked_games
            ]
            # Sort by final score descending
            recommendations.sort(key=lambda x: x.final_score, reverse=True)
            # Limit to requested amount
            recommendations = recommendations[:request.max_recommendations]
            
            # Add XAI explanations if enabled
            if self.config.enable_xai and context.user_profile:
                self._logger.info("Generating XAI explanations for recommendations...")
                for rec in recommendations:
                    game_dict = {
                        "name": rec.name,
                        "genres": rec.genres,
                        "tags": getattr(rec, "tags", []),
                        "rating": rec.final_score,
                    }
                    explanation = self.explainer.explain_recommendation(
                        game=game_dict,
                        player_profile=context.user_profile,
                        recommendation_score=rec.final_score,
                        language=request.language or "en"
                    )
                    # Store explanation in recommendation
                    rec.explanation = explanation.get("explanation", "")
                    rec.confidence = explanation.get("confidence", "MEDIUM")
            
            response.recommendations = recommendations
        
        # Add RAG context to response metadata if enabled
        if self.config.enable_rag and context.user_profile:
            response.metadata = response.metadata or {}
            response.metadata["rag_enabled"] = True
            response.metadata["retrieval_strategy"] = "hybrid"
        
        # Parse matchmaking results
        if context.matchmaking_results and request.include_matchmaking:
            response.matchmaking = MatchmakingResult(**context.matchmaking_results)
        
        return response


# ============================================================================
# ORCHESTRATOR FACTORY
# ============================================================================

class OrchestratorFactory:
    """Factory for creating configured orchestrators."""
    
    @staticmethod
    def create_default() -> Orchestrator:
        """Create orchestrator with default configuration (RAG + XAI enabled)."""
        config = OrchestratorConfig(
            enable_rag=True,
            enable_xai=True,
        )
        return Orchestrator(config=config)
    
    @staticmethod
    def create_fast() -> Orchestrator:
        """Create orchestrator optimized for speed (RAG enabled, XAI disabled)."""
        config = OrchestratorConfig(
            max_retries=1,
            timeout_seconds=15.0,
            enable_matchmaking=False,
            enable_rag=True,
            enable_xai=False,
        )
        return Orchestrator(config=config)
    
    @staticmethod
    def create_comprehensive() -> Orchestrator:
        """Create orchestrator with all features enabled (RAG + XAI + Matchmaking)."""
        config = OrchestratorConfig(
            max_retries=3,
            timeout_seconds=60.0,
            enable_matchmaking=True,
            enable_rag=True,
            enable_xai=True,
        )
        return Orchestrator(config=config)
    
    @staticmethod
    def create_minimal() -> Orchestrator:
        """Create minimal orchestrator (no RAG, no XAI)."""
        config = OrchestratorConfig(
            max_retries=1,
            timeout_seconds=15.0,
            enable_matchmaking=False,
            enable_rag=False,
            enable_xai=False,
        )
        return Orchestrator(config=config)


# ============================================================================
# EXAMPLE USAGE (for documentation)
# ============================================================================

async def example_usage():
    """
    Example showing how to use the orchestrator.
    
    This is pseudo-code demonstrating the flow:
    
    1. ProfileAgent fetches Steam data and builds taste profile
    2. RetrievalAgent gets candidate games based on taste
    3. RankerAgent scores and ranks candidates
    4. (Optional) MatchmakingAgent finds compatible players
    """
    
    # Create orchestrator
    orchestrator = OrchestratorFactory.create_default()
    
    # Create request
    request = RecommendationRequest(
        steam_id="76561198012345678",
        max_recommendations=10,
        include_free_games=True,
        max_price=30.0,
        exclude_owned=True,
        include_matchmaking=True,
    )
    
    # Run pipeline
    response = await orchestrator.run(request)
    
    # Access results
    print(f"Generated {len(response.recommendations)} recommendations")
    print(f"Pipeline took {response.total_duration_ms:.2f}ms")
    
    for rec in response.recommendations[:5]:
        print(f"- {rec.game.name}: {rec.final_score:.2f} ({rec.explanation})")
    
    return response


# ============================================================================
# PIPELINE FLOW DIAGRAM (ASCII)
# ============================================================================
"""
RECOMMENDATION PIPELINE FLOW
============================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                      │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   REQUEST   │───▶│   CONTEXT   │───▶│   AGENTS    │───▶│  RESPONSE   │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENT PIPELINE                                     │
│                                                                             │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │  PROFILE   │────▶│ RETRIEVAL  │────▶│   RANKER   │────▶│ MATCHMAKE  │   │
│  │   AGENT    │     │   AGENT    │     │   AGENT    │     │   AGENT    │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐   │
│  │  User      │     │  Candidate │     │  Ranked    │     │  Player    │   │
│  │  Profile   │     │  Games     │     │  Games     │     │  Matches   │   │
│  │  + Taste   │     │  Pool      │     │  + Scores  │     │            │   │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TOOLS                                           │
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │   STEAM API    │  │ STORE METADATA │  │ VECTOR SEARCH  │                │
│  │                │  │                │  │                │                │
│  │ - Owned Games  │  │ - Game Details │  │ - Similarity   │                │
│  │ - Friends      │  │ - Trending     │  │ - Embeddings   │                │
│  │ - Recently     │  │ - Search       │  │                │                │
│  │   Played       │  │                │  │                │                │
│  └────────────────┘  └────────────────┘  └────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

DATA FLOW:
----------
1. Request comes in with steam_id and preferences
2. ProfileAgent:
   - Calls Steam API to get owned games, friends, recently played
   - Extracts taste profile (genres, tags, playtime patterns)
   - Stores in context.user_profile and context.taste_profile
   
3. RetrievalAgent:
   - Uses taste profile to query for similar games (content-based)
   - Fetches trending games for discovery (trending)
   - Combines into candidate pool
   - Stores in context.candidate_games
   
4. RankerAgent:
   - Scores each candidate on fit, novelty, constraints
   - Applies user preferences (price, owned exclusion)
   - Generates explanations
   - Stores in context.ranked_games
   
5. MatchmakingAgent (optional):
   - Finds friends with similar taste profiles
   - Identifies games to play together
   - Stores in context.matchmaking_results
   
6. Orchestrator builds final response from context
"""
