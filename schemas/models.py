"""
Pydantic Models for the Agentic AI Steam Recommendation System.

This module defines all the data structures used throughout the system,
ensuring type safety and validation at runtime.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# ENUMS
# ============================================================================

class AgentType(str, Enum):
    """Types of agents in the system."""
    PROFILE = "profile"
    RETRIEVAL = "retrieval"
    RANKER = "ranker"
    MATCHMAKING = "matchmaking"


class AgentStatus(str, Enum):
    """Status of an agent's execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RetrievalStrategy(str, Enum):
    """Strategies for candidate retrieval."""
    CONTENT_BASED = "content_based"
    TRENDING = "trending"
    COLLABORATIVE = "collaborative"
    HYBRID = "hybrid"


# ============================================================================
# CORE GAME MODELS
# ============================================================================

class Game(BaseModel):
    """Represents a Steam game with its metadata."""
    
    model_config = ConfigDict(frozen=True)
    
    app_id: int = Field(..., description="Steam application ID")
    name: str = Field(..., description="Game title")
    genres: list[str] = Field(default_factory=list, description="Game genres")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    description: Optional[str] = Field(None, description="Game description")
    release_date: Optional[datetime] = Field(None, description="Release date")
    price: Optional[float] = Field(None, ge=0, description="Current price in USD")
    metacritic_score: Optional[int] = Field(None, ge=0, le=100)
    positive_reviews: int = Field(default=0, ge=0)
    negative_reviews: int = Field(default=0, ge=0)
    
    @property
    def review_ratio(self) -> float:
        """Calculate positive review ratio."""
        total = self.positive_reviews + self.negative_reviews
        return self.positive_reviews / total if total > 0 else 0.0


class OwnedGame(BaseModel):
    """A game owned by a user with playtime data."""
    
    app_id: int
    name: str
    playtime_forever: int = Field(default=0, ge=0, description="Total playtime in minutes")
    playtime_2weeks: int = Field(default=0, ge=0, description="Playtime in last 2 weeks")
    last_played: Optional[datetime] = None
    
    @property
    def playtime_hours(self) -> float:
        """Convert playtime to hours."""
        return self.playtime_forever / 60


class CandidateGame(BaseModel):
    """A game candidate for recommendation with scoring metadata."""
    
    game: Game
    source: RetrievalStrategy = Field(..., description="How this candidate was retrieved")
    retrieval_score: float = Field(default=0.0, ge=0, le=1.0)
    
    # Ranking scores (populated by RankerAgent)
    fit_score: float = Field(default=0.0, ge=0, le=1.0, description="How well it fits user taste")
    novelty_score: float = Field(default=0.0, ge=0, le=1.0, description="How novel/fresh it is")
    constraint_penalty: float = Field(default=0.0, ge=0, le=1.0, description="Penalty for constraints")
    
    # Final computed score
    final_score: float = Field(default=0.0, ge=0, le=1.0)
    explanation: Optional[str] = Field(None, description="Why this game is recommended")
    
    def compute_final_score(
        self,
        fit_weight: float = 0.5,
        novelty_weight: float = 0.3,
        retrieval_weight: float = 0.2,
    ) -> float:
        """Compute weighted final score."""
        base_score = (
            self.fit_score * fit_weight +
            self.novelty_score * novelty_weight +
            self.retrieval_score * retrieval_weight
        )
        self.final_score = max(0.0, base_score - self.constraint_penalty)
        return self.final_score


# ============================================================================
# USER PROFILE MODELS
# ============================================================================

class TasteProfile(BaseModel):
    """Extracted user taste preferences from gaming history."""
    
    preferred_genres: dict[str, float] = Field(
        default_factory=dict,
        description="Genre preferences with weights (0-1)"
    )
    preferred_tags: dict[str, float] = Field(
        default_factory=dict,
        description="Tag preferences with weights (0-1)"
    )
    avg_playtime_per_game: float = Field(default=0.0, ge=0)
    preferred_price_range: tuple[float, float] = Field(default=(0.0, 60.0))
    
    # Behavioral signals
    prefers_multiplayer: float = Field(default=0.5, ge=0, le=1)
    prefers_indie: float = Field(default=0.5, ge=0, le=1)
    engagement_level: float = Field(default=0.5, ge=0, le=1, description="How engaged the user is")
    
    # Time-based preferences
    active_hours: list[int] = Field(
        default_factory=list,
        description="Preferred gaming hours (0-23)"
    )
    active_days: list[int] = Field(
        default_factory=list,
        description="Preferred gaming days (0=Mon, 6=Sun)"
    )


class UserProfile(BaseModel):
    """Complete user profile for the recommendation system."""
    
    steam_id: str = Field(..., description="Steam 64-bit ID")
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    
    # Gaming data
    owned_games: list[OwnedGame] = Field(default_factory=list)
    recently_played: list[OwnedGame] = Field(default_factory=list)
    wishlist: list[int] = Field(default_factory=list, description="App IDs on wishlist")
    
    # Extracted features
    taste_profile: Optional[TasteProfile] = None
    
    # Social
    friends: list[str] = Field(default_factory=list, description="Friend Steam IDs")
    
    # Metadata
    profile_created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def total_playtime_hours(self) -> float:
        """Total playtime across all games in hours."""
        return sum(g.playtime_hours for g in self.owned_games)
    
    @property
    def game_count(self) -> int:
        """Number of owned games."""
        return len(self.owned_games)


# ============================================================================
# MATCHMAKING MODELS
# ============================================================================

class PlayerMatch(BaseModel):
    """A potential player match for multiplayer recommendations."""
    
    steam_id: str
    display_name: Optional[str] = None
    compatibility_score: float = Field(ge=0, le=1)
    shared_games: list[int] = Field(default_factory=list, description="Shared game app IDs")
    taste_similarity: float = Field(ge=0, le=1)
    schedule_overlap: float = Field(ge=0, le=1, description="How much schedules overlap")


class MatchmakingResult(BaseModel):
    """Result of matchmaking process."""
    
    user_id: str
    matches: list[PlayerMatch] = Field(default_factory=list)
    recommended_games_to_play_together: list[int] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# AGENT ORCHESTRATION MODELS
# ============================================================================

# Import ToolCall from core to avoid duplication
from core.tool_interface import ToolCall


class AgentPlan(BaseModel):
    """A plan created by an agent for execution."""
    
    agent_type: AgentType
    objective: str = Field(..., description="What the agent aims to achieve")
    steps: list[str] = Field(default_factory=list, description="Planned steps")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    dependencies: list[AgentType] = Field(
        default_factory=list,
        description="Other agents this plan depends on"
    )
    metadata: dict = Field(default_factory=dict, description="Additional plan metadata")
    
    # Execution metadata
    status: AgentStatus = AgentStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


class AgentResponse(BaseModel):
    """Standard response format from any agent."""
    
    agent_type: AgentType
    success: bool
    data: Any = None
    error: Optional[str] = None
    plan: Optional[AgentPlan] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # For chaining
    next_agent: Optional[AgentType] = None
    context_for_next: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# ORCHESTRATOR MODELS
# ============================================================================

class RecommendationRequest(BaseModel):
    """Input request for generating recommendations."""
    
    steam_id: str
    max_recommendations: int = Field(default=10, ge=1, le=50)
    include_free_games: bool = True
    max_price: Optional[float] = Field(None, ge=0)
    preferred_genres: list[str] = Field(default_factory=list)
    exclude_owned: bool = True
    include_matchmaking: bool = False


class RecommendationResponse(BaseModel):
    """Final output of the recommendation pipeline."""
    
    request: RecommendationRequest
    user_profile: Optional[UserProfile] = None
    recommendations: list[CandidateGame] = Field(default_factory=list)
    matchmaking: Optional[MatchmakingResult] = None
    
    # Pipeline metadata
    pipeline_steps: list[AgentPlan] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def top_recommendation(self) -> Optional[CandidateGame]:
        """Get the top-scored recommendation."""
        if self.recommendations:
            return max(self.recommendations, key=lambda x: x.final_score)
        return None
