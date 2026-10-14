"""
Tool Input/Output Schemas.

Separated from tools.base to avoid circular imports.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# STEAM API TOOL SCHEMAS
# ============================================================================

class GetOwnedGamesInput(BaseModel):
    """Input for fetching user's owned games."""
    steam_id: str
    include_appinfo: bool = True
    include_played_free_games: bool = True


class GetOwnedGamesOutput(BaseModel):
    """Output containing owned games data."""
    game_count: int
    games: list[dict[str, Any]]
    success: bool
    error: Optional[str] = None
    is_private: bool = False  # Flag to indicate if library is private


class GetPlayerSummaryInput(BaseModel):
    """Input for fetching player profile summary."""
    steam_id: str


class GetPlayerSummaryOutput(BaseModel):
    """Output containing player summary."""
    steam_id: str
    persona_name: Optional[str] = None
    avatar_url: Optional[str] = None
    profile_url: Optional[str] = None
    success: bool
    error: Optional[str] = None


class GetFriendsListInput(BaseModel):
    """Input for fetching user's friends list."""
    steam_id: str


class GetFriendsListOutput(BaseModel):
    """Output containing friends list."""
    friends: list[dict[str, Any]]
    friend_count: int
    success: bool
    error: Optional[str] = None


class GetRecentlyPlayedInput(BaseModel):
    """Input for fetching recently played games."""
    steam_id: str
    count: int = 10


class GetRecentlyPlayedOutput(BaseModel):
    """Output containing recently played games."""
    games: list[dict[str, Any]]
    total_count: int
    success: bool
    error: Optional[str] = None


# ============================================================================
# STORE METADATA TOOL SCHEMAS
# ============================================================================

class GetGameDetailsInput(BaseModel):
    """Input for fetching game details from Steam Store."""
    app_id: int


class GetGameDetailsOutput(BaseModel):
    """Output containing game details."""
    app_id: int
    name: Optional[str] = None
    description: Optional[str] = None
    genres: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    price: Optional[float] = None
    is_free: bool = False
    metacritic_score: Optional[int] = None
    release_date: Optional[str] = None
    success: bool
    error: Optional[str] = None


class SearchGamesInput(BaseModel):
    """Input for searching games in the store."""
    query: str
    max_results: int = 20
    filters: dict[str, Any] = Field(default_factory=dict)


class SearchGamesOutput(BaseModel):
    """Output containing search results."""
    results: list[dict[str, Any]]
    total_found: int
    success: bool
    error: Optional[str] = None


class GetTrendingGamesInput(BaseModel):
    """Input for fetching trending/popular games."""
    category: str = "top_sellers"  # top_sellers, new_releases, upcoming
    max_results: int = 20


class GetTrendingGamesOutput(BaseModel):
    """Output containing trending games."""
    games: list[dict[str, Any]]
    category: str
    success: bool
    error: Optional[str] = None


# ============================================================================
# VECTOR SEARCH TOOL SCHEMAS
# ============================================================================

class VectorSearchInput(BaseModel):
    """Input for semantic vector search."""
    query_embedding: list[float] = Field(default_factory=list)
    query_text: Optional[str] = None  # Will be embedded if provided
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)


class VectorSearchOutput(BaseModel):
    """Output containing vector search results."""
    results: list[dict[str, Any]]
    scores: list[float]
    success: bool
    error: Optional[str] = None


class IndexGameInput(BaseModel):
    """Input for indexing a game in the vector store."""
    app_id: int
    name: str
    description: str
    genres: list[str]
    tags: list[str]


class IndexGameOutput(BaseModel):
    """Output from indexing operation."""
    app_id: int
    indexed: bool
    success: bool
    error: Optional[str] = None
