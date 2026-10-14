"""
Tool Implementations for the Agentic AI Architecture.

This module provides concrete tool implementations that agents
use to interact with external systems like Steam API, databases,
and vector stores.
"""

import asyncio
import aiohttp
from typing import Any, Optional
import logging

from pydantic import BaseModel, Field

from core.tool_interface import BaseTool, ToolRegistry
from config import STEAM_API_KEY, STEAM_API_BASE_URL


logger = logging.getLogger(__name__)


# Import schemas
from tools.schemas import (
    GetOwnedGamesInput,
    GetOwnedGamesOutput,
    GetPlayerSummaryInput,
    GetPlayerSummaryOutput,
    GetFriendsListInput,
    GetFriendsListOutput,
    GetRecentlyPlayedInput,
    GetRecentlyPlayedOutput,
    GetGameDetailsInput,
    GetGameDetailsOutput,
    SearchGamesInput,
    SearchGamesOutput,
    GetTrendingGamesInput,
    GetTrendingGamesOutput,
    VectorSearchInput,
    VectorSearchOutput,
    IndexGameInput,
    IndexGameOutput,
)

# Re-export schemas for convenience
__all__ = [
    "SteamAPITool",
    "StoreMetadataTool",
    "VectorSearchTool",
    "register_all_tools",
    "GetOwnedGamesInput",
    "GetOwnedGamesOutput",
    "GetPlayerSummaryInput",
    "GetPlayerSummaryOutput",
    "GetFriendsListInput",
    "GetFriendsListOutput",
    "GetRecentlyPlayedInput",
    "GetRecentlyPlayedOutput",
    "GetGameDetailsInput",
    "GetGameDetailsOutput",
    "SearchGamesInput",
    "SearchGamesOutput",
    "GetTrendingGamesInput",
    "GetTrendingGamesOutput",
]


# ============================================================================
# STEAM API TOOL IMPLEMENTATION
# ============================================================================

class SteamAPITool(BaseTool[BaseModel, BaseModel]):
    """
    Tool for interacting with the Steam Web API.
    
    Provides methods for:
    - Fetching user's owned games
    - Getting player summaries
    - Fetching friends lists
    - Getting recently played games
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(
            name="steam_api",
            description="Interact with Steam Web API to fetch user data, games, and social info"
        )
        self.api_key = api_key or STEAM_API_KEY
        self.base_url = STEAM_API_BASE_URL or "https://api.steampowered.com"
    
    @property
    def input_schema(self):
        return BaseModel  # Dynamic based on method
    
    @property
    def output_schema(self):
        return BaseModel  # Dynamic based on method
    
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Route to appropriate method based on input type."""
        if isinstance(input_data, GetOwnedGamesInput):
            return await self.get_owned_games(input_data)
        elif isinstance(input_data, GetPlayerSummaryInput):
            return await self.get_player_summary(input_data)
        elif isinstance(input_data, GetFriendsListInput):
            return await self.get_friends_list(input_data)
        elif isinstance(input_data, GetRecentlyPlayedInput):
            return await self.get_recently_played(input_data)
        else:
            raise ValueError(f"Unknown input type: {type(input_data)}")
    
    async def get_owned_games(self, input_data: GetOwnedGamesInput) -> GetOwnedGamesOutput:
        """Fetch owned games for a Steam user."""
        try:
            url = f"{self.base_url}/IPlayerService/GetOwnedGames/v1/"
            # IMPORTANT: Use 1/0 instead of True/False for API compatibility
            params = {
                "key": self.api_key,
                "steamid": input_data.steam_id,
                "include_appinfo": 1 if input_data.include_appinfo else 0,
                "include_played_free_games": 1 if input_data.include_played_free_games else 0,
                "format": "json",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
            
            response_data = data.get("response", {})
            games = response_data.get("games", [])
            game_count = response_data.get("game_count", 0)
            
            # Check if library is private:
            # If response is empty but no error, it means private profile
            is_private = False
            if not games and "game_count" not in response_data:
                is_private = True
                logger.warning(f"Library is private for {input_data.steam_id}")
            
            # Sort by playtime
            games.sort(key=lambda x: x.get("playtime_forever", 0), reverse=True)
            
            logger.info(f"Fetched {len(games)} owned games for {input_data.steam_id} (private: {is_private})")
            
            return GetOwnedGamesOutput(
                game_count=len(games),
                games=games,
                success=True,
                is_private=is_private,
            )
        except Exception as e:
            logger.error(f"Error fetching owned games: {e}")
            return GetOwnedGamesOutput(
                game_count=0,
                games=[],
                success=False,
                error=str(e),
                is_private=False,
            )
    
    async def get_player_summary(self, input_data: GetPlayerSummaryInput) -> GetPlayerSummaryOutput:
        """Fetch player profile summary."""
        try:
            url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
            params = {
                "key": self.api_key,
                "steamids": input_data.steam_id,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    data = await response.json()
            
            players = data.get("response", {}).get("players", [])
            
            if players:
                player = players[0]
                logger.info(f"Fetched player summary: {player.get('personaname')}")
                return GetPlayerSummaryOutput(
                    steam_id=input_data.steam_id,
                    persona_name=player.get("personaname"),
                    avatar_url=player.get("avatarfull"),
                    profile_url=player.get("profileurl"),
                    success=True
                )
            
            return GetPlayerSummaryOutput(
                steam_id=input_data.steam_id,
                success=False,
                error="Player not found"
            )
        except Exception as e:
            logger.error(f"Error fetching player summary: {e}")
            return GetPlayerSummaryOutput(
                steam_id=input_data.steam_id,
                success=False,
                error=str(e)
            )
    
    async def get_friends_list(self, input_data: GetFriendsListInput) -> GetFriendsListOutput:
        """Fetch user's friends list."""
        try:
            url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
            params = {
                "key": self.api_key,
                "steamid": input_data.steam_id,
                "relationship": "friend",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 401:
                        # Private profile
                        return GetFriendsListOutput(
                            friends=[],
                            friend_count=0,
                            success=True,
                            error="Friends list is private"
                        )
                    data = await response.json()
            
            friends = data.get("friendslist", {}).get("friends", [])
            logger.info(f"Fetched {len(friends)} friends for {input_data.steam_id}")
            
            return GetFriendsListOutput(
                friends=friends,
                friend_count=len(friends),
                success=True
            )
        except Exception as e:
            logger.error(f"Error fetching friends list: {e}")
            return GetFriendsListOutput(
                friends=[],
                friend_count=0,
                success=False,
                error=str(e)
            )
    
    async def get_recently_played(self, input_data: GetRecentlyPlayedInput) -> GetRecentlyPlayedOutput:
        """Fetch recently played games."""
        try:
            url = f"{self.base_url}/IPlayerService/GetRecentlyPlayedGames/v1/"
            params = {
                "key": self.api_key,
                "steamid": input_data.steam_id,
                "count": input_data.count,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    data = await response.json()
            
            games = data.get("response", {}).get("games", [])
            total_count = data.get("response", {}).get("total_count", 0)
            
            logger.info(f"Fetched {len(games)} recently played for {input_data.steam_id}")
            
            return GetRecentlyPlayedOutput(
                games=games,
                total_count=total_count,
                success=True
            )
        except Exception as e:
            logger.error(f"Error fetching recently played: {e}")
            return GetRecentlyPlayedOutput(
                games=[],
                total_count=0,
                success=False,
                error=str(e)
            )


# ============================================================================
# STORE METADATA TOOL IMPLEMENTATION
# ============================================================================

class StoreMetadataTool(BaseTool[BaseModel, BaseModel]):
    """
    Tool for fetching game metadata from Steam Store.
    
    Provides methods for:
    - Getting detailed game information
    - Searching for games
    - Fetching trending/popular games
    """
    
    def __init__(self):
        super().__init__(
            name="store_metadata",
            description="Fetch game metadata, details, and trending info from Steam Store"
        )
        self.store_url = "https://store.steampowered.com/api"
    
    @property
    def input_schema(self):
        return BaseModel
    
    @property
    def output_schema(self):
        return BaseModel
    
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Route to appropriate method based on input type."""
        if isinstance(input_data, GetGameDetailsInput):
            return await self.get_game_details(input_data)
        elif isinstance(input_data, SearchGamesInput):
            return await self.search_games(input_data)
        elif isinstance(input_data, GetTrendingGamesInput):
            return await self.get_trending_games(input_data)
        else:
            raise ValueError(f"Unknown input type: {type(input_data)}")
    
    async def get_game_details(self, input_data: GetGameDetailsInput) -> GetGameDetailsOutput:
        """Fetch detailed information about a specific game."""
        try:
            url = f"{self.store_url}/appdetails"
            params = {"appids": input_data.app_id, "l": "english"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
            
            app_data = data.get(str(input_data.app_id), {})
            
            if app_data.get("success"):
                details = app_data.get("data", {})
                
                # Extract genres
                genres = [g.get("description", "") for g in details.get("genres", [])]
                
                # Extract tags from categories
                tags = [c.get("description", "") for c in details.get("categories", [])]
                
                # Price
                price_data = details.get("price_overview", {})
                price = price_data.get("final", 0) / 100 if price_data else None
                
                logger.info(f"Fetched details for app {input_data.app_id}: {details.get('name')}")
                
                return GetGameDetailsOutput(
                    app_id=input_data.app_id,
                    name=details.get("name"),
                    description=details.get("short_description"),
                    genres=genres,
                    tags=tags,
                    price=price,
                    is_free=details.get("is_free", False),
                    metacritic_score=details.get("metacritic", {}).get("score"),
                    release_date=details.get("release_date", {}).get("date"),
                    success=True
                )
            
            return GetGameDetailsOutput(
                app_id=input_data.app_id,
                success=False,
                error="Game not found"
            )
        except Exception as e:
            logger.error(f"Error fetching game details: {e}")
            return GetGameDetailsOutput(
                app_id=input_data.app_id,
                success=False,
                error=str(e)
            )
    
    async def search_games(self, input_data: SearchGamesInput) -> SearchGamesOutput:
        """Search for games matching criteria."""
        try:
            url = "https://store.steampowered.com/api/storesearch/"
            params = {
                "term": input_data.query,
                "l": "english",
                "cc": "US",
            }
            
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
            
            items = data.get("items", [])
            
            # Filter out DLC, soundtracks, etc.
            excluded = ['soundtrack', 'artbook', 'wallpaper', 'demo', 'digital art']
            results = []
            
            for item in items[:input_data.max_results]:
                name = item.get("name", "").lower()
                if any(skip in name for skip in excluded):
                    continue
                results.append({
                    "app_id": item.get("id"),
                    "name": item.get("name"),
                    "price": item.get("price", {}).get("final", 0) / 100 if item.get("price") else None,
                })
            
            logger.info(f"Search '{input_data.query}' returned {len(results)} results")
            
            return SearchGamesOutput(
                results=results,
                total_found=len(results),
                success=True
            )
        except Exception as e:
            logger.error(f"Error searching games: {e}")
            return SearchGamesOutput(
                results=[],
                total_found=0,
                success=False,
                error=str(e)
            )
    
    async def get_trending_games(self, input_data: GetTrendingGamesInput) -> GetTrendingGamesOutput:
        """Fetch trending/popular games by category."""
        try:
            # Use Steam featured games API
            url = "https://store.steampowered.com/api/featured/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    data = await response.json()
            
            games = []
            
            # Get featured games
            for category in ["featured_win", "large_capsules", "featured_mac", "featured_linux"]:
                for item in data.get(category, []):
                    if len(games) >= input_data.max_results:
                        break
                    
                    app_id = item.get("id")
                    if app_id and not any(g.get("app_id") == app_id for g in games):
                        games.append({
                            "app_id": app_id,
                            "name": item.get("name"),
                            "discounted": item.get("discounted", False),
                            "discount_percent": item.get("discount_percent", 0),
                        })
            
            logger.info(f"Fetched {len(games)} trending games")
            
            return GetTrendingGamesOutput(
                games=games,
                category=input_data.category,
                success=True
            )
        except Exception as e:
            logger.error(f"Error fetching trending games: {e}")
            return GetTrendingGamesOutput(
                games=[],
                category=input_data.category,
                success=False,
                error=str(e)
            )


# ============================================================================
# VECTOR SEARCH TOOL IMPLEMENTATION
# ============================================================================

class VectorSearchTool(BaseTool[BaseModel, BaseModel]):
    """
    Tool for semantic search using vector embeddings.
    
    Provides methods for:
    - Searching games by semantic similarity
    - Indexing games for search
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        super().__init__(
            name="vector_search",
            description="Semantic search for games using vector embeddings"
        )
        self.embedding_model = embedding_model
        from core.embeddings import EmbeddingModel
        from core.vector_store import get_vector_store

        self._embedding_model = EmbeddingModel()
        self._vector_store = get_vector_store()
    
    @property
    def input_schema(self):
        return BaseModel
    
    @property
    def output_schema(self):
        return BaseModel
    
    async def execute(self, input_data: BaseModel) -> BaseModel:
        """Route to appropriate method based on input type."""
        if isinstance(input_data, VectorSearchInput):
            return await self.search(input_data)
        elif isinstance(input_data, IndexGameInput):
            return await self.index_game(input_data)
        else:
            raise ValueError(f"Unknown input type: {type(input_data)}")
    
    async def search(self, input_data: VectorSearchInput) -> VectorSearchOutput:
        """Search for similar games using vector similarity."""
        try:
            top_k = max(1, min(50, input_data.top_k))

            if input_data.query_embedding:
                import numpy as np
                query_embedding = np.array(input_data.query_embedding, dtype=np.float32)
            elif input_data.query_text:
                query_embedding = self._embedding_model.embed_text(input_data.query_text)
            else:
                return VectorSearchOutput(
                    results=[],
                    scores=[],
                    success=False,
                    error="query_text or query_embedding is required",
                )

            search_results = self._vector_store.search(query_embedding, top_k=top_k)
            
            return VectorSearchOutput(
                results=[game for game, _ in search_results if game],
                scores=[float(score) for game, score in search_results if game],
                success=True
            )
        except Exception as e:
            return VectorSearchOutput(
                results=[],
                scores=[],
                success=False,
                error=str(e)
            )
    
    async def index_game(self, input_data: IndexGameInput) -> IndexGameOutput:
        """Index a game in the vector store."""
        try:
            logger.info(f"Indexing game {input_data.app_id}: {input_data.name}")

            game = {
                "app_id": input_data.app_id,
                "name": input_data.name,
                "description": input_data.description,
                "genres": input_data.genres,
                "tags": input_data.tags,
            }

            from core.embeddings import GameEmbeddings
            import numpy as np

            game_embeddings = GameEmbeddings(embedding_model=self._embedding_model)
            embedding = game_embeddings.embed_game(game)
            self._vector_store.add_games([game], np.array([embedding], dtype=np.float32))
            
            return IndexGameOutput(
                app_id=input_data.app_id,
                indexed=True,
                success=True
            )
        except Exception as e:
            return IndexGameOutput(
                app_id=input_data.app_id,
                indexed=False,
                success=False,
                error=str(e)
            )


# ============================================================================
# TOOL REGISTRATION HELPER
# ============================================================================

def register_all_tools(steam_api_key: str = None) -> ToolRegistry:
    """
    Register all tools in the global registry.
    
    Call this during application startup.
    """
    registry = ToolRegistry()
    
    # Register Steam API tool
    steam_tool = SteamAPITool(api_key=steam_api_key)
    registry.register(steam_tool)
    
    # Register Store Metadata tool
    store_tool = StoreMetadataTool()
    registry.register(store_tool)
    
    # Register Vector Search tool
    vector_tool = VectorSearchTool()
    registry.register(vector_tool)
    
    logger.info(f"Registered {len(registry.list_tools())} tools")
    
    return registry
