"""
ProfileAgent - Responsible for building user profiles.

This agent:
1. Fetches user's Steam data (owned games, friends, activity)
2. Extracts taste preferences (genres, tags, playtime patterns)
3. Creates a comprehensive UserProfile with TasteProfile
"""

from datetime import datetime
from typing import Any

from agents.base import BaseAgent, AgentContext
from schemas.models import (
    AgentType,
    AgentPlan,
    AgentResponse,
    UserProfile,
    OwnedGame,
    TasteProfile,
)
from tools.schemas import (
    GetOwnedGamesInput,
    GetPlayerSummaryInput,
    GetFriendsListInput,
    GetRecentlyPlayedInput,
)


class ProfileAgent(BaseAgent):
    """
    Agent responsible for building comprehensive user profiles.
    
    Flow:
    1. Fetch player summary (name, avatar)
    2. Fetch owned games with playtime
    3. Fetch friends list
    4. Fetch recently played games
    5. Extract taste profile from gaming history
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.PROFILE
    
    async def plan(self, context: AgentContext, objective: str) -> AgentPlan:
        """Create execution plan for profile building."""
        plan = self._create_base_plan(objective)
        
        plan.steps = [
            "Fetch player summary from Steam API",
            "Fetch owned games with playtime data",
            "Fetch friends list for social features",
            "Fetch recently played games for recency signal",
            "Extract taste profile from gaming history",
            "Build and store complete user profile",
        ]
        
        return plan
    
    async def execute(self, context: AgentContext, plan: AgentPlan) -> AgentResponse:
        """Execute profile building pipeline."""
        steam_id = context.steam_id
        
        # Step 1: Fetch player summary
        summary_call = await self.invoke_tool(
            "steam_api",
            GetPlayerSummaryInput(steam_id=steam_id),
        )
        plan.tool_calls.append(summary_call)
        
        # Step 2: Fetch owned games
        games_call = await self.invoke_tool(
            "steam_api",
            GetOwnedGamesInput(
                steam_id=steam_id,
                include_appinfo=True,
                include_played_free_games=True,
            ),
        )
        plan.tool_calls.append(games_call)
        
        # Step 3: Fetch friends
        friends_call = await self.invoke_tool(
            "steam_api",
            GetFriendsListInput(steam_id=steam_id),
        )
        plan.tool_calls.append(friends_call)
        
        # Step 4: Fetch recently played
        recent_call = await self.invoke_tool(
            "steam_api",
            GetRecentlyPlayedInput(steam_id=steam_id, count=20),
        )
        plan.tool_calls.append(recent_call)
        
        # Parse results
        owned_games = self._parse_owned_games(games_call.result)
        recently_played = self._parse_owned_games(recent_call.result)
        friends = self._parse_friends(friends_call.result)
        
        # Step 5: Extract taste profile
        taste_profile = await self._extract_taste_profile(owned_games)
        
        # Step 6: Build user profile
        profile = UserProfile(
            steam_id=steam_id,
            display_name=summary_call.result.get("persona_name") if summary_call.result else None,
            avatar_url=summary_call.result.get("avatar_url") if summary_call.result else None,
            owned_games=owned_games,
            recently_played=recently_played,
            friends=friends,
            taste_profile=taste_profile,
        )
        
        # Store in context for next agents
        context.user_profile = profile.model_dump(mode="json")
        context.taste_profile = taste_profile.model_dump(mode="json") if taste_profile else None
        
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data=profile.model_dump(mode="json"),
            context_for_next={
                "taste_profile": context.taste_profile,
                "game_count": len(owned_games),
            },
        )
    
    def _parse_owned_games(self, result: Any) -> list[OwnedGame]:
        """Parse owned games from API response."""
        if not result or not result.get("games"):
            return []
        
        return [
            OwnedGame(
                app_id=g.get("appid"),
                name=g.get("name", "Unknown"),
                playtime_forever=g.get("playtime_forever", 0),
                playtime_2weeks=g.get("playtime_2weeks", 0),
            )
            for g in result["games"]
        ]
    
    def _parse_friends(self, result: Any) -> list[str]:
        """Parse friends list from API response."""
        if not result or not result.get("friends"):
            return []
        
        return [f.get("steamid") for f in result["friends"] if f.get("steamid")]
    
    async def _extract_taste_profile(self, games: list[OwnedGame]) -> TasteProfile:
        """
        Extract taste preferences from gaming history.
        
        Analyzes:
        - Genre preferences weighted by playtime
        - Tag preferences weighted by playtime
        - Playtime patterns
        - Behavioral signals (multiplayer, indie, etc.)
        """
        if not games:
            return TasteProfile()
        
        # Calculate total playtime for weighting
        total_playtime = sum(g.playtime_forever for g in games)
        if total_playtime == 0:
            total_playtime = 1  # Avoid division by zero
        
        # In a real implementation, we would:
        # 1. Fetch genre/tag data for each game
        # 2. Weight by playtime
        # 3. Normalize scores
        
        # Placeholder taste profile
        taste_profile = TasteProfile(
            preferred_genres={},  # Would be populated from game metadata
            preferred_tags={},
            avg_playtime_per_game=total_playtime / len(games) if games else 0,
            engagement_level=min(1.0, total_playtime / (1000 * 60)),  # Normalize to 0-1
        )
        
        return taste_profile
