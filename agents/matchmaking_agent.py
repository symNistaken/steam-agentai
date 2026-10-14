"""
MatchmakingAgent - Database-Backed Friend-of-Friend Discovery.

This agent implements a smart caching matchmaking system:
1. Checks local database first for cached users
2. Falls back to FoF API discovery if cache is low
3. Stores discovered users with optimized game data
4. Returns randomized top matches for variety
"""

import asyncio
import random
import logging
from typing import Any, Optional

import aiohttp

from agents.base import BaseAgent, AgentContext
from schemas.models import (
    AgentType,
    AgentPlan,
    AgentResponse,
    PlayerMatch,
    MatchmakingResult,
)
from config import STEAM_API_KEY, STEAM_API_BASE_URL
from database.db_manager import get_db_manager, DatabaseManager


logger = logging.getLogger(__name__)


class MatchmakingAgent(BaseAgent):
    """
    Agent responsible for finding compatible players using hybrid strategy.
    
    Hybrid Strategy:
    1. Check local DB for cached users (fast path)
    2. If cache is low (<20 users), trigger FoF API discovery
    3. Store all discovered users with optimized data
    4. Score and return randomized matches
    """
    
    # Configuration constants
    MIN_CACHED_USERS = 20          # Minimum users to use cache-only mode
    MAX_PROFILES_TO_ANALYZE = 200  # Max profiles to analyze (increased from 50)
    FRIENDS_TO_SAMPLE = 5          # Random friends for FoF expansion
    TOP_CANDIDATES = 20            # Top scored candidates (increased from 10)
    MATCHES_TO_RETURN = 3          # Final matches to display
    CACHE_MAX_AGE_HOURS = 168      # 7 days
    
    def __init__(self, max_matches: int = 3):
        super().__init__()
        self.max_matches = max_matches
        self.api_key = STEAM_API_KEY
        self.base_url = STEAM_API_BASE_URL or "https://api.steampowered.com"
        self.db: DatabaseManager = get_db_manager()
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.MATCHMAKING
    
    async def plan(self, context: AgentContext, objective: str) -> AgentPlan:
        """Create execution plan for matchmaking."""
        plan = self._create_base_plan(objective)
        
        # Check cache status
        cached_count = self.db.get_user_count(max_age_hours=self.CACHE_MAX_AGE_HOURS)
        
        plan.steps = [
            f"Check local cache ({cached_count} users available)",
            f"Use cache if >= {self.MIN_CACHED_USERS} users, else trigger FoF discovery",
            "Filter and score candidates by game overlap",
            "Store new discoveries with optimized data (top 50 games, no zero-hour)",
            f"Return top {self.MATCHES_TO_RETURN} randomized matches",
        ]
        
        plan.dependencies = [AgentType.PROFILE]
        plan.metadata = {"cached_users": cached_count}
        
        return plan
    
    async def execute(self, context: AgentContext, plan: AgentPlan) -> AgentResponse:
        """Execute the hybrid matchmaking pipeline."""
        steam_id = context.steam_id
        
        if not steam_id:
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data={"error": "Steam ID required"},
            )
        
        try:
            # Step 1: Get user's games for comparison
            user_games = await self._fetch_user_games(steam_id)
            if not user_games:
                return AgentResponse(
                    agent_type=self.agent_type,
                    success=True,
                    data={
                        "match_count": 0,
                        "matches": [],
                        "total_analyzed": 0,
                        "cache_used": False,
                        "message": "Kullanıcının oyun kütüphanesi gizli veya boş",
                    },
                )
            
            user_game_ids = set(g.get("appid") for g in user_games if g.get("appid"))
            
            # Step 2: Check cache status
            cached_count = self.db.get_user_count(max_age_hours=self.CACHE_MAX_AGE_HOURS)
            logger.info(f"📦 Cache status: {cached_count} users available")
            
            candidates = []
            cache_used = False
            api_discovered = 0
            
            # Step 3: Decide strategy
            if cached_count >= self.MIN_CACHED_USERS:
                # Fast path: Use all cached users
                logger.info(f"⚡ Using all {cached_count} cached users (fast path)")
                cached_users = self.db.get_recent_users(
                    limit=10000,  # Use all cached users (no practical limit)
                    exclude_steam_id=steam_id,
                    max_age_hours=self.CACHE_MAX_AGE_HOURS,
                )
                candidates = cached_users
                cache_used = True
            else:
                # Slow path: FoF API discovery
                logger.info(f"🔍 Cache low, triggering FoF discovery...")
                api_candidates = await self._discover_via_fof(steam_id)
                api_discovered = len(api_candidates)
                
                # Also include cached users if any
                if cached_count > 0:
                    cached_users = self.db.get_recent_users(
                        limit=20,
                        exclude_steam_id=steam_id,
                    )
                    candidates = api_candidates + cached_users
                else:
                    candidates = api_candidates
            
            logger.info(f"📊 Analyzing {len(candidates)} candidates...")
            
            # Step 4: Score all candidates
            matches = self._score_candidates(
                candidates=candidates,
                user_game_ids=user_game_ids,
                user_games=user_games,
            )
            
            # Step 5: Sort and randomize
            matches.sort(key=lambda x: x["compatibility_score"], reverse=True)
            top_matches = matches[:self.TOP_CANDIDATES]
            
            if len(top_matches) > self.MATCHES_TO_RETURN:
                final_matches = random.sample(top_matches, self.MATCHES_TO_RETURN)
            else:
                final_matches = top_matches
            
            # Re-sort for display
            final_matches.sort(key=lambda x: x["compatibility_score"], reverse=True)
            
            # Record matches in history
            for match in final_matches:
                self.db.record_match(
                    user_steam_id=steam_id,
                    matched_steam_id=match["steam_id"],
                    compatibility_score=match["compatibility_score"],
                    shared_games_count=match["shared_games"],
                )
            
            # Get DB stats
            db_stats = self.db.get_stats()
            
            logger.info(f"✅ Matchmaking complete: {len(final_matches)} matches")
            
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data={
                    "match_count": len(final_matches),
                    "matches": final_matches,
                    "total_analyzed": len(candidates),
                    "cache_used": cache_used,
                    "api_discovered": api_discovered,
                    "db_stats": db_stats,
                    "message": "Eşleşmeler bulundu!" if final_matches else "Uygun eşleşme bulunamadı",
                },
            )
            
        except Exception as e:
            logger.error(f"❌ Matchmaking error: {e}")
            import traceback
            traceback.print_exc()
            return AgentResponse(
                agent_type=self.agent_type,
                success=False,
                data={"error": str(e)},
            )
    
    # =========================================================================
    # FOF DISCOVERY (API-based)
    # =========================================================================
    
    async def _discover_via_fof(self, steam_id: str) -> list[dict]:
        """
        Discover new users via Friend-of-Friend strategy.
        Results are stored in the database with optimized data.
        """
        # Step 1: Get direct friends
        direct_friends = await self._fetch_friends(steam_id)
        
        if not direct_friends:
            logger.warning("No friends found for FoF discovery")
            return []
        
        logger.info(f"✓ Found {len(direct_friends)} direct friends")
        
        # Step 2: Sample random friends for expansion
        sampled_friends = random.sample(
            direct_friends,
            min(self.FRIENDS_TO_SAMPLE, len(direct_friends))
        )
        
        # Step 3: Get 2nd degree connections
        all_candidates = set(direct_friends)
        
        fof_tasks = [self._fetch_friends(fid) for fid in sampled_friends]
        fof_results = await asyncio.gather(*fof_tasks, return_exceptions=True)
        
        for result in fof_results:
            if isinstance(result, list):
                all_candidates.update(result)
        
        all_candidates.discard(steam_id)
        
        # Step 4: Limit and shuffle
        candidate_list = list(all_candidates)
        if len(candidate_list) > self.MAX_PROFILES_TO_ANALYZE:
            candidate_list = random.sample(candidate_list, self.MAX_PROFILES_TO_ANALYZE)
        
        logger.info(f"🔄 Fetching data for {len(candidate_list)} FoF candidates...")
        
        # Step 5: Fetch full profiles and store
        discovered_users = await self._fetch_and_store_profiles(candidate_list)
        
        return discovered_users
    
    async def _fetch_and_store_profiles(self, steam_ids: list[str]) -> list[dict]:
        """
        Fetch full profiles for candidates and store in database.
        Returns list of user dicts for immediate scoring.
        """
        if not steam_ids:
            return []
        
        # Fetch summaries in bulk
        summaries = await self._fetch_player_summaries(steam_ids)
        
        users_to_store = []
        valid_users = []
        
        # Process in batches
        batch_size = 10
        for i in range(0, len(steam_ids), batch_size):
            batch = steam_ids[i:i + batch_size]
            
            tasks = [
                self._fetch_user_profile(sid, summaries.get(sid, {}))
                for sid in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result.get("games"):
                    users_to_store.append(result)
                    valid_users.append(result)
        
        # Batch store in database
        if users_to_store:
            stored = self.db.upsert_users_batch(users_to_store)
            logger.info(f"💾 Stored {stored} new users in cache")
        
        return valid_users
    
    async def _fetch_user_profile(self, steam_id: str, summary: dict) -> Optional[dict]:
        """Fetch complete profile for a single user."""
        if not summary:
            return None
        
        # Fetch games
        games = await self._fetch_user_games(steam_id)
        if not games:
            return None
        
        # Fetch level
        level = await self._fetch_player_level(steam_id)
        
        return {
            "steam_id": steam_id,
            "personaname": summary.get("personaname", "Unknown"),
            "avatar_url": summary.get("avatarfull", ""),
            "profile_url": summary.get("profileurl", ""),
            "steam_level": level,
            "games": games,
            "discovery_source": "fof",
        }
    
    # =========================================================================
    # SCORING (Works with both cached and fresh data)
    # =========================================================================
    
    def _score_candidates(
        self,
        candidates: list[dict],
        user_game_ids: set[int],
        user_games: list[dict],
    ) -> list[dict]:
        """
        Score all candidates for compatibility.
        Works with both cached users (from DB) and fresh API data.
        """
        matches = []
        
        for candidate in candidates:
            # Get games (handle both cached and fresh format)
            candidate_games = candidate.get("games", [])
            if not candidate_games:
                continue
            
            # Extract game IDs (handle both formats)
            candidate_game_ids = set()
            for g in candidate_games:
                appid = g.get("appid") or g.get("app_id")
                if appid:
                    candidate_game_ids.add(appid)
            
            if not candidate_game_ids:
                continue
            
            # Calculate shared games
            shared_game_ids = user_game_ids & candidate_game_ids
            
            if not shared_game_ids:
                continue
            
            # Calculate compatibility score
            score = self._calculate_compatibility(
                user_game_ids=user_game_ids,
                user_games=user_games,
                candidate_game_ids=candidate_game_ids,
                candidate_games=candidate_games,
                shared_game_ids=shared_game_ids,
            )
            
            # Get top shared game
            top_shared_game = self._get_top_shared_game(user_games, shared_game_ids)
            
            # Detect genres
            common_genres = self._detect_common_genres(user_games, shared_game_ids)
            
            matches.append({
                "steam_id": candidate.get("steam_id", ""),
                "name": candidate.get("personaname", candidate.get("name", "Unknown")),
                "avatar": candidate.get("avatar_url", candidate.get("avatar", "")),
                "profile_url": candidate.get("profile_url", ""),
                "level": candidate.get("steam_level", candidate.get("level", 0)),
                "compatibility_score": score,
                "shared_games": len(shared_game_ids),
                "common_genres": common_genres[:3],
                "top_game": top_shared_game,
            })
        
        return matches
    
    def _calculate_compatibility(
        self,
        user_game_ids: set[int],
        user_games: list[dict],
        candidate_game_ids: set[int],
        candidate_games: list[dict],
        shared_game_ids: set[int],
    ) -> int:
        """
        Calculate social compatibility score (0-100).
        
        Factors:
        - Minimum 5 shared games (requirement)
        - Genre match (30 points)
        - Playtime tier match (25 points)
        - Level proximity (20 points)
        - Shared game ratio (25 points)
        """
        
        # REQUIREMENT: Minimum 5 shared games
        if len(shared_game_ids) < 5:
            return 0  # No compatibility without minimum shared games
        
        score = 0
        max_score = 100
        
        # 1. GENRE MATCH (30 points)
        # ============================
        user_genres = self._get_genres_for_games(user_games, shared_game_ids)
        candidate_genres = self._get_genres_for_games(candidate_games, shared_game_ids)
        
        genre_match = len(user_genres & candidate_genres) / max(len(user_genres | candidate_genres), 1)
        genre_score = int(genre_match * 30)
        score += genre_score
        
        # 2. PLAYTIME TIER MATCH (25 points)
        # ==================================
        user_tier = self._get_playtime_tier(user_games)
        candidate_tier = self._get_playtime_tier(candidate_games)
        
        if user_tier == candidate_tier:
            tier_score = 25  # Perfect match
        elif abs(user_tier - candidate_tier) == 1:
            tier_score = 15  # Adjacent tiers
        else:
            tier_score = 5   # Different tiers
        
        score += tier_score
        
        # 3. LEVEL PROXIMITY (20 points)
        # ==============================
        # Extract level from games (average level based on playtime)
        user_exp = self._estimate_experience_level(user_games)
        candidate_exp = self._estimate_experience_level(candidate_games)
        
        level_diff = abs(user_exp - candidate_exp)
        if level_diff <= 1:
            level_score = 20  # Very close
        elif level_diff <= 2:
            level_score = 15  # Close
        elif level_diff <= 3:
            level_score = 10  # Somewhat close
        else:
            level_score = 5   # Different experience
        
        score += level_score
        
        # 4. SHARED GAME RATIO (25 points)
        # ================================
        # How many % of their smaller library is shared
        min_library_size = min(len(user_game_ids), len(candidate_game_ids))
        shared_ratio = len(shared_game_ids) / max(min_library_size, 1)
        
        shared_score = int(shared_ratio * 25)
        score += shared_score
        
        # Bonus: If they share 10+ games, add extra points
        if len(shared_game_ids) >= 10:
            score += 10
        
        return min(score, 100)
    
    def _get_playtime_tier(self, games: list[dict]) -> int:
        """
        Determine playtime tier (0-3):
        0 = Casual (< 100 hours total)
        1 = Regular (100-500 hours)
        2 = Hardcore (500-2000 hours)
        3 = Ultra (2000+ hours)
        """
        total_playtime = sum(g.get("playtime_forever", 0) for g in games) / 60  # to hours
        
        if total_playtime < 100:
            return 0
        elif total_playtime < 500:
            return 1
        elif total_playtime < 2000:
            return 2
        else:
            return 3
    
    def _estimate_experience_level(self, games: list[dict]) -> float:
        """
        Estimate experience level (0-5) based on:
        - Total playtime
        - Game count
        - Average playtime per game
        """
        if not games:
            return 0
        
        total_playtime = sum(g.get("playtime_forever", 0) for g in games) / 60  # hours
        game_count = len(games)
        avg_playtime = total_playtime / max(game_count, 1)
        
        # Score: consider total playtime, variety, and depth
        experience = min(5, (total_playtime / 1000) + (game_count / 200) + (avg_playtime / 100))
        return experience
    
    def _get_genres_for_games(self, games: list[dict], game_ids: set[int]) -> set[str]:
        """
        Get genres for specific games from a library.
        """
        genres = set()
        
        genre_keywords = {
            "FPS": ["counter-strike", "csgo", "cs2", "call of duty", "battlefield", "apex", "pubg", "valorant", "halo"],
            "RPG": ["witcher", "skyrim", "fallout", "dark souls", "elden ring", "baldur", "divinity", "final fantasy"],
            "Strategy": ["civilization", "stellaris", "crusader kings", "hearts of iron", "total war", "starcraft"],
            "Survival": ["rust", "dayz", "ark", "subnautica", "valheim", "forest", "minecraft"],
            "Action": ["gta", "just cause", "devil may cry", "dmc", "bayonetta"],
            "Racing": ["forza", "need for speed", "assetto", "f1 ", "gran turismo"],
            "Horror": ["resident evil", "dead by daylight", "phasmophobia", "outlast", "amnesia"],
            "Co-op": ["left 4 dead", "deep rock", "payday", "it takes two", "don't starve", "valheim"],
            "Multiplayer": ["dota", "rocket league", "among us", "overwatch", "apex"],
            "Casual": ["sims", "stardew", "animal crossing", "cooking mama", "peglin"],
        }
        
        for game in games:
            if (game.get("appid") or game.get("app_id")) not in game_ids:
                continue
            
            game_name = game.get("name", "").lower()
            
            for genre, keywords in genre_keywords.items():
                if any(kw in game_name for kw in keywords):
                    genres.add(genre)
        
        return genres if genres else {"Gaming"}
    
    
    def _get_top_shared_game(self, user_games: list[dict], shared_game_ids: set[int]) -> str:
        """Get name of most played shared game."""
        shared = [
            g for g in user_games
            if (g.get("appid") or g.get("app_id")) in shared_game_ids
        ]
        
        if not shared:
            return "Unknown"
        
        shared.sort(key=lambda x: x.get("playtime_forever", x.get("playtime", 0)), reverse=True)
        return shared[0].get("name", "Unknown")
    
    def _detect_common_genres(self, user_games: list[dict], shared_game_ids: set[int]) -> list[str]:
        """Detect genres from shared games for display."""
        shared = [
            g for g in user_games
            if (g.get("appid") or g.get("app_id")) in shared_game_ids
        ]
        
        if not shared:
            return ["Gaming"]
        
        genres = self._get_genres_for_games(user_games, shared_game_ids)
        return list(genres)[:5]  # Return top 5 genres
    
    # =========================================================================
    # STEAM API HELPERS
    # =========================================================================
    
    async def _fetch_friends(self, steam_id: str) -> list[str]:
        """Fetch friends list."""
        try:
            url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
            params = {"key": self.api_key, "steamid": steam_id, "relationship": "friend"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
            
            friends = data.get("friendslist", {}).get("friends", [])
            return [f["steamid"] for f in friends if "steamid" in f]
        except Exception:
            return []
    
    async def _fetch_user_games(self, steam_id: str) -> list[dict]:
        """Fetch owned games."""
        try:
            url = f"{self.base_url}/IPlayerService/GetOwnedGames/v1/"
            params = {
                "key": self.api_key,
                "steamid": steam_id,
                "include_appinfo": 1,
                "include_played_free_games": 1,
                "format": "json",
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
            
            return data.get("response", {}).get("games", [])
        except Exception:
            return []
    
    async def _fetch_player_summaries(self, steam_ids: list[str]) -> dict[str, dict]:
        """Fetch summaries for multiple users."""
        if not steam_ids:
            return {}
        
        try:
            ids_str = ",".join(steam_ids[:100])
            url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
            params = {"key": self.api_key, "steamids": ids_str}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        return {}
                    data = await response.json()
            
            players = data.get("response", {}).get("players", [])
            return {p["steamid"]: p for p in players}
        except Exception:
            return {}
    
    async def _fetch_player_level(self, steam_id: str) -> int:
        """Fetch Steam level."""
        try:
            url = f"{self.base_url}/IPlayerService/GetSteamLevel/v1/"
            params = {"key": self.api_key, "steamid": steam_id}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status != 200:
                        return 0
                    data = await response.json()
            
            return data.get("response", {}).get("player_level", 0)
        except Exception:
            return 0


# Export
__all__ = ["MatchmakingAgent"]
