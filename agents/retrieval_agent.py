"""
RetrievalAgent - Responsible for candidate game retrieval.

This agent:
1. Uses taste profile to find content-based matches
2. Fetches trending/popular games for discovery
3. Combines candidates into a pool for ranking
"""

from typing import Any

from agents.base import BaseAgent, AgentContext
from schemas.models import (
    AgentType,
    AgentPlan,
    AgentResponse,
    CandidateGame,
    Game,
    RetrievalStrategy,
)
from tools.schemas import (
    GetTrendingGamesInput,
    SearchGamesInput,
    VectorSearchInput,
)


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving candidate games.
    
    Strategies:
    1. Content-based: Games similar to user's preferences
    2. Trending: Popular games for discovery
    3. Collaborative: Games liked by similar users (future)
    """
    
    def __init__(self, max_candidates: int = 50):
        super().__init__()
        self.max_candidates = max_candidates
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.RETRIEVAL
    
    async def plan(self, context: AgentContext, objective: str) -> AgentPlan:
        """Create execution plan for candidate retrieval."""
        plan = self._create_base_plan(objective)
        
        has_taste = context.taste_profile is not None
        
        plan.steps = [
            "Analyze taste profile for search criteria" if has_taste else "Use default search criteria",
            "Retrieve content-based candidates using vector search",
            "Retrieve trending games for discovery",
            "Merge and deduplicate candidate pool",
            "Attach retrieval scores and metadata",
        ]
        
        plan.dependencies = [AgentType.PROFILE]
        
        return plan
    
    async def execute(self, context: AgentContext, plan: AgentPlan) -> AgentResponse:
        """Execute candidate retrieval pipeline."""
        candidates: list[CandidateGame] = []
        
        # Step 1: Analyze taste profile
        search_criteria = self._build_search_criteria(context.taste_profile)
        
        # Step 2: Content-based retrieval
        if context.taste_profile:
            content_candidates = await self._retrieve_content_based(
                context.taste_profile,
                plan,
            )
            candidates.extend(content_candidates)
        
        # Step 3: Trending retrieval
        trending_candidates = await self._retrieve_trending(plan)
        candidates.extend(trending_candidates)
        
        # Step 4: Merge and deduplicate
        candidates = self._deduplicate_candidates(candidates)
        
        # Limit to max candidates
        candidates = candidates[:self.max_candidates]
        
        # Step 5: Store in context
        context.candidate_games = [c.model_dump(mode="json") for c in candidates]
        
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data={"candidate_count": len(candidates)},
            context_for_next={
                "candidate_count": len(candidates),
                "strategies_used": list(set(c.source.value for c in candidates)),
            },
        )
    
    def _build_search_criteria(self, taste_profile: dict[str, Any] | None) -> dict[str, Any]:
        """Build search criteria from taste profile."""
        if not taste_profile:
            return {"genres": [], "tags": []}
        
        # Get top genres and tags
        genres = taste_profile.get("preferred_genres", {})
        tags = taste_profile.get("preferred_tags", {})
        
        # Sort by weight and take top N
        top_genres = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "genres": [g[0] for g in top_genres],
            "tags": [t[0] for t in top_tags],
        }
    
    async def _retrieve_content_based(
        self,
        taste_profile: dict[str, Any],
        plan: AgentPlan,
    ) -> list[CandidateGame]:
        """Retrieve candidates using content-based filtering."""
        candidates = []
        
        # Use vector search for semantic similarity
        search_call = await self.invoke_tool(
            "vector_search",
            VectorSearchInput(
                query_text=self._taste_to_query(taste_profile),
                top_k=30,
            ),
        )
        plan.tool_calls.append(search_call)
        
        if search_call.result and search_call.result.get("results"):
            for i, result in enumerate(search_call.result["results"]):
                score = search_call.result.get("scores", [])[i] if i < len(search_call.result.get("scores", [])) else 0.5
                
                game = Game(
                    app_id=result.get("app_id", 0),
                    name=result.get("name", "Unknown"),
                    genres=result.get("genres", []),
                    tags=result.get("tags", []),
                )
                
                candidates.append(CandidateGame(
                    game=game,
                    source=RetrievalStrategy.CONTENT_BASED,
                    retrieval_score=score,
                ))
        
        return candidates
    
    async def _retrieve_trending(self, plan: AgentPlan) -> list[CandidateGame]:
        """Retrieve trending/popular games."""
        candidates = []
        
        # Fetch top sellers
        trending_call = await self.invoke_tool(
            "store_metadata",
            GetTrendingGamesInput(category="top_sellers", max_results=20),
        )
        plan.tool_calls.append(trending_call)
        
        if trending_call.result and trending_call.result.get("games"):
            for i, game_data in enumerate(trending_call.result["games"]):
                game = Game(
                    app_id=game_data.get("app_id", 0),
                    name=game_data.get("name", "Unknown"),
                    genres=game_data.get("genres", []),
                    tags=game_data.get("tags", []),
                )
                
                # Higher score for higher ranked games
                score = 1.0 - (i / 20)
                
                candidates.append(CandidateGame(
                    game=game,
                    source=RetrievalStrategy.TRENDING,
                    retrieval_score=score,
                ))
        
        return candidates
    
    def _taste_to_query(self, taste_profile: dict[str, Any]) -> str:
        """Convert taste profile to a search query string."""
        genres = list(taste_profile.get("preferred_genres", {}).keys())[:3]
        tags = list(taste_profile.get("preferred_tags", {}).keys())[:5]
        
        query_parts = []
        if genres:
            query_parts.append(f"genres: {', '.join(genres)}")
        if tags:
            query_parts.append(f"features: {', '.join(tags)}")
        
        return " | ".join(query_parts) if query_parts else "popular games"
    
    def _deduplicate_candidates(self, candidates: list[CandidateGame]) -> list[CandidateGame]:
        """Remove duplicate games, keeping highest scoring version."""
        seen = {}
        
        for candidate in candidates:
            app_id = candidate.game.app_id
            if app_id not in seen or candidate.retrieval_score > seen[app_id].retrieval_score:
                seen[app_id] = candidate
        
        return list(seen.values())
