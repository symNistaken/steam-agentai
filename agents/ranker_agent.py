"""
RankerAgent - Responsible for scoring and ranking candidates.

This agent:
1. Scores each candidate on fit, novelty, and constraints
2. Applies user preferences (price, owned exclusion)
3. Generates explanations for recommendations
4. ENHANCED: Uses RAG context for improved scoring
5. ENHANCED: Integrates deep learning semantic similarity

FEATURES:
- Fit Score: How well game matches user taste + RAG context
- Novelty Score: How fresh/different the recommendation is
- Constraint Penalty: Deductions for price, ownership, etc.
- RAG Boost: Additional score from retrieval similarity
"""

from typing import Any
import logging

from agents.base import BaseAgent, AgentContext
from schemas.models import (
    AgentType,
    AgentPlan,
    AgentResponse,
    CandidateGame,
    Game,
)
from services.rag_service import RAGService
from core.explainability import get_recommendation_explainer

logger = logging.getLogger(__name__)


class RankerAgent(BaseAgent):
    """
    Agent responsible for ranking candidate games.
    
    Scoring dimensions:
    1. Fit Score: How well game matches user taste
    2. Novelty Score: How fresh/different the recommendation is
    3. Constraint Penalty: Deductions for price, ownership, etc.
    """
    
    def __init__(
        self,
        fit_weight: float = 0.4,
        novelty_weight: float = 0.25,
        retrieval_weight: float = 0.25,
        rag_weight: float = 0.1,
        enable_rag: bool = True,
    ):
        super().__init__()
        self.fit_weight = fit_weight
        self.novelty_weight = novelty_weight
        self.retrieval_weight = retrieval_weight
        self.rag_weight = rag_weight
        self.enable_rag = enable_rag
        self.rag_service = RAGService() if enable_rag else None
        self.explainer = get_recommendation_explainer()
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.RANKER
    
    async def plan(self, context: AgentContext, objective: str) -> AgentPlan:
        """Create execution plan for ranking."""
        plan = self._create_base_plan(objective)
        
        candidate_count = len(context.candidate_games)
        
        steps = [
            f"Load {candidate_count} candidates from context",
            "Calculate fit scores based on taste profile",
            "Calculate novelty scores based on user's library",
        ]
        
        # Add RAG steps if enabled
        if self.enable_rag:
            steps.append("Retrieve RAG context for semantic boost")
            steps.append("Calculate RAG similarity scores")
        
        steps.extend([
            "Apply constraint penalties (price, owned, etc.)",
            "Compute weighted final scores",
            "Generate XAI explanations for top recommendations",
            "Sort and return ranked results",
        ])
        
        plan.steps = steps
        plan.dependencies = [AgentType.RETRIEVAL]
        
        return plan
    
    async def execute(self, context: AgentContext, plan: AgentPlan) -> AgentResponse:
        """Execute ranking pipeline with RAG and XAI."""
        # Step 1: Load candidates
        candidates = [
            CandidateGame(**c) for c in context.candidate_games
        ]
        
        if not candidates:
            return AgentResponse(
                agent_type=self.agent_type,
                success=True,
                data={"ranked_count": 0},
                metadata={"warning": "No candidates to rank"},
            )
        
        # Get user context
        taste_profile = context.taste_profile or {}
        user_profile = context.user_profile or {}
        owned_app_ids = set(
            g.get("app_id") for g in user_profile.get("owned_games", [])
        )
        
        # Step 2-5: Score each candidate
        ranked_candidates = []
        for candidate in candidates:
            scored = await self._score_candidate(
                candidate=candidate,
                taste_profile=taste_profile,
                owned_app_ids=owned_app_ids,
            )
            ranked_candidates.append(scored)
        
        # Step 6: Generate XAI explanations for top candidates
        ranked_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"Generating XAI explanations for top {min(10, len(ranked_candidates))} candidates")
        for candidate in ranked_candidates[:10]:
            explanation_dict = self.explainer.explain_recommendation(
                game={
                    "name": candidate.game.name,
                    "genres": candidate.game.genres,
                    "tags": candidate.game.tags,
                    "rating": candidate.final_score,
                },
                player_profile=user_profile,
                recommendation_score=candidate.final_score,
                language="tr"  # Turkish explanations
            )
            candidate.explanation = explanation_dict.get("explanation", "")
            candidate.confidence = explanation_dict.get("confidence", "MEDIUM")
        
        # Step 7: Store in context
        context.ranked_games = [c.model_dump(mode="json") for c in ranked_candidates]
        
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data={
                "ranked_count": len(ranked_candidates),
                "top_score": ranked_candidates[0].final_score if ranked_candidates else 0,
                "rag_enabled": self.enable_rag,
            },
        )
        
        # Step 7: Store in context
        context.ranked_games = [c.model_dump(mode="json") for c in ranked_candidates]
        
        return AgentResponse(
            agent_type=self.agent_type,
            success=True,
            data={
                "ranked_count": len(ranked_candidates),
                "top_score": ranked_candidates[0].final_score if ranked_candidates else 0,
            },
        )
    
    async def _score_candidate(
        self,
        candidate: CandidateGame,
        taste_profile: dict[str, Any],
        owned_app_ids: set[int],
    ) -> CandidateGame:
        """Score a single candidate game with RAG boost."""
        
        # Fit score: genre/tag overlap with preferences
        candidate.fit_score = self._calculate_fit_score(
            candidate.game, taste_profile
        )
        
        # Novelty score: how different from owned games
        candidate.novelty_score = self._calculate_novelty_score(
            candidate.game, taste_profile
        )
        
        # Constraint penalty
        candidate.constraint_penalty = self._calculate_constraint_penalty(
            candidate.game, owned_app_ids
        )
        
        # RAG boost: semantic similarity bonus
        rag_boost = 0.0
        if self.enable_rag:
            # Get RAG similarity if available
            rag_boost = getattr(candidate, "rag_similarity", 0.0)
            logger.debug(f"RAG boost for {candidate.game.name}: {rag_boost:.3f}")
        
        candidate.rag_boost = rag_boost
        
        # Compute final score with RAG
        candidate.compute_final_score(
            fit_weight=self.fit_weight,
            novelty_weight=self.novelty_weight,
            retrieval_weight=self.retrieval_weight,
        )
        
        # Apply RAG boost to final score
        if self.enable_rag and rag_boost > 0:
            candidate.final_score = (candidate.final_score * (1 - self.rag_weight)) + (rag_boost * self.rag_weight)
            candidate.final_score = min(1.0, candidate.final_score)
        
        return candidate
    
    def _calculate_fit_score(
        self,
        game: Game,
        taste_profile: dict[str, Any],
    ) -> float:
        """Calculate how well game fits user's taste."""
        if not taste_profile:
            return 0.5  # Neutral score without profile
        
        preferred_genres = taste_profile.get("preferred_genres", {})
        preferred_tags = taste_profile.get("preferred_tags", {})
        
        # Genre overlap
        genre_score = 0.0
        for genre in game.genres:
            if genre.lower() in [g.lower() for g in preferred_genres]:
                genre_score += preferred_genres.get(genre, 0.5)
        genre_score = min(1.0, genre_score)
        
        # Tag overlap
        tag_score = 0.0
        for tag in game.tags:
            if tag.lower() in [t.lower() for t in preferred_tags]:
                tag_score += preferred_tags.get(tag, 0.3)
        tag_score = min(1.0, tag_score)
        
        # Weighted combination
        return (genre_score * 0.6) + (tag_score * 0.4)
    
    def _calculate_novelty_score(
        self,
        game: Game,
        taste_profile: dict[str, Any],
    ) -> float:
        """Calculate how novel/fresh this recommendation is."""
        if not taste_profile:
            return 0.7  # Default novelty
        
        preferred_genres = set(taste_profile.get("preferred_genres", {}).keys())
        
        # Games outside comfort zone get higher novelty
        game_genres = set(g.lower() for g in game.genres)
        overlap = len(game_genres & preferred_genres)
        
        if not game_genres:
            return 0.5
        
        # More overlap = less novelty
        novelty = 1.0 - (overlap / len(game_genres))
        
        # But some overlap is good (not too novel)
        # Optimal is around 30-50% overlap
        if novelty > 0.7:
            novelty = 0.7  # Cap extreme novelty
        
        return novelty
    
    def _calculate_constraint_penalty(
        self,
        game: Game,
        owned_app_ids: set[int],
    ) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Already owned
        if game.app_id in owned_app_ids:
            penalty += 1.0  # Full penalty - can't recommend owned games
        
        # Price constraints would go here
        # if game.price and game.price > max_price:
        #     penalty += 0.3
        
        return min(1.0, penalty)
    
    def _generate_explanation(
        self,
        candidate: CandidateGame,
        taste_profile: dict[str, Any],
    ) -> str:
        """Generate human-readable explanation for recommendation."""
        game = candidate.game
        reasons = []
        
        # Genre match
        preferred_genres = list(taste_profile.get("preferred_genres", {}).keys())
        matching_genres = [g for g in game.genres if g in preferred_genres]
        if matching_genres:
            reasons.append(f"matches your interest in {', '.join(matching_genres[:2])}")
        
        # Source-based reason
        if candidate.source.value == "trending":
            reasons.append("currently popular on Steam")
        elif candidate.source.value == "content_based":
            reasons.append("similar to games you've enjoyed")
        
        # Novelty angle
        if candidate.novelty_score > 0.5:
            reasons.append("offers something new to explore")
        
        # Build explanation
        if reasons:
            return f"Recommended because it {' and '.join(reasons)}."
        
        return "Recommended based on your gaming profile."
