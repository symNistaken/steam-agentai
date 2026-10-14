"""
RAG (Retrieval Augmented Generation) Service

This module implements RAG for game recommendations by:
1. Retrieving relevant games/context from vector store
2. Building rich LLM prompts with retrieved data
3. Generating recommendations with context awareness

This significantly improves recommendation quality by providing
the LLM with specific, relevant examples and information.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from core.vector_store import GameVectorIndex, VectorStore
from core.embeddings import (
    EmbeddingModel,
    GameEmbeddings,
    PlayerProfileEmbeddings,
)

logger = logging.getLogger(__name__)


def get_game_id(game: Dict[str, Any]) -> Any:
    """Return a game id across Steam/RAG naming conventions."""
    return game.get("app_id") or game.get("appid") or game.get("name")


class RAGRetriever:
    """
    Retrieves relevant games and context for augmented LLM generation.
    
    Strategies:
    1. Content-based: Find similar games to player's favorites
    2. Genre-based: Retrieve games matching preferred genres
    3. Trend-based: Include trending games with similarity filter
    4. Hybrid: Combine multiple strategies
    """
    
    def __init__(self, vector_index: Optional[GameVectorIndex] = None):
        """
        Initialize RAG retriever.
        
        Args:
            vector_index: Pre-initialized GameVectorIndex or None to create
        """
        from core.vector_store import get_vector_store
        from core.embeddings import get_embedding_model
        
        self.vector_store = vector_index.vector_store if vector_index is not None else get_vector_store()
        self.vector_index = vector_index
        self.embedding_model = vector_index.embedding_model if vector_index is not None else get_embedding_model()
    
    def retrieve_similar_to_favorites(
        self,
        player_profile: Dict[str, Any],
        top_k: int = 15,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Retrieve games similar to player's favorite games.
        
        Args:
            player_profile: Player profile with top_games
            top_k: Maximum games to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (game_dict, similarity_score, reason) tuples
        """
        results = []
        
        if not player_profile.get("top_games"):
            return results
        
        game_embeddings = GameEmbeddings(embedding_model=self.embedding_model)
        
        # For each top game, find similar games
        for favorite_game in player_profile["top_games"][:5]:  # Top 5 favorites
            embedding = game_embeddings.embed_game(favorite_game)
            similar_games = self.vector_store.search(embedding, top_k=5)
            
            for game, similarity in similar_games:
                if similarity >= similarity_threshold and get_game_id(game) != get_game_id(favorite_game):
                    reason = f"Similar to '{favorite_game.get('name')}'"
                    results.append((game, similarity, reason))
        
        # Remove duplicates, keep highest similarity
        seen = set()
        unique_results = []
        for game, sim, reason in sorted(results, key=lambda x: x[1], reverse=True):
            game_id = get_game_id(game)
            if game_id not in seen:
                seen.add(game_id)
                unique_results.append((game, sim, reason))
        
        return unique_results[:top_k]
    
    def retrieve_by_genres(
        self,
        player_profile: Dict[str, Any],
        all_games: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Retrieve games matching player's preferred genres.
        
        Args:
            player_profile: Player profile with favorite_genres
            all_games: List of all available games
            top_k: Maximum games to retrieve
            
        Returns:
            List of (game_dict, relevance_score, reason) tuples
        """
        results = []
        favorite_genres = set(player_profile.get("favorite_genres", []))
        
        if not favorite_genres:
            return results
        
        for game in all_games:
            game_genres = set(game.get("genres", []))
            intersection = favorite_genres & game_genres
            
            if intersection:
                # Relevance = number of matching genres / total genres in preference
                relevance = len(intersection) / len(favorite_genres)
                matching_genres = ", ".join(intersection)
                reason = f"Matches genres: {matching_genres}"
                results.append((game, relevance, reason))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def retrieve_trending_filtered(
        self,
        player_profile: Dict[str, Any],
        trending_games: List[Dict[str, Any]],
        top_k: int = 10,
        similarity_threshold: float = 0.4
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Retrieve trending games that match player preferences.
        
        Args:
            player_profile: Player profile
            trending_games: List of currently trending games
            top_k: Maximum games to retrieve
            similarity_threshold: Minimum similarity to player profile
            
        Returns:
            List of (game_dict, combined_score, reason) tuples
        """
        results = []
        profile_embeddings = PlayerProfileEmbeddings(embedding_model=self.embedding_model)
        player_embedding = profile_embeddings.embed_player_profile(player_profile)
        
        game_embeddings = GameEmbeddings(embedding_model=self.embedding_model)
        
        for game in trending_games:
            game_embedding = game_embeddings.embed_game(game)
            
            # Calculate similarity
            player_norm = np.linalg.norm(player_embedding) or 1.0
            game_norm = np.linalg.norm(game_embedding) or 1.0
            similarity = float(np.dot(player_embedding, game_embedding) / (player_norm * game_norm))
            
            if similarity >= similarity_threshold:
                # Combined score: 70% similarity + 30% popularity
                popularity_score = game.get("rating", 0.5)  # Assume 0-1 rating
                combined_score = 0.7 * similarity + 0.3 * popularity_score
                
                reason = "Trending and matches your profile"
                results.append((game, combined_score, reason))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
    
    def retrieve_hybrid(
        self,
        player_profile: Dict[str, Any],
        all_games: List[Dict[str, Any]],
        trending_games: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 25
    ) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Hybrid retrieval combining multiple strategies.
        
        Args:
            player_profile: Player profile
            all_games: All available games
            trending_games: Optional list of trending games
            top_k: Maximum games to retrieve
            
        Returns:
            Ranked list of (game_dict, score, reason) tuples
        """
        all_results = []
        
        # Strategy 1: Similar to favorites (40%)
        similar = self.retrieve_similar_to_favorites(player_profile, top_k=15)
        all_results.extend([(g, s * 0.4, r) for g, s, r in similar])
        
        # Strategy 2: Genre matching (30%)
        genre_matches = self.retrieve_by_genres(player_profile, all_games, top_k=15)
        all_results.extend([(g, s * 0.3, r) for g, s, r in genre_matches])
        
        # Strategy 3: Trending (30%)
        if trending_games:
            trending = self.retrieve_trending_filtered(player_profile, trending_games, top_k=10)
            all_results.extend([(g, s * 0.3, r) for g, s, r in trending])
        
        # Aggregate and rank
        from collections import defaultdict
        game_scores = defaultdict(list)
        game_reasons = defaultdict(list)
        
        for game, score, reason in all_results:
            game_id = get_game_id(game)
            game_scores[game_id].append((game, score))
            game_reasons[game_id].append(reason)
        
        # Average scores and combine reasons
        final_results = []
        for game_id, scores in game_scores.items():
            game = scores[0][0]  # Get game object
            avg_score = sum(s[1] for s in scores) / len(scores)
            combined_reason = " + ".join(game_reasons[game_id])
            final_results.append((game, avg_score, combined_reason))
        
        return sorted(final_results, key=lambda x: x[1], reverse=True)[:top_k]


class RAGPromptBuilder:
    """
    Builds enhanced LLM prompts with retrieved context.
    
    Creates rich, context-aware prompts that include:
    - Retrieved similar games
    - Player preferences summary
    - Game metadata
    - Clear instruction for the LLM
    """
    
    @staticmethod
    def build_recommendation_prompt(
        player_profile: Dict[str, Any],
        retrieved_games: List[Tuple[Dict[str, Any], float, str]],
        candidate_games: List[Dict[str, Any]],
        language: str = "en"
    ) -> str:
        """
        Build a comprehensive prompt for game recommendations with RAG context.
        
        Args:
            player_profile: Player's profile and preferences
            retrieved_games: Games retrieved by RAG (context examples)
            candidate_games: Games to choose recommendations from
            language: Language for prompt (en/tr)
            
        Returns:
            Formatted prompt string
        """
        if language == "tr":
            return RAGPromptBuilder._build_prompt_turkish(
                player_profile, retrieved_games, candidate_games
            )
        else:
            return RAGPromptBuilder._build_prompt_english(
                player_profile, retrieved_games, candidate_games
            )
    
    @staticmethod
    def _build_prompt_english(
        player_profile: Dict[str, Any],
        retrieved_games: List[Tuple[Dict[str, Any], float, str]],
        candidate_games: List[Dict[str, Any]]
    ) -> str:
        """Build English RAG-enhanced prompt."""
        prompt_parts = []
        
        # Header
        prompt_parts.append("=== STEAM GAME RECOMMENDATION SYSTEM (RAG-Enhanced) ===\n")
        
        # Player Profile Section
        prompt_parts.append("## PLAYER PROFILE:\n")
        prompt_parts.append(f"- Favorite Genres: {', '.join(player_profile.get('favorite_genres', []))}\n")
        prompt_parts.append(f"- Playstyle: {player_profile.get('playstyle', 'Unknown')}\n")
        prompt_parts.append(f"- Total Playtime: {player_profile.get('total_playtime_hours', 0)} hours\n")
        prompt_parts.append(f"- Difficulty Preference: {player_profile.get('difficulty_preference', 'Any')}\n")
        
        if player_profile.get("top_games"):
            prompt_parts.append("\n- Top Games Played:\n")
            for game in player_profile.get("top_games", [])[:5]:
                prompt_parts.append(f"  * {game.get('name', 'Unknown')} ({game.get('playtime_hours', 0)}h)\n")
        
        # Context from Retrieved Games
        prompt_parts.append("\n## SIMILAR GAMES REFERENCE (Retrieved from Database):\n")
        prompt_parts.append("These games are similar to what the player enjoys:\n")
        for i, (game, score, reason) in enumerate(retrieved_games[:5], 1):
            prompt_parts.append(f"{i}. {game.get('name', 'Unknown')}\n")
            prompt_parts.append(f"   - Genres: {', '.join(game.get('genres', []))}\n")
            prompt_parts.append(f"   - Reason: {reason}\n")
            prompt_parts.append(f"   - Relevance Score: {score:.2%}\n")
        
        # Candidate Games
        prompt_parts.append("\n## CANDIDATE GAMES TO RECOMMEND FROM:\n")
        for i, game in enumerate(candidate_games[:20], 1):
            prompt_parts.append(f"{i}. {game.get('name', 'Unknown')}\n")
            prompt_parts.append(f"   - Genres: {', '.join(game.get('genres', []))}\n")
            prompt_parts.append(f"   - Description: {game.get('description', 'N/A')[:100]}\n")
        
        # Instructions
        prompt_parts.append("\n## TASK:\n")
        prompt_parts.append("Based on the player profile and similar games reference:\n")
        prompt_parts.append("1. Select 10-15 games from candidates that best match the player's preferences\n")
        prompt_parts.append("2. For each recommendation, explain WHY it matches the player\n")
        prompt_parts.append("3. Rate confidence for each (HIGH/MEDIUM/LOW)\n")
        prompt_parts.append("4. Return as JSON with: [name, reason, confidence]\n")
        
        return "".join(prompt_parts)
    
    @staticmethod
    def _build_prompt_turkish(
        player_profile: Dict[str, Any],
        retrieved_games: List[Tuple[Dict[str, Any], float, str]],
        candidate_games: List[Dict[str, Any]]
    ) -> str:
        """Build Turkish RAG-enhanced prompt."""
        prompt_parts = []
        
        prompt_parts.append("=== STEAM OYUN ÖNERİ SİSTEMİ (RAG-Geliştirilmiş) ===\n")
        
        # Oyuncu Profili
        prompt_parts.append("## OYUNCU PROFİLİ:\n")
        prompt_parts.append(f"- Favori Türler: {', '.join(player_profile.get('favorite_genres', []))}\n")
        prompt_parts.append(f"- Oyun Stili: {player_profile.get('playstyle', 'Bilinmiyor')}\n")
        prompt_parts.append(f"- Toplam Oynama Süresi: {player_profile.get('total_playtime_hours', 0)} saat\n")
        prompt_parts.append(f"- Zorluk Seviyesi Tercihi: {player_profile.get('difficulty_preference', 'Herhangi')}\n")
        
        if player_profile.get("top_games"):
            prompt_parts.append("\n- En Çok Oynanan Oyunlar:\n")
            for game in player_profile.get("top_games", [])[:5]:
                prompt_parts.append(f"  * {game.get('name', 'Bilinmiyor')} ({game.get('playtime_hours', 0)}s)\n")
        
        # Benzer Oyunlar Referansı
        prompt_parts.append("\n## BENZER OYUNLAR (Veri Tabanından Alınan):\n")
        for i, (game, score, reason) in enumerate(retrieved_games[:5], 1):
            prompt_parts.append(f"{i}. {game.get('name', 'Bilinmiyor')}\n")
            prompt_parts.append(f"   - Türler: {', '.join(game.get('genres', []))}\n")
            prompt_parts.append(f"   - Neden: {reason}\n")
            prompt_parts.append(f"   - İlişki Puanı: {score:.2%}\n")
        
        # Aday Oyunları
        prompt_parts.append("\n## ADAY OYUNLAR:\n")
        for i, game in enumerate(candidate_games[:20], 1):
            prompt_parts.append(f"{i}. {game.get('name', 'Bilinmiyor')}\n")
            prompt_parts.append(f"   - Türler: {', '.join(game.get('genres', []))}\n")
            prompt_parts.append(f"   - Açıklama: {game.get('description', 'N/A')[:100]}\n")
        
        # Talimatlar
        prompt_parts.append("\n## GÖREV:\n")
        prompt_parts.append("Oyuncu profili ve benzer oyunlar referansı temel alınarak:\n")
        prompt_parts.append("1. Aday oyunlardan oyuncunun tercihlerine en uygun 10-15 oyun seçin\n")
        prompt_parts.append("2. Her bir tavsiye için NEDEN oyuncuya uygun olduğunu açıklayın\n")
        prompt_parts.append("3. Her bir tavsiye için güven düzeyi verin (YÜKSEK/ORTA/DÜŞÜK)\n")
        prompt_parts.append("4. JSON formatında döndürün: [oyun_adı, neden, güven]\n")
        
        return "".join(prompt_parts)


class RAGService:
    """
    Main RAG service combining retrieval and generation.
    
    Orchestrates the RAG pipeline for game recommendations.
    """
    
    def __init__(self, vector_index: Optional[GameVectorIndex] = None):
        """Initialize RAG service."""
        self.retriever = RAGRetriever(vector_index=vector_index)
        self.prompt_builder = RAGPromptBuilder()
    
    def generate_rag_context(
        self,
        player_profile: Dict[str, Any],
        all_games: List[Dict[str, Any]],
        candidate_games: List[Dict[str, Any]],
        trending_games: Optional[List[Dict[str, Any]]] = None,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate RAG context and enhanced prompt for LLM.
        
        Args:
            player_profile: Player profile
            all_games: All available games (for retrieval)
            candidate_games: Games to recommend from
            trending_games: Optional trending games
            language: Prompt language
            
        Returns:
            Dict with retrieved_games, prompt, and metadata
        """
        logger.info("🔍 Starting RAG context generation...")
        
        # Retrieve relevant games
        retrieved_games = self.retriever.retrieve_hybrid(
            player_profile=player_profile,
            all_games=all_games,
            trending_games=trending_games,
            top_k=15
        )
        
        logger.info(f"Retrieved {len(retrieved_games)} context games")
        
        # Build enhanced prompt
        prompt = self.prompt_builder.build_recommendation_prompt(
            player_profile=player_profile,
            retrieved_games=retrieved_games,
            candidate_games=candidate_games,
            language=language
        )
        
        return {
            "retrieved_games": retrieved_games,
            "prompt": prompt,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "language": language,
                "retrieved_count": len(retrieved_games),
                "candidate_count": len(candidate_games),
            }
        }
