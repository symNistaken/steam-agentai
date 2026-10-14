"""
Deep Learning Embeddings Module - Using SentenceTransformer

This module handles creating embeddings for games and player profiles
using pre-trained deep learning models. These embeddings are then used
for semantic similarity search and RAG retrieval.

Architecture:
- Uses sentence-transformers (based on BERT/RoBERTa)
- Creates dense vector representations of game descriptions
- Enables semantic similarity matching beyond keyword matching
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:
    def cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        a_norm[a_norm == 0] = 1.0
        b_norm[b_norm == 0] = 1.0
        return (a / a_norm) @ (b / b_norm).T

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Deep Learning Embedding Model using SentenceTransformer.
    
    Attributes:
        model_name: Pre-trained model identifier
        model: Loaded SentenceTransformer model
        embedding_dim: Dimension of output embeddings
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
                - "all-MiniLM-L6-v2": Fast, 384-dim (recommended for this project)
                - "all-mpnet-base-v2": More accurate, 768-dim
                - "paraphrase-multilingual-MiniLM-L12-v2": Turkish support
        """
        try:
            self.model_name = model_name
            if SentenceTransformer is None:
                self.model = None
                self.embedding_dim = 384
                logger.warning("sentence-transformers is not installed; using local hashing embeddings fallback")
            else:
                logger.info(f"Loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✓ Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _embed_text_fallback(self, text: str) -> np.ndarray:
        """Create a deterministic lightweight embedding when transformers are unavailable."""
        import hashlib
        import re

        vector = np.zeros(self.embedding_dim, dtype=np.float32)
        tokens = re.findall(r"[\w']+", (text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.embedding_dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        return vector
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text to embedding vector.
        
        Args:
            text: Text to embed (game description, review, etc.)
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        try:
            if self.model is None:
                return self._embed_text_fallback(text)
            embedding = self.model.encode(text, convert_to_tensor=False)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to embeddings (batch processing).
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of shape (len(texts), embedding_dim)
        """
        try:
            if self.model is None:
                return np.array([self._embed_text_fallback(text) for text in texts], dtype=np.float32)
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def find_similar_texts(
        self, 
        query_text: str, 
        candidates: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query (semantic search).
        
        Args:
            query_text: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (text, similarity_score) tuples, sorted by score descending
        """
        query_embedding = self.embed_text(query_text)
        candidate_embeddings = self.embed_texts(candidates)
        
        similarities = cosine_similarity([query_embedding], candidate_embeddings)[0]
        
        # Get top_k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        return results


class GameEmbeddings:
    """
    Game-specific embeddings handler.
    
    Combines game metadata (title, genres, tags, description) into
    rich semantic representations for matching with player preferences.
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """
        Initialize game embeddings handler.
        
        Args:
            embedding_model: Pre-initialized EmbeddingModel, or None to create default
        """
        self.model = embedding_model or EmbeddingModel()
    
    def create_game_text_representation(
        self, 
        game: Dict[str, Any]
    ) -> str:
        """
        Create a rich text representation of a game for embedding.
        
        Args:
            game: Game dict with keys: name, genres, tags, description, etc.
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Game title (highest weight conceptually)
        if "name" in game:
            parts.append(f"Game: {game['name']}")
        
        # Genres
        if "genres" in game and game["genres"]:
            genres_text = ", ".join(game["genres"])
            parts.append(f"Genres: {genres_text}")
        
        # Tags (user-provided classifications)
        if "tags" in game and game["tags"]:
            tags_text = ", ".join(game["tags"][:10])  # Top 10 tags
            parts.append(f"Tags: {tags_text}")
        
        # Short description
        if "description" in game and game["description"]:
            parts.append(f"Description: {game['description'][:500]}")
        
        # Gameplay style
        if "gameplay_style" in game and game["gameplay_style"]:
            parts.append(f"Gameplay: {game['gameplay_style']}")
        
        return " ".join(parts)
    
    def embed_game(self, game: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding for a single game.
        
        Args:
            game: Game dictionary
            
        Returns:
            Embedding vector
        """
        text = self.create_game_text_representation(game)
        return self.model.embed_text(text)
    
    def embed_games_batch(self, games: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for multiple games.
        
        Args:
            games: List of game dictionaries
            
        Returns:
            Dict mapping game names/IDs to their embeddings
        """
        texts = [self.create_game_text_representation(game) for game in games]
        embeddings = self.model.embed_texts(texts)
        
        result = {}
        for game, embedding in zip(games, embeddings):
            game_id = game.get("app_id") or game.get("appid") or game.get("name")
            result[game_id] = embedding
        
        return result


class PlayerProfileEmbeddings:
    """
    Player profile embeddings - represents player's taste and preferences.
    
    Combines player's favorite games, playtime distribution, and interests
    into a semantic representation for matching with candidate games.
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize player profile embeddings handler."""
        self.model = embedding_model or EmbeddingModel()
        self.game_embeddings = GameEmbeddings(embedding_model=self.model)
    
    def create_player_profile_text(
        self, 
        player_profile: Dict[str, Any]
    ) -> str:
        """
        Create text representation of player profile.
        
        Args:
            player_profile: Profile dict with player preferences
            
        Returns:
            Combined text describing player's taste
        """
        parts = []
        
        # Favorite genres
        if "favorite_genres" in player_profile:
            genres = ", ".join(player_profile["favorite_genres"])
            parts.append(f"Favorite genres: {genres}")
        
        # Favorite games
        if "top_games" in player_profile:
            top_games = ", ".join(
                [game.get("name", "Unknown") for game in player_profile["top_games"][:5]]
            )
            parts.append(f"Top games: {top_games}")
        
        # Playstyle preferences
        if "playstyle" in player_profile:
            parts.append(f"Playstyle: {player_profile['playstyle']}")
        
        # Game tags preference
        if "preferred_tags" in player_profile:
            tags = ", ".join(player_profile["preferred_tags"][:10])
            parts.append(f"Preferred tags: {tags}")
        
        # Difficulty preference
        if "difficulty_preference" in player_profile:
            parts.append(f"Difficulty: {player_profile['difficulty_preference']}")
        
        return " ".join(parts)
    
    def embed_player_profile(self, player_profile: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding for player profile.
        
        Args:
            player_profile: Player profile dictionary
            
        Returns:
            Embedding vector representing player's taste
        """
        text = self.create_player_profile_text(player_profile)
        return self.model.embed_text(text)
    
    def calculate_player_game_similarity(
        self, 
        player_profile: Dict[str, Any], 
        game: Dict[str, Any]
    ) -> float:
        """
        Calculate semantic similarity between player and game.
        
        Args:
            player_profile: Player profile
            game: Game to match
            
        Returns:
            Similarity score (0-1)
        """
        player_embedding = self.embed_player_profile(player_profile)
        game_embedding = self.game_embeddings.embed_game(game)
        
        similarity = cosine_similarity([player_embedding], [game_embedding])[0][0]
        return float(similarity)


# Singleton instance for global use
_embedding_model_instance = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Get or create singleton embedding model instance.
    
    Args:
        model_name: Model to use (only used on first call)
        
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel(model_name)
    return _embedding_model_instance
