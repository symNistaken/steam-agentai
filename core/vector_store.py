"""
Vector Database Module - Using FAISS for Fast Similarity Search

This module provides a vector database using Facebook's FAISS library
for efficient similarity search across game embeddings. This enables
fast retrieval of similar games for RAG context.

Architecture:
- Stores game embeddings in FAISS index
- Provides O(log n) search instead of O(n)
- Supports both in-memory and persistent storage
"""

import logging
import pickle
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    faiss = None
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for storing and searching game embeddings.
    
    Uses FAISS for efficient nearest-neighbor search across
    game embeddings and player profile vectors.
    """
    
    def __init__(self, embedding_dim: int = 384, use_gpu: bool = False):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (default for all-MiniLM-L6-v2)
            use_gpu: Whether to use GPU (requires faiss-gpu)
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self._embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_game: Dict[int, Dict[str, Any]] = {}
        
        self._initialize_index()
        logger.info(f"✓ Vector store initialized (dim={embedding_dim}, GPU={use_gpu})")
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index for similarity search."""
        if FAISS_AVAILABLE:
            # Use Flat L2 index (measures Euclidean distance)
            # For cosine similarity, we normalize embeddings before adding
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = None
            self._embeddings = np.empty((0, self.embedding_dim), dtype=np.float32)
            logger.warning("FAISS is not installed; using slower NumPy vector search fallback")

    @staticmethod
    def _normalize_l2(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine-like L2 search."""
        normalized = embeddings.astype(np.float32).copy()
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return normalized / norms

    @property
    def total_vectors(self) -> int:
        """Return number of vectors stored regardless of backend."""
        if FAISS_AVAILABLE and self.index is not None:
            return int(self.index.ntotal)
        return 0 if self._embeddings is None else int(self._embeddings.shape[0])
    
    def add_games(self, games: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Add games with their embeddings to the vector store.
        
        Args:
            games: List of game dictionaries
            embeddings: Numpy array of shape (n_games, embedding_dim)
        """
        if embeddings.shape[0] != len(games):
            raise ValueError("Mismatch between number of games and embeddings")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.astype(np.float32).copy()
        if FAISS_AVAILABLE:
            faiss.normalize_L2(embeddings_normalized)
        else:
            embeddings_normalized = self._normalize_l2(embeddings_normalized)
        
        # Add to index
        start_id = self.total_vectors
        if FAISS_AVAILABLE:
            self.index.add(embeddings_normalized)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings_normalized])
        
        # Store metadata
        for i, game in enumerate(games):
            game_id = start_id + i
            self.id_to_game[game_id] = game
            self.metadata.append({
                "game_id": game_id,
                "app_id": game.get("app_id") or game.get("appid"),
                "name": game.get("name"),
            })
        
        logger.info(f"Added {len(games)} games to vector store. Total: {self.total_vectors}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar games using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (game_dict, similarity_score) tuples
        """
        if self.total_vectors == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Normalize query embedding for cosine similarity
        query_normalized = query_embedding.astype(np.float32).copy().reshape(1, -1)
        if FAISS_AVAILABLE:
            faiss.normalize_L2(query_normalized)
        else:
            query_normalized = self._normalize_l2(query_normalized)
        
        if FAISS_AVAILABLE:
            distances, indices = self.index.search(query_normalized, min(top_k, self.total_vectors))
            distance_idx_pairs = list(zip(distances[0], indices[0]))
        else:
            distances = np.sum((self._embeddings - query_normalized) ** 2, axis=1)
            indices = np.argsort(distances)[:min(top_k, self.total_vectors)]
            distance_idx_pairs = [(distances[idx], idx) for idx in indices]
        
        results = []
        for distance, idx in distance_idx_pairs:
            if idx == -1:  # Invalid result
                continue
            
            game = self.id_to_game.get(idx)
            # Convert L2 distance back to similarity (1 / (1 + distance))
            similarity = 1.0 / (1.0 + float(distance))
            results.append((game, similarity))
        
        return results
    
    def batch_search(
        self, 
        query_embeddings: np.ndarray, 
        top_k: int = 10
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Array of shape (n_queries, embedding_dim)
            top_k: Number of results per query
            
        Returns:
            List of search results, one per query
        """
        query_normalized = query_embeddings.astype(np.float32).copy()
        if FAISS_AVAILABLE:
            faiss.normalize_L2(query_normalized)
        else:
            query_normalized = self._normalize_l2(query_normalized)
        
        if FAISS_AVAILABLE:
            distances, indices = self.index.search(query_normalized, min(top_k, self.total_vectors))
        else:
            all_distances = ((query_normalized[:, None, :] - self._embeddings[None, :, :]) ** 2).sum(axis=2)
            indices = np.argsort(all_distances, axis=1)[:, :min(top_k, self.total_vectors)]
            distances = np.take_along_axis(all_distances, indices, axis=1)
        
        results = []
        for dist_row, idx_row in zip(distances, indices):
            query_results = []
            for distance, idx in zip(dist_row, idx_row):
                if idx == -1:
                    continue
                game = self.id_to_game.get(idx)
                similarity = 1.0 / (1.0 + float(distance))
                query_results.append((game, similarity))
            results.append(query_results)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_games": self.total_vectors,
            "embedding_dim": self.embedding_dim,
            "memory_usage_mb": self.total_vectors * self.embedding_dim * 4 / (1024 * 1024),
        }
    
    def save(self, path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Path to save to (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, str(path) + ".index")
        else:
            np.save(str(path) + ".npy", self._embeddings)
        
        # Save metadata
        with open(str(path) + ".pkl", "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_to_game": self.id_to_game,
            }, f)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Path to load from (without extension)
        """
        path = Path(path)
        
        index_path = path.parent / (path.name + ".index")
        numpy_path = path.parent / (path.name + ".npy")
        if not index_path.exists() and not numpy_path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        if index_path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(path) + ".index")
        else:
            self.index = None
            self._embeddings = np.load(str(path) + ".npy")
        
        # Load metadata
        with open(str(path) + ".pkl", "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.id_to_game = data["id_to_game"]
        
        logger.info(f"Vector store loaded from {path}. Games: {self.total_vectors}")


class GameVectorIndex:
    """
    High-level interface for managing game vectors and retrieving similar games.
    
    Combines VectorStore with embeddings for easy game similarity retrieval.
    """
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize game vector index.
        
        Args:
            vector_store: Pre-initialized VectorStore or None to create default
        """
        from core.embeddings import EmbeddingModel
        
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        from core.embeddings import get_embedding_model
        self.embedding_model = get_embedding_model()
        self.is_initialized = False
    
    def build_index_from_games(self, games: List[Dict[str, Any]]) -> None:
        """
        Build vector index from list of games.
        
        Args:
            games: List of game dictionaries to index
        """
        logger.info(f"Building vector index for {len(games)} games...")
        
        from core.embeddings import GameEmbeddings
        
        # Create embeddings
        game_embeddings = GameEmbeddings(embedding_model=self.embedding_model)
        embeddings_dict = game_embeddings.embed_games_batch(games)
        
        # Convert to array
        embeddings_list = [embeddings_dict[game.get("app_id") or game.get("appid") or game.get("name")] for game in games]
        embeddings = np.array(embeddings_list)
        
        # Add to vector store
        self.vector_store.add_games(games, embeddings)
        self.is_initialized = True
        
        logger.info(f"✓ Vector index built. Games: {len(games)}")
    
    def find_similar_games(
        self, 
        game: Dict[str, Any], 
        top_k: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar games to a given game.
        
        Args:
            game: Game to find similar games for
            top_k: Number of similar games to return
            
        Returns:
            List of (game_dict, similarity_score) tuples
        """
        from core.embeddings import GameEmbeddings
        
        game_embeddings = GameEmbeddings(embedding_model=self.embedding_model)
        embedding = game_embeddings.embed_game(game)
        
        return self.vector_store.search(embedding, top_k=top_k)
    
    def find_games_for_player_profile(
        self, 
        player_profile: Dict[str, Any], 
        top_k: int = 20
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find best matching games for a player profile.
        
        Args:
            player_profile: Player profile with preferences
            top_k: Number of games to return
            
        Returns:
            List of (game_dict, similarity_score) tuples
        """
        from core.embeddings import PlayerProfileEmbeddings
        
        profile_embeddings = PlayerProfileEmbeddings(embedding_model=self.embedding_model)
        embedding = profile_embeddings.embed_player_profile(player_profile)
        
        return self.vector_store.search(embedding, top_k=top_k)


# Singleton instance
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    """Get or create singleton vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
