"""
Simple Storage Layer for the Agentic AI Architecture.

Provides basic persistence for user profiles and matchmaking data.
Can be extended to use Redis, PostgreSQL, or other backends.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from abc import ABC, abstractmethod

from schemas.models import UserProfile, MatchmakingResult, TasteProfile


logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT STORAGE INTERFACE
# ============================================================================

class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve an item by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store an item with optional TTL in seconds."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an item by key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass


# ============================================================================
# FILE-BASED STORAGE (Simple implementation)
# ============================================================================

class FileStorage(BaseStorage):
    """
    Simple file-based JSON storage.
    
    Suitable for development and small-scale deployments.
    For production, consider Redis or a proper database.
    """
    
    def __init__(self, storage_dir: str = "./data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        # Sanitize key for filesystem
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.storage_dir / f"{safe_key}.json"
    
    async def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve an item from file storage."""
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check TTL
            if "ttl_expires" in data and data["ttl_expires"]:
                if datetime.fromisoformat(data["ttl_expires"]) < datetime.utcnow():
                    await self.delete(key)
                    return None
            
            self._cache[key] = data.get("value", data)
            return self._cache[key]
            
        except Exception as e:
            logger.error(f"Error reading {key}: {e}")
            return None
    
    async def set(self, key: str, value: dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Store an item to file storage."""
        file_path = self._get_file_path(key)
        
        try:
            data = {
                "key": key,
                "value": value,
                "created_at": datetime.utcnow().isoformat(),
                "ttl_expires": None,
            }
            
            if ttl:
                from datetime import timedelta
                expires = datetime.utcnow() + timedelta(seconds=ttl)
                data["ttl_expires"] = expires.isoformat()
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            
            self._cache[key] = value
            return True
            
        except Exception as e:
            logger.error(f"Error writing {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete an item from file storage."""
        file_path = self._get_file_path(key)
        
        try:
            if file_path.exists():
                file_path.unlink()
            self._cache.pop(key, None)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in file storage."""
        if key in self._cache:
            return True
        return self._get_file_path(key).exists()


# ============================================================================
# PROFILE STORAGE
# ============================================================================

class ProfileStorage:
    """
    Storage manager for user profiles.
    
    Handles caching and persistence of UserProfile objects.
    """
    
    def __init__(self, storage: BaseStorage = None):
        self.storage = storage or FileStorage("./data/profiles")
        self._prefix = "profile:"
    
    def _make_key(self, steam_id: str) -> str:
        """Create storage key from steam ID."""
        return f"{self._prefix}{steam_id}"
    
    async def get_profile(self, steam_id: str) -> Optional[UserProfile]:
        """Retrieve a user profile by Steam ID."""
        data = await self.storage.get(self._make_key(steam_id))
        if data:
            return UserProfile(**data)
        return None
    
    async def save_profile(self, profile: UserProfile, ttl: int = 3600) -> bool:
        """
        Save a user profile.
        
        Args:
            profile: The UserProfile to save
            ttl: Time-to-live in seconds (default 1 hour)
        """
        profile.last_updated = datetime.utcnow()
        return await self.storage.set(
            self._make_key(profile.steam_id),
            profile.model_dump(mode="json"),
            ttl=ttl,
        )
    
    async def delete_profile(self, steam_id: str) -> bool:
        """Delete a user profile."""
        return await self.storage.delete(self._make_key(steam_id))
    
    async def profile_exists(self, steam_id: str) -> bool:
        """Check if a profile exists."""
        return await self.storage.exists(self._make_key(steam_id))
    
    async def update_taste_profile(
        self,
        steam_id: str,
        taste_profile: TasteProfile,
    ) -> bool:
        """Update just the taste profile portion."""
        profile = await self.get_profile(steam_id)
        if profile:
            profile.taste_profile = taste_profile
            return await self.save_profile(profile)
        return False


# ============================================================================
# MATCH STORAGE
# ============================================================================

class MatchStorage:
    """
    Storage manager for matchmaking results.
    
    Handles caching of matchmaking computations.
    """
    
    def __init__(self, storage: BaseStorage = None):
        self.storage = storage or FileStorage("./data/matches")
        self._prefix = "match:"
    
    def _make_key(self, user_id: str) -> str:
        """Create storage key from user ID."""
        return f"{self._prefix}{user_id}"
    
    async def get_matches(self, user_id: str) -> Optional[MatchmakingResult]:
        """Retrieve matchmaking results for a user."""
        data = await self.storage.get(self._make_key(user_id))
        if data:
            return MatchmakingResult(**data)
        return None
    
    async def save_matches(
        self,
        result: MatchmakingResult,
        ttl: int = 1800,
    ) -> bool:
        """
        Save matchmaking results.
        
        Args:
            result: The MatchmakingResult to save
            ttl: Time-to-live in seconds (default 30 minutes)
        """
        return await self.storage.set(
            self._make_key(result.user_id),
            result.model_dump(mode="json"),
            ttl=ttl,
        )
    
    async def delete_matches(self, user_id: str) -> bool:
        """Delete matchmaking results."""
        return await self.storage.delete(self._make_key(user_id))


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """
    Central manager for all storage operations.
    
    Provides a unified interface to all storage backends.
    """
    
    _instance: Optional["DatabaseManager"] = None
    
    def __new__(cls) -> "DatabaseManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.profiles = ProfileStorage()
        self.matches = MatchStorage()
        self._initialized = True
        logger.info("DatabaseManager initialized")
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of storage backends."""
        return {
            "profiles": True,  # File storage is always available
            "matches": True,
        }
    
    async def clear_all(self) -> None:
        """Clear all stored data (use with caution!)."""
        import shutil
        
        for storage in [self.profiles.storage, self.matches.storage]:
            if isinstance(storage, FileStorage):
                if storage.storage_dir.exists():
                    shutil.rmtree(storage.storage_dir)
                storage.storage_dir.mkdir(parents=True, exist_ok=True)
                storage._cache.clear()
        
        logger.warning("All data cleared from storage")
