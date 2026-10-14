"""
Database Manager - Smart Caching System for Matchmaking.

This module provides:
1. SQLite-based user storage with optimized game data
2. Filtered upsert logic (no zero-hour games, top 50 only)
3. Efficient retrieval for matchmaking queries
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# Database configuration
DB_PATH = Path(__file__).parent / "matchmaking_cache.db"
CACHE_EXPIRY_HOURS = 24 * 7  # 7 days
MAX_GAMES_TO_STORE = 50
MIN_PLAYTIME_MINUTES = 1  # Minimum playtime to keep (filter zero-hour games)


class DatabaseManager:
    """
    Manages the local SQLite database for matchmaking cache.
    
    Features:
    - Optimized storage (top 50 games only, no zero-playtime)
    - Automatic schema creation
    - Upsert logic with timestamp tracking
    - Efficient batch operations
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_schema()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table - stores cached player profiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    steam_id TEXT PRIMARY KEY,
                    personaname TEXT,
                    avatar_url TEXT,
                    profile_url TEXT,
                    steam_level INTEGER DEFAULT 0,
                    game_count INTEGER DEFAULT 0,
                    total_playtime_hours REAL DEFAULT 0,
                    game_data_json TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    discovery_source TEXT DEFAULT 'fof'
                )
            """)
            
            # Index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_users_last_updated 
                ON users(last_updated DESC)
            """)
            
            # Matchmaking history table (optional, for analytics)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_steam_id TEXT,
                    matched_steam_id TEXT,
                    compatibility_score INTEGER,
                    shared_games_count INTEGER,
                    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_steam_id) REFERENCES users(steam_id),
                    FOREIGN KEY (matched_steam_id) REFERENCES users(steam_id)
                )
            """)
            
            conn.commit()
            logger.info(f"✅ Database schema ensured at {self.db_path}")
    
    # =========================================================================
    # OPTIMIZED UPSERT LOGIC
    # =========================================================================
    
    def optimize_game_data(self, games: list[dict]) -> list[dict]:
        """
        Optimize game data for storage:
        1. Filter out zero-playtime games
        2. Keep only top 50 by playtime
        3. Strip unnecessary fields
        
        Args:
            games: Raw game list from Steam API
            
        Returns:
            Optimized game list (max 50 items)
        """
        if not games:
            return []
        
        # Filter 1: Remove zero-hour games
        filtered = [
            g for g in games 
            if g.get("playtime_forever", 0) >= MIN_PLAYTIME_MINUTES
        ]
        
        # Sort by playtime (descending)
        filtered.sort(key=lambda x: x.get("playtime_forever", 0), reverse=True)
        
        # Filter 2: Keep only top N games
        top_games = filtered[:MAX_GAMES_TO_STORE]
        
        # Strip to essential fields only (save space)
        optimized = []
        for g in top_games:
            optimized.append({
                "appid": g.get("appid"),
                "name": g.get("name", "Unknown"),
                "playtime": g.get("playtime_forever", 0),  # in minutes
            })
        
        logger.debug(f"Optimized games: {len(games)} → {len(optimized)}")
        return optimized
    
    def upsert_user(
        self,
        steam_id: str,
        personaname: str,
        avatar_url: str = "",
        profile_url: str = "",
        steam_level: int = 0,
        games: list[dict] = None,
        discovery_source: str = "fof",
    ) -> bool:
        """
        Insert or update a user with optimized game data.
        
        Args:
            steam_id: Steam ID (primary key)
            personaname: Display name
            avatar_url: Avatar image URL
            profile_url: Steam profile URL
            steam_level: Steam level
            games: Raw game list from API (will be optimized)
            discovery_source: How user was discovered (fof, direct, etc.)
            
        Returns:
            True if successful
        """
        try:
            # Optimize game data before storage
            optimized_games = self.optimize_game_data(games or [])
            game_data_json = json.dumps(optimized_games, ensure_ascii=False)
            
            # Calculate stats
            game_count = len(optimized_games)
            total_playtime = sum(g.get("playtime", 0) for g in optimized_games) / 60  # hours
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (
                        steam_id, personaname, avatar_url, profile_url,
                        steam_level, game_count, total_playtime_hours,
                        game_data_json, last_updated, discovery_source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(steam_id) DO UPDATE SET
                        personaname = excluded.personaname,
                        avatar_url = excluded.avatar_url,
                        profile_url = excluded.profile_url,
                        steam_level = excluded.steam_level,
                        game_count = excluded.game_count,
                        total_playtime_hours = excluded.total_playtime_hours,
                        game_data_json = excluded.game_data_json,
                        last_updated = excluded.last_updated,
                        discovery_source = excluded.discovery_source
                """, (
                    steam_id, personaname, avatar_url, profile_url,
                    steam_level, game_count, total_playtime,
                    game_data_json, datetime.utcnow().isoformat(),
                    discovery_source,
                ))
                conn.commit()
            
            logger.debug(f"Upserted user {steam_id} ({personaname}) with {game_count} games")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting user {steam_id}: {e}")
            return False
    
    def upsert_users_batch(self, users: list[dict]) -> int:
        """
        Batch upsert multiple users efficiently.
        
        Args:
            users: List of user dicts with keys:
                   steam_id, personaname, avatar_url, games, etc.
                   
        Returns:
            Number of successfully upserted users
        """
        success_count = 0
        
        for user in users:
            if self.upsert_user(
                steam_id=user.get("steam_id", ""),
                personaname=user.get("personaname", user.get("name", "Unknown")),
                avatar_url=user.get("avatar_url", user.get("avatar", "")),
                profile_url=user.get("profile_url", ""),
                steam_level=user.get("steam_level", user.get("level", 0)),
                games=user.get("games", []),
                discovery_source=user.get("discovery_source", "fof"),
            ):
                success_count += 1
        
        logger.info(f"Batch upserted {success_count}/{len(users)} users")
        return success_count
    
    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================
    
    def get_user(self, steam_id: str) -> Optional[dict]:
        """Get a single user by Steam ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM users WHERE steam_id = ?
            """, (steam_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
        return None
    
    def get_recent_users(
        self,
        limit: int = 50,
        exclude_steam_id: str = None,
        max_age_hours: int = CACHE_EXPIRY_HOURS,
    ) -> list[dict]:
        """
        Get recently cached users for matchmaking.
        
        Args:
            limit: Maximum number of users to return
            exclude_steam_id: Exclude this user (the requesting user)
            max_age_hours: Only return users updated within this many hours
            
        Returns:
            List of user dicts with parsed game data
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if exclude_steam_id:
                cursor.execute("""
                    SELECT * FROM users 
                    WHERE steam_id != ? 
                    AND last_updated > ?
                    AND game_count > 0
                    ORDER BY last_updated DESC
                    LIMIT ?
                """, (exclude_steam_id, cutoff.isoformat(), limit))
            else:
                cursor.execute("""
                    SELECT * FROM users 
                    WHERE last_updated > ?
                    AND game_count > 0
                    ORDER BY last_updated DESC
                    LIMIT ?
                """, (cutoff.isoformat(), limit))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_user_count(self, max_age_hours: int = CACHE_EXPIRY_HOURS) -> int:
        """Get count of cached users within the age limit."""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM users 
                WHERE last_updated > ? AND game_count > 0
            """, (cutoff.isoformat(),))
            return cursor.fetchone()[0]
    
    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a database row to a dictionary with parsed JSON."""
        d = dict(row)
        
        # Parse game_data_json
        if d.get("game_data_json"):
            try:
                d["games"] = json.loads(d["game_data_json"])
            except json.JSONDecodeError:
                d["games"] = []
        else:
            d["games"] = []
        
        return d
    
    # =========================================================================
    # MATCH HISTORY (Optional Analytics)
    # =========================================================================
    
    def record_match(
        self,
        user_steam_id: str,
        matched_steam_id: str,
        compatibility_score: int,
        shared_games_count: int,
    ):
        """Record a match for analytics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO match_history (
                        user_steam_id, matched_steam_id,
                        compatibility_score, shared_games_count
                    ) VALUES (?, ?, ?, ?)
                """, (user_steam_id, matched_steam_id, compatibility_score, shared_games_count))
                conn.commit()
        except Exception as e:
            logger.debug(f"Could not record match: {e}")
    
    # =========================================================================
    # MAINTENANCE
    # =========================================================================
    
    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove entries older than the specified days."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM users WHERE last_updated < ?
            """, (cutoff.isoformat(),))
            deleted = cursor.rowcount
            conn.commit()
        
        logger.info(f"Cleaned up {deleted} old user entries")
        return deleted
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM match_history")
            total_matches = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT AVG(game_count), AVG(total_playtime_hours) FROM users
            """)
            row = cursor.fetchone()
            avg_games = row[0] or 0
            avg_playtime = row[1] or 0
            
            # Database file size
            db_size_kb = self.db_path.stat().st_size / 1024 if self.db_path.exists() else 0
        
        return {
            "total_users": total_users,
            "total_matches": total_matches,
            "avg_games_per_user": round(avg_games, 1),
            "avg_playtime_hours": round(avg_playtime, 1),
            "db_size_kb": round(db_size_kb, 2),
        }


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
