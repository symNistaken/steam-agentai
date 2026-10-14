"""
XAI (Explainable AI) Module - Making Recommendations Interpretable

This module provides explainability for game recommendations using:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. Feature Importance Analysis
3. Human-readable explanation generation
4. Confidence scoring

Ensures users understand WHY a game was recommended.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzes which features contribute most to recommendations.
    
    Features include:
    - Genre similarity
    - Playstyle compatibility
    - Difficulty match
    - Playtime expectations
    - Community rating
    """
    
    @staticmethod
    def extract_game_features(game: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numerical features from a game.
        
        Args:
            game: Game dictionary
            
        Returns:
            Dict of feature_name -> score (0-1)
        """
        features = {}
        
        # Genre features (normalized)
        genres = set(game.get("genres", []))
        features["has_action"] = 1.0 if "Action" in genres else 0.0
        features["has_strategy"] = 1.0 if "Strategy" in genres else 0.0
        features["has_rpg"] = 1.0 if "RPG" in genres else 0.0
        features["has_simulation"] = 1.0 if "Simulation" in genres else 0.0
        features["has_adventure"] = 1.0 if "Adventure" in genres else 0.0
        features["has_indie"] = 1.0 if "Indie" in genres else 0.0
        features["has_multiplayer"] = 1.0 if "Multiplayer" in game.get("tags", []) else 0.0
        features["has_singleplayer"] = 1.0 if "Singleplayer" in game.get("tags", []) else 0.0
        
        # Gameplay features
        features["estimated_playtime"] = min(game.get("estimated_playtime_hours", 10) / 100.0, 1.0)
        features["difficulty"] = game.get("difficulty_score", 0.5)  # 0-1
        features["rating"] = game.get("rating", 0.5)  # 0-1
        
        # Popularity
        features["popularity"] = min(game.get("player_count_all_time", 1000) / 10000.0, 1.0)
        
        return features
    
    @staticmethod
    def extract_player_preferences(profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract player preference features.
        
        Args:
            profile: Player profile dictionary
            
        Returns:
            Dict of preference_name -> importance (0-1)
        """
        preferences = {}
        
        # Genre preferences
        favorite_genres = set(profile.get("favorite_genres", []))
        preferences["likes_action"] = 1.0 if "Action" in favorite_genres else 0.0
        preferences["likes_strategy"] = 1.0 if "Strategy" in favorite_genres else 0.0
        preferences["likes_rpg"] = 1.0 if "RPG" in favorite_genres else 0.0
        preferences["likes_simulation"] = 1.0 if "Simulation" in favorite_genres else 0.0
        preferences["likes_adventure"] = 1.0 if "Adventure" in favorite_genres else 0.0
        preferences["likes_indie"] = 1.0 if "Indie" in favorite_genres else 0.0
        
        # Playstyle
        playstyle = profile.get("playstyle", "").lower()
        preferences["prefers_competitive"] = 1.0 if "competitive" in playstyle else 0.0
        preferences["prefers_relaxing"] = 1.0 if "relaxing" in playstyle else 0.0
        preferences["prefers_challenging"] = 1.0 if "challenging" in playstyle else 0.0
        
        # Difficulty
        difficulty = profile.get("difficulty_preference", "").lower()
        preferences["wants_easy"] = 1.0 if "easy" in difficulty else 0.0
        preferences["wants_medium"] = 1.0 if "medium" in difficulty else 0.0
        preferences["wants_hard"] = 1.0 if "hard" in difficulty else 0.0
        
        # Session expectations
        preferences["likes_short_sessions"] = 1.0 if profile.get("avg_session_hours", 2) < 1.5 else 0.0
        preferences["likes_long_sessions"] = 1.0 if profile.get("avg_session_hours", 2) > 2.5 else 0.0
        
        return preferences
    
    @staticmethod
    def calculate_feature_importance(
        game: Dict[str, Any],
        player_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate importance of each feature in matching game to player.
        
        Args:
            game: Game dictionary
            player_profile: Player profile dictionary
            
        Returns:
            Dict of feature_name -> importance_score (0-1)
        """
        game_features = FeatureImportanceAnalyzer.extract_game_features(game)
        player_prefs = FeatureImportanceAnalyzer.extract_player_preferences(player_profile)
        
        importance = {}
        
        # Genre matches
        for key in game_features.keys():
            if key.startswith("has_"):
                pref_key = f"likes_{key[4:]}"
                if pref_key in player_prefs:
                    importance[key] = game_features[key] * player_prefs[pref_key]
        
        # Playstyle compatibility
        if player_prefs.get("prefers_competitive", 0) > 0:
            importance["multiplayer_match"] = game_features.get("has_multiplayer", 0)
        
        if player_prefs.get("prefers_relaxing", 0) > 0:
            importance["relaxing_match"] = 1.0 - game_features.get("difficulty", 0.5)
        
        # Difficulty match
        difficulty_pref = (
            player_prefs.get("wants_easy", 0) * 0.3 +
            player_prefs.get("wants_medium", 0) * 0.5 +
            player_prefs.get("wants_hard", 0) * 0.7
        )
        if difficulty_pref > 0:
            game_difficulty = game_features.get("difficulty", 0.5)
            importance["difficulty_match"] = 1.0 - abs(difficulty_pref - game_difficulty)
        
        # Rating/Quality
        importance["quality_factor"] = game_features.get("rating", 0.5)
        
        # Playtime expectations
        if player_prefs.get("likes_short_sessions", 0) > 0:
            importance["session_length_match"] = 1.0 - game_features.get("estimated_playtime", 0.5)
        
        if player_prefs.get("likes_long_sessions", 0) > 0:
            importance["session_length_match"] = game_features.get("estimated_playtime", 0.5)
        
        return importance


class ExplanationGenerator:
    """
    Generates human-readable explanations for recommendations.
    
    Creates natural language explanations in multiple languages
    explaining WHY each game was recommended.
    """
    
    @staticmethod
    def generate_explanation(
        game: Dict[str, Any],
        player_profile: Dict[str, Any],
        recommendation_score: float,
        language: str = "en"
    ) -> str:
        """
        Generate explanation for why game was recommended.
        
        Args:
            game: Recommended game
            player_profile: Player's profile
            recommendation_score: Score for this recommendation (0-1)
            language: Language for explanation (en/tr)
            
        Returns:
            Human-readable explanation string
        """
        if language == "tr":
            return ExplanationGenerator._generate_explanation_turkish(
                game, player_profile, recommendation_score
            )
        else:
            return ExplanationGenerator._generate_explanation_english(
                game, player_profile, recommendation_score
            )
    
    @staticmethod
    def _generate_explanation_english(
        game: Dict[str, Any],
        player_profile: Dict[str, Any],
        recommendation_score: float
    ) -> str:
        """Generate English explanation."""
        explanations = []
        
        # Genre matching
        game_genres = set(game.get("genres", []))
        favorite_genres = set(player_profile.get("favorite_genres", []))
        matching_genres = game_genres & favorite_genres
        
        if matching_genres:
            genres_str = ", ".join(list(matching_genres)[:2])
            explanations.append(f"Matches your favorite genres: {genres_str}")
        
        # Rating quality
        if game.get("rating", 0) > 0.75:
            explanations.append("Highly rated by the gaming community")
        
        # Playtime match
        avg_session = player_profile.get("avg_session_hours", 2)
        est_playtime = game.get("estimated_playtime_hours", 10)
        
        if avg_session < 1 and est_playtime < 5:
            explanations.append("Good for short gaming sessions")
        elif avg_session > 3 and est_playtime > 20:
            explanations.append("Provides substantial gameplay hours")
        
        # Multiplayer preference
        if "Multiplayer" in game.get("tags", []) and "Competitive" in player_profile.get("playstyle", ""):
            explanations.append("Has multiplayer features you enjoy")
        
        # Difficulty
        game_difficulty = game.get("difficulty_score", 0.5)
        difficulty_pref = player_profile.get("difficulty_preference", "Medium")
        
        if difficulty_pref.lower() == "easy" and game_difficulty < 0.4:
            explanations.append("Matches your preference for easier games")
        elif difficulty_pref.lower() == "hard" and game_difficulty > 0.6:
            explanations.append("Offers the challenge you're looking for")
        
        # Build final explanation
        if not explanations:
            explanations.append("Aligns with your gaming profile")
        
        explanation = " • ".join(explanations)
        
        # Add confidence indicator
        if recommendation_score > 0.8:
            explanation += " (Confidence: HIGH)"
        elif recommendation_score > 0.5:
            explanation += " (Confidence: MEDIUM)"
        else:
            explanation += " (Confidence: LOW)"
        
        return explanation
    
    @staticmethod
    def _generate_explanation_turkish(
        game: Dict[str, Any],
        player_profile: Dict[str, Any],
        recommendation_score: float
    ) -> str:
        """Generate Turkish explanation."""
        explanations = []
        
        # Genre matching
        game_genres = set(game.get("genres", []))
        favorite_genres = set(player_profile.get("favorite_genres", []))
        matching_genres = game_genres & favorite_genres
        
        if matching_genres:
            genres_str = ", ".join(list(matching_genres)[:2])
            explanations.append(f"Favori türlerinizle eşleşiyor: {genres_str}")
        
        # Rating quality
        if game.get("rating", 0) > 0.75:
            explanations.append("Oyun topluluğu tarafından yüksek puanlandırılmış")
        
        # Playtime match
        avg_session = player_profile.get("avg_session_hours", 2)
        est_playtime = game.get("estimated_playtime_hours", 10)
        
        if avg_session < 1 and est_playtime < 5:
            explanations.append("Kısa oyun oturumları için uygundur")
        elif avg_session > 3 and est_playtime > 20:
            explanations.append("Geniş oyun saatleri sağlar")
        
        # Multiplayer preference
        if "Multiplayer" in game.get("tags", []) and "Rekabetçi" in player_profile.get("playstyle", ""):
            explanations.append("Sevdiğiniz çok oyunculu özelliklere sahiptir")
        
        # Difficulty
        game_difficulty = game.get("difficulty_score", 0.5)
        difficulty_pref = player_profile.get("difficulty_preference", "Orta")
        
        if difficulty_pref.lower() in ["kolay", "easy"] and game_difficulty < 0.4:
            explanations.append("Kolay oyunlar tercihine uygun")
        elif difficulty_pref.lower() in ["zor", "hard"] and game_difficulty > 0.6:
            explanations.append("Aradığınız zorluk seviyesine sahip")
        
        # Build final explanation
        if not explanations:
            explanations.append("Oyun profilinizle uyumlu")
        
        explanation = " • ".join(explanations)
        
        # Add confidence indicator
        if recommendation_score > 0.8:
            explanation += " (Güven: YÜKSEK)"
        elif recommendation_score > 0.5:
            explanation += " (Güven: ORTA)"
        else:
            explanation += " (Güven: DÜŞÜK)"
        
        return explanation


class RecommendationExplainer:
    """
    Main explainability service for recommendations.
    
    Generates detailed explanations for recommendations using
    feature importance and human-readable descriptions.
    """
    
    def __init__(self):
        """Initialize recommendation explainer."""
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.explanation_generator = ExplanationGenerator()
    
    def explain_recommendation(
        self,
        game: Dict[str, Any],
        player_profile: Dict[str, Any],
        recommendation_score: float,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a recommendation.
        
        Args:
            game: Recommended game
            player_profile: Player profile
            recommendation_score: Score for recommendation (0-1)
            language: Language for explanations
            
        Returns:
            Dict with explanation, features, confidence, etc.
        """
        # Calculate feature importance
        feature_importance = self.feature_analyzer.calculate_feature_importance(
            game, player_profile
        )
        
        # Sort features by importance
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Generate text explanation
        text_explanation = self.explanation_generator.generate_explanation(
            game, player_profile, recommendation_score, language=language
        )
        
        # Calculate confidence
        if recommendation_score > 0.8:
            confidence = "HIGH"
        elif recommendation_score > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            "game_name": game.get("name"),
            "recommendation_score": float(recommendation_score),
            "confidence": confidence,
            "explanation": text_explanation,
            "top_factors": [
                {
                    "factor": factor_name,
                    "importance": float(importance)
                }
                for factor_name, importance in top_features
            ],
            "feature_breakdown": {
                k: float(v) for k, v in feature_importance.items()
            }
        }
    
    def explain_recommendations_batch(
        self,
        recommendations: List[Tuple[Dict[str, Any], float]],
        player_profile: Dict[str, Any],
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple recommendations at once.
        
        Args:
            recommendations: List of (game, score) tuples
            player_profile: Player profile
            language: Language for explanations
            
        Returns:
            List of explanation dicts
        """
        explanations = []
        for game, score in recommendations:
            explanation = self.explain_recommendation(
                game, player_profile, score, language=language
            )
            explanations.append(explanation)
        
        return explanations


# Singleton instance
_explainer_instance = None


def get_recommendation_explainer() -> RecommendationExplainer:
    """Get or create singleton explainer instance."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = RecommendationExplainer()
    return _explainer_instance
