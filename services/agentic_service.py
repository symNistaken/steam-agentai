"""
Agentic Recommendation Service.

This service provides the bridge between Flask routes and the
Agentic AI orchestration system. It handles async execution
and converts between the old API format and new agentic responses.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL
from schemas.models import (
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    TasteProfile,
    CandidateGame,
    Game,
    RetrievalStrategy,
)
from orchestrator import Orchestrator, OrchestratorFactory
from agents.base import AgentContext, AgentRegistry
from tools.base import (
    SteamAPITool,
    StoreMetadataTool,
    GetOwnedGamesInput,
    GetPlayerSummaryInput,
    GetGameDetailsInput,
    GetTrendingGamesInput,
)


logger = logging.getLogger(__name__)


class AgenticRecommendationService:
    """
    Service that combines Steam data fetching with LLM-powered analysis
    using the new Agentic AI architecture.
    """
    
    def __init__(self):
        self.steam_tool = SteamAPITool()
        self.store_tool = StoreMetadataTool()
        self.llm_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        logger.info(f"🔑 GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")
        logger.info(f"🔑 LLM Client initialized: {bool(self.llm_client)}")
        
    async def analyze_profile(self, steam_id: str) -> Dict[str, Any]:
        """
        Full agentic analysis pipeline.
        
        1. Fetch Steam data (async)
        2. Build taste profile
        3. Get recommendations from LLM
        4. Enrich with store data
        """
        result = {
            "success": False,
            "player": None,
            "stats": None,
            "library": [],
            "analysis": None,
            "error": None,
        }
        
        try:
            # Step 1: Fetch player data in parallel
            logger.info(f"🔍 Fetching data for Steam ID: {steam_id}")
            
            player_task = self.steam_tool.get_player_summary(
                GetPlayerSummaryInput(steam_id=steam_id)
            )
            games_task = self.steam_tool.get_owned_games(
                GetOwnedGamesInput(
                    steam_id=steam_id,
                    include_appinfo=True,
                    include_played_free_games=True,
                )
            )
            
            player_result, games_result = await asyncio.gather(player_task, games_task)
            
            if not player_result.success:
                result["error"] = "Profil bulunamadı"
                return result
            
            logger.info(f"✓ Oyuncu: {player_result.persona_name}")
            logger.info(f"✓ Oyun sayısı: {games_result.game_count}")
            
            # Check if library is private
            is_library_private = games_result.is_private
            
            # Step 2: Build taste profile
            games = games_result.games
            taste_profile = self._build_taste_profile(games)
            
            # Step 3: Get player level (bonus)
            level = await self._get_player_level(steam_id)
            
            # Step 4: LLM Analysis
            logger.info("🤖 AI analizi başlıyor...")
            analysis = await self._llm_analyze(
                player_name=player_result.persona_name,
                level=level,
                games=games,
                taste_profile=taste_profile,
            )
            logger.info("✓ AI analizi tamamlandı")
            
            # Step 5: Build response
            result["success"] = True
            result["player"] = {
                "steam_id": steam_id,
                "name": player_result.persona_name,
                "avatar": player_result.avatar_url,
                "profile_url": player_result.profile_url,
                "level": level,
            }
            result["stats"] = {
                "total_games": len(games),
                "total_playtime": round(sum(g.get("playtime_forever", 0) for g in games) / 60, 1),
                "avg_playtime": round(
                    sum(g.get("playtime_forever", 0) for g in games) / max(len(games), 1) / 60, 1
                ),
                "is_library_private": is_library_private,  # Add privacy flag
            }
            result["top_games"] = [
                {
                    "name": g.get("name", "Unknown"),
                    "playtime": round(g.get("playtime_forever", 0) / 60, 1),
                    "appid": g.get("appid"),
                }
                for g in games[:5]
            ]
            result["library"] = self._build_library(games[:10])
            result["analysis"] = analysis
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _build_taste_profile(self, games: List[Dict]) -> TasteProfile:
        """Extract taste profile from gaming history."""
        total_playtime = sum(g.get("playtime_forever", 0) for g in games)
        
        # Simple taste extraction (can be enhanced with genre data)
        return TasteProfile(
            avg_playtime_per_game=total_playtime / max(len(games), 1),
            engagement_level=min(1.0, total_playtime / (1000 * 60)),
        )
    
    async def _get_player_level(self, steam_id: str) -> int:
        """Get player Steam level."""
        try:
            import aiohttp
            url = "https://api.steampowered.com/IPlayerService/GetSteamLevel/v1/"
            params = {"key": self.steam_tool.api_key, "steamid": steam_id}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", {}).get("player_level", 0)
        except Exception as e:
            logger.warning(f"Could not fetch level: {e}")
        return 0
    
    async def _llm_analyze(
        self,
        player_name: str,
        level: int,
        games: List[Dict],
        taste_profile: TasteProfile,
    ) -> Dict[str, Any]:
        """Use LLM to analyze player and generate recommendations."""
        
        import sys
        sys.stderr.write(f"\n🔍 _llm_analyze called for {player_name}\n")
        sys.stderr.flush()
        
        print(f"\n🔍 _llm_analyze called for {player_name}", flush=True)
        print(f"🔑 LLM Client: {self.llm_client}", flush=True)
        logger.info(f"🔑 LLM Client check: {self.llm_client is not None}")
        
        if not self.llm_client:
            sys.stderr.write("❌ LLM client is None!\n")
            sys.stderr.flush()
            print("❌ LLM client is None!", flush=True)
            logger.warning("⚠️  LLM client is None, returning fallback analysis")
            return self._fallback_analysis(games)
        
        print("✅ LLM client exists, preparing prompt...", flush=True)
        
        # Prepare top games text
        top_games_text = "\n".join([
            f"• {g.get('name', 'Unknown')}: {g.get('playtime_forever', 0) // 60} saat"
            for g in games[:10]
        ])
        
        prompt = f"""Sen bir Steam oyun uzmanısın. Aşağıdaki Steam oyuncu profilini analiz et ve
SADECE JSON formatında yanıt ver.

⚠️ UYARI: Kullanıcıya oyun önerirken **ASLA** Steam'de olmayan oyunları 
(Valorant, League of Legends, Overwatch, Escape from Tarkov, Minecraft vb.) önerme. 
Sadece Steam mağazasında mevcut olan oyunları öner.

**OYUNCU BİLGİSİ:**
İsim: {player_name}
Seviye: {level}
Toplam Oyun: {len(games)}
Toplam Saat: {sum(g.get('playtime_forever', 0) for g in games) // 60} saat

**EN ÇOK OYNANAN OYUNLAR:**
{top_games_text}

---

ŞU JSON'I DÖNDÜR (başka şey yazma):
{{
    "character": "2-3 cümle oyuncu karakteri analizi (Türkçe)",
    "playstyle": "Oyun stili ve tercihleri (Türkçe)",
    "strengths": "Güçlü yönleri",
    "weaknesses": "Geliştirebileceği yönler",
    "recommendations": [
        {{"name": "Oyun Adı 1", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 2", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 3", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 4", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 5", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 6", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 7", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 8", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 9", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 10", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}}
    ]
}}"""
        
        try:
            print(f"📡 Calling Groq API with model: {LLM_MODEL}")
            response = self.llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=LLM_MODEL,
                temperature=0.7,
                max_tokens=1500,
            )
            print(f"✅ Groq API response received")
            
            text = response.choices[0].message.content
            print(f"📝 Response text length: {len(text)}")
            return self._parse_llm_response(text)
            
        except Exception as e:
            import traceback
            print(f"❌ LLM Exception: {e}")
            logger.error(f"❌ LLM error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._fallback_analysis(games)
    
    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                
                # Process recommendations - add images and store URLs
                recommendations = []
                for rec in data.get("recommendations", []):
                    appid = rec.get("appid", 0)
                    recommendations.append({
                        "name": rec.get("name", ""),
                        "reason": rec.get("reason", ""),
                        "appid": appid,
                        "image": f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg" if appid else "",
                        "store_url": f"https://store.steampowered.com/app/{appid}/" if appid else "#",
                    })
                
                return {
                    "character": data.get("character", "-"),
                    "playstyle": data.get("playstyle", "-"),
                    "strengths": data.get("strengths", "-"),
                    "weaknesses": data.get("weaknesses", "-"),
                    "recommendations": recommendations,
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
        
        return self._fallback_analysis()
    
    def _fallback_analysis(self, games: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        games = games or []
        owned_appids = {g.get("appid") for g in games if g.get("appid")}
        owned_names = {str(g.get("name", "")).strip().lower() for g in games}
        total_hours = sum(g.get("playtime_forever", 0) for g in games) // 60
        top_names = [g.get("name", "Unknown") for g in games[:5]]

        fallback_pool = [
            (1086940, "Baldur's Gate 3", "Derin rol yapma mekanikleri ve uzun soluklu ilerleme seven oyuncular için güçlü bir seçim."),
            (1245620, "ELDEN RING", "Keşif, ustalaşma ve zorlu savaş döngüsünü seven profillere iyi oturur."),
            (1174180, "Red Dead Redemption 2", "Açık dünya, atmosfer ve hikaye ağırlıklı deneyim arayan oyunculara uygun."),
            (1091500, "Cyberpunk 2077", "Aksiyon, karakter geliştirme ve hikaye odaklı açık dünya isteyenler için iyi bir eşleşme."),
            (292030, "The Witcher 3: Wild Hunt", "Uzun oyun süresi, görev çeşitliliği ve RPG tercihleriyle uyumlu."),
            (413150, "Stardew Valley", "Rahat tempo, ilerleme hissi ve uzun vadeli oynanış arayanlar için ideal."),
            (367520, "Hollow Knight", "Keşif, beceri ve atmosferik dünyaları seven oyunculara uygun."),
            (1145360, "Hades", "Kısa ama tekrar oynanabilir aksiyon döngüsü ve güçlü ilerleme sistemi sunar."),
            (892970, "Valheim", "Hayatta kalma, craft ve co-op keşif seven oyuncular için doğal bir öneri."),
            (105600, "Terraria", "Sandbox, keşif ve uzun süreli gelişim döngüsü sevenlere hitap eder."),
            (582010, "Monster Hunter: World", "Ekipman geliştirme, boss savaşları ve co-op aksiyon sevenler için güçlü."),
            (553850, "HELLDIVERS 2", "Takım oyunu, görev bazlı aksiyon ve co-op aksiyon seven profillere uygun."),
            (1282100, "Remnant II", "Co-op, zorlu çatışma ve loot odaklı ilerleme seven oyunculara iyi gider."),
            (221100, "DayZ", "Hayatta kalma, gerilim ve oyuncu etkileşimi arayanlar için uygun."),
            (393380, "Squad", "Taktiksel takım oyunu ve iletişim odaklı FPS sevenlere önerilir."),
        ]

        recommendations = []
        for appid, name, reason in fallback_pool:
            if appid in owned_appids or name.lower() in owned_names:
                continue
            recommendations.append({
                "name": name,
                "reason": reason,
                "appid": appid,
                "image": f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg",
                "store_url": f"https://store.steampowered.com/app/{appid}/",
            })
            if len(recommendations) >= 10:
                break

        if top_names:
            playstyle = (
                f"En çok oynadığın oyunlar ({', '.join(top_names)}) üzerinden uzun soluklu "
                "ve tekrar oynanabilir deneyimlere yatkın görünüyorsun."
            )
        else:
            playstyle = "Profil verisi sınırlı olduğu için genel Steam tercihleri üzerinden öneri üretildi."

        return {
            "character": f"Profil analizi temel Steam verileriyle hazırlandı. Toplam yaklaşık {total_hours} saatlik oyun geçmişi dikkate alındı.",
            "playstyle": playstyle,
            "strengths": "Oyun geçmişinden oynama süresi ve kütüphane yoğunluğu okunabiliyor.",
            "weaknesses": "LLM servisi yanıt vermediği için kişisel yorumlar sınırlı tutuldu.",
            "recommendations": recommendations,
        }
    
    def _build_library(self, games: List[Dict]) -> List[Dict]:
        """Build library response with images."""
        library = []
        for game in games:
            appid = game.get("appid", 0)
            library.append({
                "name": game.get("name", "Unknown"),
                "playtime": round(game.get("playtime_forever", 0) / 60, 1),
                "appid": appid,
                "image": f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg" if appid else "",
                "store_url": f"https://store.steampowered.com/app/{appid}/" if appid else "#",
            })
        return library


# Singleton instance
_service: Optional[AgenticRecommendationService] = None


def get_recommendation_service() -> AgenticRecommendationService:
    """Get or create the recommendation service singleton."""
    global _service
    if _service is None:
        _service = AgenticRecommendationService()
    return _service


def run_async(coro):
    """Helper to run async code from sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
