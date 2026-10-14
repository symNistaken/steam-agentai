"""
Steam Web API Servisi
Oyuncu profili, oyunlar, başarımlar vb. verileri çeken
"""
import requests
from typing import Dict, List, Optional
from config import STEAM_API_BASE_URL, STEAM_API_KEY


class SteamService:
    """Steam Web API ile iletişim"""
    
    @staticmethod
    def resolve_steam_id(input_str: str) -> Optional[str]:
        """
        Steam ID, URL veya Vanity ID'den gerçek Steam ID'yi bul
        Destekler:
        - 17 haneli Steam ID: 76561197960265730
        - URL: https://steamcommunity.com/profiles/76561197960265730
        - Vanity URL: https://steamcommunity.com/id/gabelogannewell
        - Vanity ID: gabelogannewell
        """
        input_str = input_str.strip()
        
        # Direkt Steam ID ise (17 haneli sayı)
        if input_str.isdigit() and len(input_str) == 17:
            return input_str
        
        # URL ise parse et
        if "steamcommunity.com" in input_str:
            if "/profiles/" in input_str:
                # https://steamcommunity.com/profiles/76561197960265730
                steam_id = input_str.split("/profiles/")[-1].split("/")[0]
                return steam_id if steam_id.isdigit() else None
            elif "/id/" in input_str:
                # https://steamcommunity.com/id/vanityid
                vanity_id = input_str.split("/id/")[-1].split("/")[0]
                return SteamService._resolve_vanity(vanity_id)
        
        # Vanity ID olabilir
        if not input_str.isdigit():
            return SteamService._resolve_vanity(input_str)
        
        return None
    
    @staticmethod
    def _resolve_vanity(vanity_id: str) -> Optional[str]:
        """Vanity ID'yi gerçek Steam ID'ye çevir"""
        try:
            url = f"{STEAM_API_BASE_URL}/ISteamUser/ResolveVanityURL/v0001/"
            params = {"key": STEAM_API_KEY, "vanityurl": vanity_id}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if data.get("response", {}).get("success") == 1:
                return str(data["response"]["steamid"])
        except Exception as e:
            print(f"❌ Vanity çözme hatası: {e}")
        
        return None
    
    @staticmethod
    def get_player_summary(steam_id: str) -> Optional[Dict]:
        """Oyuncu özet bilgisini al (isim, avatar, ülke, vb.)"""
        try:
            url = f"{STEAM_API_BASE_URL}/ISteamUser/GetPlayerSummaries/v0002/"
            params = {"key": STEAM_API_KEY, "steamids": steam_id}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            players = data.get("response", {}).get("players", [])
            if players:
                return players[0]
        except Exception as e:
            print(f"❌ Player summary hatası: {e}")
        
        return None
    
    @staticmethod
    def get_owned_games(steam_id: str) -> List[Dict]:
        """
        Oyuncu kütüphanesini al
        PlayTime'a göre sıralı
        """
        try:
            url = f"{STEAM_API_BASE_URL}/IPlayerService/GetOwnedGames/v1/"
            params = {
                "key": STEAM_API_KEY,
                "steamid": steam_id,
                "include_appinfo": True,
                "include_played_free_games": True
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            games = data.get("response", {}).get("games", [])
            # PlayTime'a göre azalan sıra
            games.sort(key=lambda x: x.get("playtime_forever", 0), reverse=True)
            return games
        except Exception as e:
            print(f"❌ Owned games hatası: {e}")
        
        return []
    
    @staticmethod
    def get_player_level(steam_id: str) -> int:
        """Oyuncunun Steam seviyesini al (hata yakalama ile)"""
        try:
            url = f"{STEAM_API_BASE_URL}/IPlayerService/GetSteamLevel/v1/"
            params = {"key": STEAM_API_KEY, "steamid": steam_id}
            resp = requests.get(url, params=params, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", {}).get("player_level", 0)
            else:
                print(f"⚠️  Level API Status {resp.status_code}")
                return 0
        except requests.exceptions.Timeout:
            print(f"⚠️  Level API Timeout")
            return 0
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Level API hatası: {e}")
            return 0
        except Exception as e:
            print(f"⚠️  Steam level parse hatası: {e}")
            return 0
    
    @staticmethod
    def get_player_badges(steam_id: str) -> Dict:
        """Oyuncunun rozetlerini ve başarımlarını al"""
        try:
            url = f"{STEAM_API_BASE_URL}/IPlayerService/GetBadges/v1/"
            params = {"key": STEAM_API_KEY, "steamid": steam_id}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            return data.get("response", {})
        except Exception as e:
            print(f"❌ Badges hatası: {e}")
        
        return {}
