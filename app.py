"""Steam User Analysis and Game Recommendation System - Agentic AI Architecture"""
import json
import uuid
import requests
from flask import Flask, render_template, request, jsonify
from config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from services.steam_service import SteamService
from services.ai_agent import AIAgent

# Initialize Agentic AI System
from init_app import ensure_initialized
ensure_initialized()

# Import the new agentic service
from services.agentic_service import get_recommendation_service, run_async

# Pre-initialize the recommendation service
logger = __import__('logging').getLogger(__name__)
logger.info("🔧 Pre-initializing recommendation service...")
_service = get_recommendation_service()
logger.info("✅ Recommendation service pre-initialized")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


def get_json_payload():
    """Read request JSON safely for local API endpoints."""
    return request.get_json(silent=True) or {}


def api_error(message: str = "İşlem sırasında bir hata oluştu", status_code: int = 500):
    """Return a user-facing error without leaking internal exception details."""
    return jsonify({'success': False, 'error': message}), status_code


def clamp_int(value, default: int, minimum: int, maximum: int) -> int:
    """Parse and clamp integer request parameters."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))

MANUAL_GAME_DB = {
    "dayz": 221100,
    "day z": 221100,
    "squad": 393380,
    "scum": 513710,
    "gtfo": 493520,
    "insurgency sandstorm": 581320,
    "insurgency: sandstorm": 581320,
    "conan exiles": 440900,
    "arma 3": 107410,
    "don't starve together": 322330,
    "dont starve together": 322330,
    "euro truck simulator 2": 227300,
    "ets2": 227300,
    "american truck simulator": 270880,
    "ats": 270880,
    "farming simulator 22": 1248130,
    "fs22": 1248130,
    "cities: skylines": 255710,
    "cities skylines": 255710,
    "hades ii": 1145350,
    "hades 2": 1145350,
    "left 4 dead 2": 550,
    "l4d2": 550,
    "left4dead2": 550,
    "l4d 2": 550,
    "warframe": 230410,
    "team fortress 2": 440,
    "tf2": 440,
    "cs2": 730, "cs:2": 730, "counter-strike 2": 730, "counter strike 2": 730,
    "csgo": 730, "cs:go": 730, "counter-strike": 730, "counter strike": 730, "cs": 730,
    "pubg": 578080, "playerunknown's battlegrounds": 578080,
    "apex legends": 1172470, "apex": 1172470,
    "rust": 252490,
    "dota 2": 570, "dota": 570,
    "gta 5": 271590, "gta v": 271590, "gta5": 271590, "gtav": 271590,
    "grand theft auto 5": 271590, "grand theft auto v": 271590,
}


KNOWN_GAMES = {
    'cs2': 730, 'cs:2': 730, 'counter-strike 2': 730, 'counter strike 2': 730,
    'csgo': 730, 'cs:go': 730, 'counter-strike': 730, 'counter strike': 730, 'cs': 730,
    'gta 5': 271590, 'gta v': 271590, 'gta5': 271590, 'gtav': 271590,
    'grand theft auto 5': 271590, 'grand theft auto v': 271590, 'gta iv': 12210, 'gta 4': 12210,
    'skyrim': 72850, 'tes v': 72850, 'elder scrolls': 72850, 'elder scrolls v': 72850,
    'skyrim se': 489830, 'skyrim special edition': 489830,
    'red dead 2': 1174180, 'rdr2': 1174180, 'red dead redemption 2': 1174180,
    'red dead 1': 40350, 'rdr1': 40350,
    'witcher 3': 292030, 'the witcher 3': 292030, 'tw3': 292030, 'witcher': 292030, 'witcher 2': 20920,
    'cyberpunk 2077': 1091500, 'cyberpunk': 1091500, 'cp2077': 1091500,
    'dark souls 3': 374320, 'ds3': 374320, 'dark souls iii': 374320,
    'dark souls': 374320, 'dark souls remastered': 570940, 'dsr': 570940,
    'elden ring': 801050, 'er': 801050,
    'sekiro': 814380, 'sekiro shadows die twice': 814380,
    'bloodborne': 359870,
    'baldurs gate 3': 1238140, 'bg3': 1238140, 'baldur\'s gate 3': 1238140, 'baldurs gate': 1238140,
    'starfield': 1716740,
    'no mans sky': 275850, 'nms': 275850,
    'minecraft': 578080, 'minecraft java': 578080, 'terraria': 105600,
    'rust': 252490, 'rust game': 252490,
    'ark': 346110, 'ark survival': 346110, 'ark survival evolved': 346110,
    'valheim': 892970, 'subnautica': 264710, 'palworld': 1623730,
    'dayz': 221100, 'day z': 221100,
    'left 4 dead 2': 550, 'l4d2': 550, 'l4d 2': 550, 'left4dead2': 550,
    'warframe': 230410, 'warframe online': 230410,
    'hades': 1145360, 'hollow knight': 367520, 'stardew valley': 413150, 'stardew': 413150,
    'greedfall': 606800,
    'dying light 2': 534850, 'dying light': 239200,
    'control': 528560,
    'dishonored': 205100, 'dishonored 2': 403640,
    'monster hunter': 582010, 'monster hunter world': 582010, 'mhw': 582010,
    'ffxiv': 39120, 'final fantasy 14': 39120, 'ff14': 39120,
    'mass effect': 1329320, 'mass effect legendary': 1329320,
    'fallout 4': 377160, 'fallout': 377160,
    'fallout new vegas': 22380, 'fnv': 22380, 'fallout 76': 1151340,
    'pubg': 578080, 'pubg: battlegrounds': 578080, 'playerunknown\'s battlegrounds': 578080,
    'dota 2': 570, 'dota': 570,
    'squad': 393380,
    'hunting ground': 593000, 'predator hunting grounds': 593000,
    'team fortress 2': 440, 'tf2': 440,
    'rainbow six siege': 359550, 'r6': 359550, 'r6 siege': 359550, 'tom clancy\'s rainbow six siege': 359550,
    'arma 3': 107410,
    'insurgency sandstorm': 581320, 'insurgency: sandstorm': 581320,
    'garry\'s mod': 4000, 'gmod': 4000,
    'scum': 513710,
    'gtfo': 493520,
    'conan exiles': 440900,
    'euro truck simulator 2': 227300, 'ets2': 227300, 'euro truck': 227300,
    'phasmophobia': 739630,
    'dead by daylight': 381210, 'dbd': 381210,
    'lossless scaling': 993090,
    'crosshair x': 1366800,
    'sea of thieves': 1172620, 'sea of theives': 1172620,
    'mount & blade ii: bannerlord': 261550, 'bannerlord': 261550, 'mount and blade': 261550,
    'don\'t starve together': 322330, 'dont starve together': 322330,
    'valorant': 'https://images.contentstack.io/v3/assets/blt3a7e3725a11d11e8/blt5f47050229983699/632a121d4c22154109147735/valorant-share.jpg',
    'league of legends': 'https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Ahri_0.jpg',
    'lol': 'https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Ahri_0.jpg',
    'league': 'https://ddragon.leagueoflegends.com/cdn/img/champion/splash/Ahri_0.jpg',
    'overwatch 2': 'https://blz-contentstack-images.akamaized.net/v3/assets/blt3a7e3725a11d11e8/bltf3f6f49eb5f66ae6/632c0e69a4ac0a1a5c2d0f9a/overwatch-2-key-art.jpg',
    'overwatch': 'https://blz-contentstack-images.akamaized.net/v3/assets/blt3a7e3725a11d11e8/bltf3f6f49eb5f66ae6/632c0e69a4ac0a1a5c2d0f9a/overwatch-2-key-art.jpg',
    'ow': 'https://blz-contentstack-images.akamaized.net/v3/assets/blt3a7e3725a11d11e8/bltf3f6f49eb5f66ae6/632c0e69a4ac0a1a5c2d0f9a/overwatch-2-key-art.jpg',
    'ow2': 'https://blz-contentstack-images.akamaized.net/v3/assets/blt3a7e3725a11d11e8/bltf3f6f49eb5f66ae6/632c0e69a4ac0a1a5c2d0f9a/overwatch-2-key-art.jpg',
    'escape from tarkov': 'https://www.escapefromtarkov.com/image/design/general/share-image.jpg',
    'tarkov': 'https://www.escapefromtarkov.com/image/design/general/share-image.jpg',
    'eft': 'https://www.escapefromtarkov.com/image/design/general/share-image.jpg',
    'fortnite': 'https://www.epicgames.com/fortnite/en-US/_next/static/images/ogimage-1200x630-8b96e23c6ce05767db2251e6b25e9a9a.jpg',
    'apex legends': 'https://media.contentapi.ea.com/content/dam/apex-legends/images/2023-02/apex-social-share.jpg',
    'apex': 'https://media.contentapi.ea.com/content/dam/apex-legends/images/2023-02/apex-social-share.jpg',
    'call of duty': 'https://images.ctfassets.net/c1xwx36lyc59/25nzQV9YE4zf4fpvUq5H6N/be8f6e9e40f3c30e72f3c7c1e2c4d5e6/warzone-key-art.jpg',
    'cod': 'https://images.ctfassets.net/c1xwx36lyc59/25nzQV9YE4zf4fpvUq5H6N/be8f6e9e40f3c30e72f3c7c1e2c4d5e6/warzone-key-art.jpg',
    'warzone': 'https://images.ctfassets.net/c1xwx36lyc59/25nzQV9YE4zf4fpvUq5H6N/be8f6e9e40f3c30e72f3c7c1e2c4d5e6/warzone-key-art.jpg',
}


def search_game_id(game_name: str):
    """Find game AppID or image URL from database or Steam API"""
    if not game_name:
        return None
    
    game_name_lower = game_name.lower().strip()
    
    if game_name_lower in MANUAL_GAME_DB:
        result = MANUAL_GAME_DB[game_name_lower]
        print(f"  ✓ MANUAL_GAME_DB: '{game_name}' → {result}")
        return result
    
    if game_name_lower in KNOWN_GAMES:
        result = KNOWN_GAMES[game_name_lower]
        if isinstance(result, str):
            print(f"  ✓ KNOWN_GAMES (URL): '{game_name}'")
            return result
        else:
            print(f"  ✓ KNOWN_GAMES (AppID): '{game_name}' → {result}")
            return result
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        url = f"https://store.steampowered.com/api/storesearch/?term={game_name}&l=english&cc=US"
        resp = requests.get(url, headers=headers, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            items = data.get('items', [])
            
            if items:
                excluded = ['soundtrack', ' artbook', ' wallpaper', ' demo', 'digital art']
                for item in items:
                    item_name = item.get('name', '').lower()
                    appid = item.get('appid', 0)
                    
                    if any(skip in item_name for skip in excluded):
                        continue
                    
                    if appid > 0:
                        print(f"  ✓ STEAM API: '{game_name}' → AppID {appid}")
                        return appid
                
                return None
    except Exception as e:
        print(f"  ⚠️ Steam API error: {game_name}")
        return None
    
    return None

def get_game_store_url(appid: int) -> str:
    """Create Steam Store URL from AppID"""
    if appid and appid > 0:
        return f"https://store.steampowered.com/app/{appid}/"
    return "#"

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Steam profilini analiz et - AGENTIC AI VERSION
    POST /api/analyze
    JSON: {"steam_input": "76561197960265730"} veya URL
    """
    try:
        data = get_json_payload()
        steam_input = data.get('steam_input', '').strip()
        use_agentic = data.get('use_agentic', True)  # Default to new system
        
        if not steam_input:
            return jsonify({'success': False, 'error': 'Steam ID veya URL girin'}), 400
        
        print(f"\n🔍 Profil araması: {steam_input}")
        
        # 1. Steam ID'yi çöz
        steam_id = SteamService.resolve_steam_id(steam_input)
        if not steam_id:
            return jsonify({'success': False, 'error': 'Geçersiz Steam ID/URL'}), 400
        
        print(f"✓ Steam ID: {steam_id}")
        
        # Use new Agentic AI system
        if use_agentic:
            print("🤖 Agentic AI sistemi kullanılıyor...")
            service = get_recommendation_service()
            result = run_async(service.analyze_profile(steam_id))
            
            if not result["success"]:
                return jsonify({'success': False, 'error': result.get('error', 'Analiz başarısız')}), 400
            
            response = {
                'success': True,
                'player': result['player'],
                'stats': result['stats'],
                'top_games': result.get('top_games', []),
                'library': result['library'],
                'analysis': result['analysis'],
            }
            
            print(f"✅ Agentic analysis successful\n")
            return jsonify(response), 200
        
        # Fallback to old system
        print("📦 Klasik sistem kullanılıyor...")
        
        # 2. Oyuncu bilgisini al
        player = SteamService.get_player_summary(steam_id)
        if not player:
            return jsonify({'success': False, 'error': 'Profil bulunamadı'}), 404
        
        print(f"✓ Oyuncu: {player.get('personaname')}")
        
        # 3. Oyunları al
        games = SteamService.get_owned_games(steam_id)
        print(f"✓ Oyun sayısı: {len(games)}")
        
        # 4. Level al (hata yakalama ile)
        level = SteamService.get_player_level(steam_id)
        print(f"✓ Level: {level}")
        
        # 5. AI analizi yap
        print("🤖 AI analizi başlıyor...")
        agent = AIAgent()
        analysis = agent.analyze_player({'personaname': player.get('personaname'), 'level': level}, games)
        print(f"✓ AI analizi tamamlandı")
        
        # 6. Önerilere resim/link ekle (AI'DAN AppID GELİYOR)
        print(f"\n📋 Oyun Önerileri İşleniyor ({len(analysis.get('recommendations', []))} adet):")
        recommendations = []
        
        for idx, rec in enumerate(analysis.get('recommendations', []), 1):
            game_name = rec.get('name', '')
            reason = rec.get('reason', '')
            ai_appid = rec.get('appid', 0)  # ✓ AI'DAN AppID GELİYOR
            
            print(f"  [{idx}] {game_name} (AI AppID: {ai_appid})")
            
            # AppID'yi belirle (AI'nın verdiğini koru veya fallback'e git)
            appid = ai_appid if ai_appid > 0 else search_game_id(game_name)
            if appid is None:
                appid = 0
            
            if isinstance(appid, int) and appid > 0:
                image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
            else:
                image_url = "https://via.placeholder.com/460x215/2d3e50/66c0f4?text=Image+Not+Found"
            
            store_url = get_game_store_url(appid)
            recommendations.append({
                'name': game_name,
                'reason': reason,
                'appid': appid,
                'image': image_url,
                'store_url': store_url
            })
            
            if len(recommendations) >= 10:
                break
        
        print(f"\n✅ {len(recommendations)} games processed")
        
        library = []
        for game in games[:10]:
            game_name = game.get('name', 'Unknown')
            appid = game.get('appid', 0)
            
            if appid > 0:
                image_url = f"https://cdn.akamai.steamstatic.com/steam/apps/{appid}/header.jpg"
            else:
                image_url = "https://via.placeholder.com/460x215/1b2838/66c0f4?text=No+Image"
            
            library.append({
                'name': game_name,
                'playtime': round(game.get('playtime_forever', 0) / 60, 1),
                'appid': appid,
                'image': image_url,
                'store_url': get_game_store_url(appid)
            })
        
        response = {
            'success': True,
            'player': {
                'steam_id': steam_id,
                'name': player.get('personaname', 'Unknown'),
                'avatar': player.get('avatarfull', ''),
                'country': player.get('loccountrycode', 'Unknown'),
                'profile_url': player.get('profileurl', '#'),
                'level': level
            },
            'stats': {
                'total_games': len(games),
                'total_playtime': round(sum(g.get('playtime_forever', 0) for g in games) / 60, 1),
                'avg_playtime': round(sum(g.get('playtime_forever', 0) for g in games) / max(len(games), 1) / 60, 1)
            },
            'top_games': [{'name': g.get('name', 'Unknown'), 'playtime': round(g.get('playtime_forever', 0) / 60, 1), 'appid': g.get('appid')} for g in games[:5]],
            'library': library,
            'analysis': {
                'character': analysis.get('character', '-'),
                'playstyle': analysis.get('playstyle', '-'),
                'strengths': analysis.get('strengths', '-'),
                'weaknesses': analysis.get('weaknesses', '-'),
                'recommendations': recommendations
            }
        }
        
        print(f"✅ Analysis successful\n")
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return api_error()


@app.route('/api/agentic/recommend', methods=['POST'])
def agentic_recommend():
    """
    New Agentic AI Recommendation Endpoint.
    Uses the full orchestrator pipeline.
    
    POST /api/agentic/recommend
    JSON: {
        "steam_id": "76561197960265730",
        "max_recommendations": 10,
        "include_matchmaking": false
    }
    """
    try:
        data = get_json_payload()
        steam_input = data.get('steam_id', '').strip()
        
        if not steam_input:
            return jsonify({'success': False, 'error': 'Steam ID gerekli'}), 400
        
        # Resolve Steam ID
        steam_id = SteamService.resolve_steam_id(steam_input)
        if not steam_id:
            return jsonify({'success': False, 'error': 'Geçersiz Steam ID'}), 400
        
        print(f"\n🚀 Agentic Recommendation Pipeline başlatılıyor...")
        print(f"   Steam ID: {steam_id}")
        
        # Use the agentic service
        service = get_recommendation_service()
        result = run_async(service.analyze_profile(steam_id))
        
        if not result["success"]:
            return jsonify({'success': False, 'error': result.get('error')}), 400
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Agentic Error: {e}")
        return api_error()

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    from agents.base import AgentRegistry, ToolRegistry
    from database.db_manager import get_db_manager
    
    agent_count = len(AgentRegistry().list_agents())
    tool_count = len(ToolRegistry().list_tools())
    
    # Get DB stats
    db = get_db_manager()
    db_stats = db.get_stats()
    
    return jsonify({
        'status': 'ok',
        'system': 'agentic_ai',
        'agents_registered': agent_count,
        'tools_registered': tool_count,
        'database': db_stats,
    }), 200


@app.route('/api/db/stats', methods=['GET'])
def db_stats():
    """Get database statistics."""
    from database.db_manager import get_db_manager
    
    db = get_db_manager()
    stats = db.get_stats()
    
    return jsonify({
        'success': True,
        'stats': stats,
    }), 200


@app.route('/api/matchmaking', methods=['POST'])
def find_matches():
    """
    Matchmaking endpoint - Find compatible players using FoF strategy.
    
    POST /api/matchmaking
    JSON: {"steam_id": "76561197960265730"}
    """
    try:
        data = get_json_payload()
        steam_input = data.get('steam_id', '').strip()
        
        if not steam_input:
            return jsonify({'success': False, 'error': 'Steam ID gerekli'}), 400
        
        # Resolve Steam ID
        steam_id = SteamService.resolve_steam_id(steam_input)
        if not steam_id:
            return jsonify({'success': False, 'error': 'Geçersiz Steam ID'}), 400
        
        print(f"\n🎯 Matchmaking başlatılıyor: {steam_id}")
        
        # Use the real MatchmakingAgent
        from agents.matchmaking_agent import MatchmakingAgent
        from agents.base import AgentContext
        from schemas.models import AgentPlan
        
        agent = MatchmakingAgent()
        context = AgentContext(steam_id=steam_id, request_id=str(uuid.uuid4()))
        
        # Run the agent with proper async handling
        import asyncio
        
        async def run_matchmaking():
            plan = await agent.plan(context, "Find compatible players")
            return await agent.execute(context, plan)
        
        # Use asyncio.run() for cleaner event loop handling
        try:
            result = asyncio.run(run_matchmaking())
        except RuntimeError:
            # Fallback for nested event loops
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(run_matchmaking())
            finally:
                loop.close()
        
        if not result.success:
            return jsonify({
                'success': False, 
                'error': result.data.get('error', 'Matchmaking başarısız')
            }), 400
        
        return jsonify({
            'success': True,
            'matches': result.data.get('matches', []),
            'total_analyzed': result.data.get('total_analyzed', 0),
            'message': result.data.get('message', 'Tamamlandı')
        }), 200
        
    except Exception as e:
        print(f"❌ Matchmaking Error: {e}")
        import traceback
        traceback.print_exc()
        return api_error()


# ============================================================================
# RAG & XAI ENDPOINTS (Deep Learning + Explainability)
# ============================================================================

@app.route('/api/rag/context', methods=['POST'])
def get_rag_context():
    """
    Get RAG (Retrieval Augmented Generation) context for recommendations.
    
    Returns retrieved similar games from vector database for context.
    
    POST /api/rag/context
    JSON: {
        "steam_id": "76561197960265730",
        "max_context": 15
    }
    """
    try:
        data = get_json_payload()
        steam_input = data.get('steam_id', '').strip()
        max_context = clamp_int(data.get('max_context', 15), default=15, minimum=1, maximum=50)
        
        if not steam_input:
            return jsonify({'success': False, 'error': 'Steam ID gerekli'}), 400
        
        # Resolve Steam ID
        steam_id = SteamService.resolve_steam_id(steam_input)
        if not steam_id:
            return jsonify({'success': False, 'error': 'Geçersiz Steam ID'}), 400
        
        print(f"\n🔍 RAG Context alınıyor: {steam_id}")
        
        from services.rag_service import RAGService
        from services.steam_service import SteamService as SS
        
        rag_service = RAGService()
        steam_service = SS()
        
        # Fetch player data
        player_data = steam_service.get_player_summary(steam_id)
        owned_games = steam_service.get_owned_games(steam_id)
        
        if not player_data or not owned_games:
            return jsonify({'success': False, 'error': 'Oyuncu verisi alınamadı'}), 400
        
        if rag_service.retriever.vector_store.total_vectors == 0:
            from core.embeddings import GameEmbeddings
            import numpy as np

            game_embeddings = GameEmbeddings(embedding_model=rag_service.retriever.embedding_model)
            indexed_games = owned_games[:50]
            embeddings = np.array(
                [game_embeddings.embed_game(game) for game in indexed_games],
                dtype=np.float32
            )
            rag_service.retriever.vector_store.add_games(indexed_games, embeddings)

        # Get context games
        retrieved = rag_service.retriever.retrieve_similar_to_favorites(
            player_profile={
                "top_games": owned_games[:5],
                "favorite_genres": []
            },
            top_k=max_context
        )
        
        context_games = [
            {
                "name": game.get("name"),
                "app_id": game.get("app_id") or game.get("appid"),
                "genres": game.get("genres", []),
                "similarity": float(sim),
                "reason": reason
            }
            for game, sim, reason in retrieved
        ]
        
        print(f"✓ {len(context_games)} context game bulundu")
        
        return jsonify({
            'success': True,
            'steam_id': steam_id,
            'context_games': context_games,
            'total_retrieved': len(context_games)
        }), 200
        
    except Exception as e:
        print(f"❌ RAG Context Error: {e}")
        import traceback
        traceback.print_exc()
        return api_error()


@app.route('/api/xai/explain', methods=['POST'])
def get_xai_explanation():
    """
    Get XAI (Explainable AI) explanation for why a game was recommended.
    
    Explains the decision factors using feature importance and LIME.
    
    POST /api/xai/explain
    JSON: {
        "steam_id": "76561197960265730",
        "game_name": "Dark Souls 3",
        "recommendation_score": 0.85
    }
    """
    try:
        data = get_json_payload()
        steam_input = data.get('steam_id', '').strip()
        game_name = data.get('game_name', '').strip()
        rec_score = float(data.get('recommendation_score', 0.7))
        
        if not steam_input or not game_name:
            return jsonify({'success': False, 'error': 'Steam ID ve oyun adı gerekli'}), 400
        
        # Resolve Steam ID
        steam_id = SteamService.resolve_steam_id(steam_input)
        if not steam_id:
            return jsonify({'success': False, 'error': 'Geçersiz Steam ID'}), 400
        
        print(f"\n🤔 XAI Explanation üretiliyor: {game_name}")
        
        from core.explainability import get_recommendation_explainer
        from services.steam_service import SteamService as SS
        
        explainer = get_recommendation_explainer()
        steam_service = SS()
        
        # Fetch player profile
        player_data = steam_service.get_player_summary(steam_id)
        owned_games = steam_service.get_owned_games(steam_id)
        
        if not player_data:
            return jsonify({'success': False, 'error': 'Oyuncu verisi alınamadı'}), 400
        
        # Extract genres from owned games
        favorite_genres = []
        for game in owned_games[:10]:
            favorite_genres.extend(game.get("genres", []))
        
        player_profile = {
            "favorite_genres": list(set(favorite_genres)),
            "top_games": owned_games[:5],
            "playstyle": player_data.get("persona_name", "Unknown"),
            "difficulty_preference": "Medium",
            "avg_session_hours": 2.0
        }
        
        # Create game object for explanation
        game_obj = {
            "name": game_name,
            "genres": [],
            "tags": [],
            "rating": rec_score,
            "difficulty_score": 0.5,
            "estimated_playtime_hours": 30
        }
        
        # Generate explanation
        explanation = explainer.explain_recommendation(
            game=game_obj,
            player_profile=player_profile,
            recommendation_score=rec_score,
            language="tr"
        )
        
        print(f"✓ Explanation generated")
        
        return jsonify({
            'success': True,
            'game_name': game_name,
            'recommendation_score': rec_score,
            'confidence': explanation.get('confidence', 'MEDIUM'),
            'explanation': explanation.get('explanation', ''),
            'top_factors': explanation.get('top_factors', []),
            'feature_breakdown': explanation.get('feature_breakdown', {})
        }), 200
        
    except Exception as e:
        print(f"❌ XAI Explanation Error: {e}")
        import traceback
        traceback.print_exc()
        return api_error()


@app.route('/api/embeddings/similarity', methods=['POST'])
def get_embeddings_similarity():
    """
    Calculate semantic similarity between games using Deep Learning embeddings.
    
    POST /api/embeddings/similarity
    JSON: {
        "game1_name": "Dark Souls 3",
        "game2_name": "Elden Ring"
    }
    """
    try:
        data = get_json_payload()
        game1_name = data.get('game1_name', '').strip()
        game2_name = data.get('game2_name', '').strip()
        
        if not game1_name or not game2_name:
            return jsonify({'success': False, 'error': 'İki oyun adı gerekli'}), 400
        
        print(f"\n📊 Embedding similarity hesaplanıyor: {game1_name} vs {game2_name}")
        
        from core.embeddings import EmbeddingModel
        
        embedding_model = EmbeddingModel()
        similarity = embedding_model.calculate_similarity(game1_name, game2_name)
        
        print(f"✓ Similarity: {similarity:.3f}")
        
        return jsonify({
            'success': True,
            'game1': game1_name,
            'game2': game2_name,
            'similarity_score': float(similarity),
            'interpretation': (
                'Çok benzer oyunlar' if similarity > 0.7
                else 'Kısmen benzer' if similarity > 0.4
                else 'Farklı oyunlar'
            )
        }), 200
        
    except Exception as e:
        print(f"❌ Embeddings Similarity Error: {e}")
        import traceback
        traceback.print_exc()
        return api_error()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🎮 STEAM USER ANALYSIS AND RECOMMENDATION SYSTEM")
    print("🤖 AGENTIC AI ARCHITECTURE v3.0 - WITH DEEP LEARNING & XAI")
    print("="*70)
    print(f"✓ http://localhost:{FLASK_PORT}")
    print(f"✓ Debug: {FLASK_DEBUG}")
    print("="*70)
    print("\n📡 API Endpoints:")
    print(f"   POST /api/analyze           - Profil analizi (Agentic)")
    print(f"   POST /api/agentic/recommend - Tam pipeline (RAG + XAI)")
    print(f"   POST /api/matchmaking       - Eşleşme bul")
    print(f"   GET  /api/health            - Sistem durumu")
    print("\n🔍 RAG (Retrieval Augmented Generation):")
    print(f"   POST /api/rag/context       - Context oyunları getir")
    print("\n🤔 XAI (Explainable AI):")
    print(f"   POST /api/xai/explain       - Açıklamayı getir")
    print(f"   POST /api/embeddings/similarity - Benzerlik hesapla")
    print("="*70)
    print("\n⚙️  Enabled Features:")
    print("   ✓ Deep Learning (SentenceTransformer)")
    print("   ✓ Vector Database (FAISS)")
    print("   ✓ RAG (Retrieval Augmented Generation)")
    print("   ✓ XAI (Explainable AI - LIME/SHAP)")
    print("   ✓ Multi-Agent Orchestration")
    print("   ✓ Generative AI (Groq LLM)")
    print("="*70 + "\n")
    
    app.run(debug=FLASK_DEBUG, host=FLASK_HOST, port=FLASK_PORT)
