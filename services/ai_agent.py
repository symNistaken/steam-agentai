"""
AI Agent Servisi
Groq/Ollama ile Agentic Workflow

📋 AGENTIC WORKFLOW AÇIKLAMASI:
=====================================
"Agent AI" basit bir prompt engineering yapısıdır:

1. VERİ TOPLAMA: Steam API'den oyuncu verisi çek
2. PROMPT OLUŞTUR: Veriyi formatlanmış bir prompt'a koy
3. LLM ÇAĞIR: Groq/Ollama'ya gönder
4. YANIT PARSE ET: LLM çıktısını JSON olarak parse et
5. GÖRÜNTÜLE: Sonuçları frontend'de göster

⚠️ NOT: Bu proje LangChain kullanmaz çünkü:
  - Çok karmaşık (overcomplicated)
  - Token harcaması fazla
  - ReAct/Memory gereksiz
  - Basit one-shot prompting yeterli

✅ Bu yapıyla Agent AI yapabiliriz:
  - Analiz & Karar verme (Reasoning)
  - JSON strukturlu çıktı
  - Hızlı ve verimli
=====================================
"""
import json
from typing import Dict, List, Optional
from config import LLM_PROVIDER, GROQ_API_KEY, LLM_MODEL


class AIAgent:
    """
    AI Agent - Profil analizi ve oyun önerisi
    Veri → Prompt → LLM → JSON Parse → Sonuç
    """
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self._init_client()
    
    def _init_client(self):
        """LLM client'ını başlat"""
        if self.provider == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=GROQ_API_KEY)
                print("✓ Groq API bağlantısı kuruldu")
            except ImportError:
                raise ImportError("Groq yüklü değil: pip install groq")
    
    def analyze_player(self, player_data: Dict, games: List[Dict]) -> Dict:
        """
        Oyuncu profilini analiz et
        
        Return:
            {
                'character': 'Oyuncunun karakter profili',
                'playstyle': 'Oyun stili analizi',
                'recommendations': [{'name': '...', 'reason': '...', 'appid': 123}] (15 adet)
            }
        """
        try:
            # Top 10 oyunu hazırla
            top_games_text = "\n".join([
                f"• {g.get('name', 'Unknown')}: {g.get('playtime_forever', 0)//60} saat"
                for g in games[:10]
            ])
            
            # Agentic Prompt - Sadece Steam oyunları, 20 önerisine çıkar
            prompt = f"""Sen bir Steam oyun uzmanısın. Aşağıdaki Steam oyuncu profilini analiz et ve
SADECE JSON formatında yanıt ver.

⚠️ UYARI: Kullanıcıya oyun önerirken **ASLA** Steam'de olmayan oyunları 
(Valorant, League of Legends, Overwatch, Escape from Tarkov, Minecraft vb.) önerme. 
Sadece Steam mağazasında mevcut olan oyunları öner.

**OYUNCU BİLGİSİ:**
İsim: {player_data.get('personaname', 'Bilinmiyor')}
Seviye: {player_data.get('level', 0)}
Toplam Oyun: {len(games)}
Saati: {sum(g.get('playtime_forever', 0) for g in games)//60} saat

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
        {{"name": "Oyun Adı 10", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 11", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 12", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 13", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 14", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 15", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 16", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 17", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 18", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 19", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}},
        {{"name": "Oyun Adı 20", "appid": 123456, "reason": "Neden önerildiği (Türkçe)"}}
    ]
}}"""
            
            response_text = self._call_llm(prompt)
            result = self._parse_json(response_text)
            return result
        
        except Exception as e:
            print(f"❌ AI analiz hatası: {e}")
            return {
                'character': 'Analiz yapılamadı',
                'playstyle': 'Hata oluştu',
                'strengths': '-',
                'weaknesses': '-',
                'recommendations': []
            }
    
    def _call_llm(self, prompt: str) -> str:
        """LLM'e soru sor"""
        try:
            if self.provider == "groq":
                message = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=LLM_MODEL,
                    temperature=0.7,
                    max_tokens=1500,
                )
                return message.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM çağrı hatası: {e}")
        
        return ""
    
    def _parse_json(self, text: str) -> Dict:
        """Metinden JSON'ı çıkar ve parse et"""
        try:
            # JSON'ı bul
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse hatası: {e}")
        
        return {
            'character': 'Parse hatası',
            'playstyle': 'Yanıt işlenemiyor',
            'strengths': '-',
            'weaknesses': '-',
            'recommendations': []
        }
