# Calisma Notlari: Steam Agentic AI Projesi

Bu notun amaci sunum ezberi yapmak degil; projedeki yapay zeka kavramlarini ne yaptigini bilerek anlatabilmektir.

Projenin kisa ozeti:

Bu proje, bir Steam kullanicisinin profilini ve oyun gecmisini alir. Daha sonra oyuncunun oyun tarzini analiz eder, ona uygun oyun onerileri uretir ve bu onerilerin neden yapildigini aciklar.

Ana fikir:

> "Hocam, bu projede Steam verisini alip yapay zeka ile oyuncu profili cikariyorum. Deep Learning ile oyun/oyuncu benzerligi hesapliyorum, RAG ile ilgili oyun baglamini getiriyorum, LLM ile metin ve oneriler uretiyorum, XAI ile onerilerin nedenini acikliyorum. Bunlari da agent yapisi ile parcalara bolerek calistiriyorum."

---

## 1. Deep Learning

### Basit Anlatim

Deep Learning burada oyuncu ve oyunlari sayisal vektorlere cevirmek icin kullaniliyor. Bilgisayar "RPG", "aksiyon", "hayatta kalma", "takim oyunu" gibi kavramlari direkt insan gibi anlamaz. Bu yuzden oyun aciklamalari, turleri ve oyuncunun oynadigi oyunlar embedding adi verilen sayisal temsillere cevrilir.

Bu sayede sistem su soruyu cevaplayabilir:

> "Bu oyuncunun sevdigi oyunlara anlamsal olarak benzeyen baska hangi oyunlar var?"

### Projede Nerede?

Dosya:

`core/embeddings.py`

Onemli sinif:

```python
class EmbeddingModel:
```

Bu sinif metni vektore cevirir.

Koddan ornek:

```python
def embed_text(self, text: str) -> np.ndarray:
    embedding = self.model.encode(text, convert_to_tensor=False)
    return np.array(embedding, dtype=np.float32)
```

Bu kodun anlami:

Bir metin aliyor, ornegin `"Dark Souls action RPG difficult combat"`, bunu sayisal bir vektore ceviriyor.

Projede ayrica oyun icin ozel metin hazirlaniyor:

```python
class GameEmbeddings:
    def create_game_text_representation(self, game):
        parts = []
        parts.append(f"Game: {game['name']}")
        parts.append(f"Genres: {genres_text}")
        parts.append(f"Description: {game['description'][:500]}")
        return " ".join(parts)
```

Yani oyun sadece ismiyle degil; turu, etiketi ve aciklamasiyla birlikte temsil ediliyor.

### Hocaya Kisa Cevap

> "Deep Learning kisminda oyunlari ve oyuncu profilini embedding vektorlerine ceviriyorum. Sonra bu vektorler arasinda benzerlik hesaplayarak oyuncuya yakin oyunlari buluyorum."

---

## 2. XAI - Explainable AI

### Basit Anlatim

XAI, yapay zekanin verdigi karari aciklamasi demektir. Sadece "Bu oyunu oneriyorum" demek yeterli degildir. Sistem "Neden bu oyunu onerdi?" sorusuna da cevap vermelidir.

Ornek:

> "Bu oyunu oneriyorum cunku oyuncu daha once uzun sure RPG ve aksiyon oyunlari oynamis."

### Projede Nerede?

Dosya:

`core/explainability.py`

Onemli sinif:

```python
class RecommendationExplainer:
```

Onemli fonksiyon:

```python
def explain_recommendation(self, game, player_profile, recommendation_score, language="en"):
```

Bu fonksiyon bir oyun onerisi icin aciklama uretir.

`agents/ranker_agent.py` icinde de bu aciklama kullaniliyor:

```python
explanation_dict = self.explainer.explain_recommendation(
    game=candidate.game.model_dump(),
    player_profile=user_profile,
    recommendation_score=candidate.final_score,
    language="tr"
)
```

Bu kodun anlami:

RankerAgent oyunu puanladiktan sonra, XAI modulu bu puanin neden verildigini acikliyor.

### Hocaya Kisa Cevap

> "XAI kisminda sistemin verdigi oneriyi acikliyorum. Yani kullanici sadece sonucu gormuyor, oyunun neden kendisine uygun oldugunu da gorebiliyor."

---

## 3. Generative AI

### Basit Anlatim

Generative AI yeni metin ureten yapay zekadir. Bu projede oyuncu hakkinda yorum, oyun tarzi analizi, guclu/zayif yonler ve oyun onerilerinin aciklamalari metin olarak uretiliyor.

Ornek cikti:

> "Bu oyuncu uzun soluklu, rekabetci ve takim tabanli oyunlari tercih ediyor."

### Projede Nerede?

Dosya:

`services/agentic_service.py`

Onemli fonksiyon:

```python
async def _llm_analyze(self, player_name, level, games, taste_profile):
```

Bu fonksiyon oyuncunun verilerini LLM'e gonderir ve analiz ister.

Koddan ornek:

```python
prompt = f"""Sen bir Steam oyun uzmanısın.
Aşağıdaki Steam oyuncu profilini analiz et ve
SADECE JSON formatında yanıt ver.
"""
```

Burada sistem LLM'e rol veriyor:

> "Sen bir Steam oyun uzmanisin."

Sonra hangi formatta cevap vermesi gerektigini soyluyor:

```python
{
    "character": "...",
    "playstyle": "...",
    "strengths": "...",
    "weaknesses": "...",
    "recommendations": [...]
}
```

### Hocaya Kisa Cevap

> "Generative AI kisminda oyuncu hakkinda dogal dilde analiz ve oyun onerisi aciklamalari uretiyorum."

---

## 4. LLM

### Basit Anlatim

LLM, buyuk dil modelidir. Bu projede Groq uzerinden Llama modeli kullaniliyor. LLM'in gorevi oyuncunun oyun gecmisini okuyup anlamli bir analiz ve oneriler uretmek.

### Projede Nerede?

Dosya:

`services/agentic_service.py`

Koddan ornek:

```python
from groq import Groq
```

```python
self.llm_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
```

Bu kod Groq istemcisini baslatiyor.

LLM cagrisi:

```python
response = self.llm_client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model=LLM_MODEL,
    temperature=0.7,
    max_tokens=1500,
)
```

Bu kodun anlami:

Hazirlanan prompt LLM'e gonderiliyor. LLM de JSON formatinda oyuncu analizi ve oyun onerileri donduruyor.

### Hocaya Kisa Cevap

> "LLM kisminda Groq uzerinden buyuk dil modeline oyuncunun oyun gecmisini gonderiyorum. Model bana oyuncu analizi ve oyun onerilerini JSON olarak donduruyor."

---

## 5. RAG - Retrieval Augmented Generation

### Basit Anlatim

RAG su demektir:

Once ilgili bilgileri getir, sonra LLM'e bu bilgilerle cevap urettir.

Normal LLM sadece prompt'taki bilgiye gore cevap verir. RAG'de ise sistem once oyuncuya benzeyen oyunlari bulur. Daha sonra bu oyunlari LLM'e baglam olarak verir.

Basit ornek:

1. Oyuncu Dark Souls, Elden Ring, Sekiro oynamis.
2. Sistem benzer oyunlari bulur.
3. LLM'e "Bu oyuncu bu tarz oyunlari seviyor, su aday oyunlar arasindan oner" denir.

### Projede Nerede?

Dosya:

`services/rag_service.py`

Onemli siniflar:

```python
class RAGRetriever:
class RAGPromptBuilder:
class RAGService:
```

`RAGRetriever` ilgili oyunlari getirir:

```python
def retrieve_similar_to_favorites(self, player_profile, top_k=15):
```

Bu fonksiyon oyuncunun en cok oynadigi oyunlara benzer oyunlari arar.

Koddan ornek:

```python
for favorite_game in player_profile["top_games"][:5]:
    embedding = game_embeddings.embed_game(favorite_game)
    similar_games = self.vector_store.search(embedding, top_k=5)
```

Bu kodun anlami:

Oyuncunun favori oyunlari embedding'e cevriliyor ve vektor veritabaninda benzer oyunlar araniyor.

RAG prompt'u burada hazirlaniyor:

```python
prompt = self.prompt_builder.build_recommendation_prompt(
    player_profile=player_profile,
    retrieved_games=retrieved_games,
    candidate_games=candidate_games,
    language=language
)
```

Yani LLM'e sadece oyuncu bilgisi degil, ayrica sistemin buldugu ilgili oyunlar da veriliyor.

### Hocaya Kisa Cevap

> "RAG kisminda once oyuncuya benzer oyunlari vektor arama ile getiriyorum. Sonra bu oyunlari LLM'e baglam olarak vererek daha isabetli oneriler urettiriyorum."

---

## 6. AI Agent

### Basit Anlatim

AI Agent, belli bir gorevi olan yapay zeka bilesenidir. Tek bir buyuk fonksiyon yerine her is icin ayri agent vardir.

Bu projede agent mantigi:

- ProfileAgent: Oyuncu profilini analiz eder.
- RetrievalAgent: Oneri adayi oyunlari getirir.
- RankerAgent: Aday oyunlari puanlar ve siralar.
- MatchmakingAgent: Uyumlu oyuncu/eslesme arar.

### Projede Nerede?

Klasor:

`agents/`

Dosyalar:

```text
agents/profile_agent.py
agents/retrieval_agent.py
agents/ranker_agent.py
agents/matchmaking_agent.py
```

Ornek:

`agents/profile_agent.py`

```python
class ProfileAgent(BaseAgent):
```

`agents/retrieval_agent.py`

```python
class RetrievalAgent(BaseAgent):
```

`agents/ranker_agent.py`

```python
class RankerAgent(BaseAgent):
```

Her agent `plan` ve `execute` mantigiyla calisir. Yani once ne yapacagini planlar, sonra uygular.

### Hocaya Kisa Cevap

> "AI Agent kisminda sistemi gorevlere boluyorum. Bir agent profil cikariyor, biri oyun adaylarini getiriyor, biri puanlayip siraliyor. Boylece sistem daha moduler oluyor."

---

## 7. Agentic AI

### Basit Anlatim

Agentic AI, birden fazla agent'in bir is akisi icinde birlikte calismasidir.

Yani tek bir AI cagrisi yok. Sistem adim adim ilerler:

1. Steam ID alinir.
2. Oyuncu bilgisi cekilir.
3. Oyun gecmisi analiz edilir.
4. Aday oyunlar bulunur.
5. Oyunlar puanlanir.
6. Aciklamalar eklenir.
7. Sonuc frontend'e dondurulur.

### Projede Nerede?

Dosya:

`orchestrator.py`

Onemli sinif:

```python
class Orchestrator:
```

Orchestrator agent'lari sirayla calistiran ana yoneticidir.

`app.py` icinde analiz endpoint'i bu sistemi kullanir:

```python
service = get_recommendation_service()
result = run_async(service.analyze_profile(steam_id))
```

`services/agentic_service.py` icinde:

```python
async def analyze_profile(self, steam_id: str) -> Dict[str, Any]:
```

Bu fonksiyon tum analiz akisini baslatir:

- Steam'den oyuncu verisini alir.
- Oyunlari alir.
- Oyuncu profili olusturur.
- LLM analizi yapar.
- Sonucu JSON olarak hazirlar.

### Hocaya Kisa Cevap

> "Agentic AI kisminda agent'lar bir pipeline gibi birlikte calisiyor. Sistem Steam verisini alip analiz ediyor, aday oyunlari getiriyor, siraliyor, acikliyor ve sonucu kullaniciya donduruyor."

---

## Projenin Genel Akisi

Projeyi anlatirken su sirayi kullan:

1. Kullanici Steam ID veya profil linki girer.
2. `app.py` icindeki `/api/analyze` endpoint'i calisir.
3. `SteamService` oyuncu ve oyun verilerini Steam API'den ceker.
4. `AgenticRecommendationService.analyze_profile()` analiz surecini baslatir.
5. LLM oyuncu profili icin metinsel analiz ve oneriler uretir.
6. RAG sistemi benzer oyun baglami saglar.
7. XAI sistemi onerilerin nedenini aciklar.
8. Frontend sonucu kartlar halinde gosterir.

Kod baglantisi:

```text
app.py
  -> services/agentic_service.py
  -> services/steam_service.py
  -> core/embeddings.py
  -> services/rag_service.py
  -> core/explainability.py
  -> agents/
```

---

## Hocaya 1 Dakikalik Anlatim

> "Hocam, projem Steam oyuncu profili uzerinden oyun onerisi yapan bir AI sistemidir. Kullanici Steam ID giriyor, sistem Steam API ile oyuncunun oyun gecmisini aliyor. Deep Learning tarafinda oyunlari ve oyuncu profilini embedding vektorlerine cevirip benzerlik hesapliyorum. RAG tarafinda oyuncuya uygun benzer oyunlari getirip LLM'e baglam olarak veriyorum. LLM ve Generative AI ile oyuncunun oyun tarzini, guclu-zayif yonlerini ve oyun onerilerini metin olarak uretiyorum. XAI ile de onerilerin neden yapildigini acikliyorum. AI Agent ve Agentic AI mimarisinde ise bu sureci profil agent'i, retrieval agent'i, ranker agent'i gibi parcalara bolerek yonetiyorum."

---

## Hocadan Gelebilecek Sorular

### Soru: Deep Learning nerede?

> "`core/embeddings.py` dosyasinda. Oyun ve oyuncu metinlerini embedding vektorlerine ceviriyorum. Bu vektorlerle benzerlik hesapliyorum."

### Soru: XAI nerede?

> "`core/explainability.py` dosyasinda. Onerilen oyunun neden onerildigini kullaniciya acikliyorum."

### Soru: LLM nerede?

> "`services/agentic_service.py` dosyasinda Groq istemcisi ile LLM cagrisi yapiyorum. Prompt hazirlanip modele gonderiliyor."

### Soru: RAG nerede?

> "`services/rag_service.py` dosyasinda. Once benzer oyunlari getiriyorum, sonra bunlari prompt'a ekleyerek LLM'e daha iyi baglam veriyorum."

### Soru: AI Agent nerede?

> "`agents/` klasorunde. ProfileAgent, RetrievalAgent, RankerAgent ve MatchmakingAgent farkli gorevleri yerine getiriyor."

### Soru: Agentic AI nerede?

> "`orchestrator.py` ve `services/agentic_service.py` dosyalarinda. Farkli agent'larin birlikte calistigi analiz pipeline'i burada yonetiliyor."

---

## En Kisa Ezber Cumlesi

> "Bu projede Steam verisini alip oyuncu profilini analiz ediyorum. Deep Learning ile benzerlik hesapliyorum, RAG ile ilgili oyunlari getiriyorum, LLM ile analiz ve oneriler uretiyorum, XAI ile nedenlerini acikliyorum. Agentic AI mimarisiyle de bu sureci farkli agent'lara bolerek yonetiyorum."
