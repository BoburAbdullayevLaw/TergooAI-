# 🤖 TergooAI - Semantic Search System

ML-powered semantic qidiruv tizimi (JK, JPK, Kriminalistika).

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/
├── JK/              # Jinoyat Kodeksi
├── JPK/             # Jinoyat-Protsessual Kodeks
└── kriminalistika/  # Kriminalistika
```

### 3. Run
```bash
python run.py
```

### 4. Test API
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "JK 97-2(a) nima?", "top_k": 5}'
```

## ✨ Features

- ✅ Semantic search (multilingual)
- ✅ 3 data sources: JK, JPK, Kriminalistika
- ✅ FAISS vector database
- ✅ FastAPI REST API
- ✅ Emoji preserved (✅ emojilar saqlanadi!)

## 📊 API Endpoints

- `POST /search` - Semantic search
- `GET /document/{id}` - Get document by ID
- `GET /stats` - System statistics
- `GET /health` - Health check

## 🎯 Architecture
```

TergooAI/
│
├── data/                                    # 📁 BARCHA MA'LUMOTLAR
│   ├── JK/                                  # Jinoyat Kodeksi
│   │   ├── jk-097/                          # Odam o'ldirish
│   │   │   ├── metadata.json                # Modda haqida metadata
│   │   │   ├── main.json                    # Asosiy modda matni
│   │   │   ├── sharh-001.json               # 1-sharh (oddiy o'ldirish)
│   │   │   ├── sharh-002.json               # 2-sharh (ikki odam)
│   │   │   ├── sharh-003.json               # 3-sharh (og'irlashtiruvchi)
│   │   │   └── ...
│   │   │
│   │   ├── jk-169/                          # O'g'irlik
│   │   │   ├── metadata.json
│   │   │   ├── main.json
│   │   │   └── sharh-001.json
│   │   │
│   │   └── jk-170/                          # Talon-taroj
│   │       ├── metadata.json
│   │       └── main.json
│   │
│   ├── JPK/                                 # Jinoyat-Protsessual Kodeks
│   │   ├── jpk-091/                         # Tergovchi
│   │   │   ├── metadata.json
│   │   │   ├── main.json
│   │   │   └── sharh-001.json
│   │   │
│   │   └── jpk-.../ 
│   │
│   ├── kriminalistika/                      # Kriminalistika
│   │   ├── barmoq_izlari/                   # Daktiloskopiya
│   │   │   ├── metadata.json
│   │   │   └── umumiy.json
│   │   │
│   │   ├── DNK_tahlil/                      # DNK
│   │   │   └── ...
│   │   │
│   │   └── oy_izlari/                       # Trasologiya
│   │       └── ...
│   │
│   ├── shablon_umumiy.json                  # Umumiy shablonlar
│   ├── stopwords_uz.txt                     # Stop-words
│   │
│   ├── faiss_index/                         # FAISS index (auto-generated)
│   │   └── main/
│   │       ├── faiss.index
│   │       └── metadata.pkl
│   │
│   └── cache/                               # Cache (auto-generated)
│       └── embeddings/
│
├── ML/                                      # 🤖 MACHINE LEARNING
│   ├── init.py
│   │
│   ├── embeddings/                          # Embedding generation
│   │   ├── init.py
│   │   ├── generator.py                     # ✅ EMOJI SAQLANADI
│   │   ├── models.py                        # Model management
│   │   └── cache.py                         # Caching
│   │
│   ├── vector_db/                           # Vector database
│   │   ├── init.py
│   │   ├── faiss_db.py                      # FAISS operations
│   │   ├── indexing.py                      # Index management
│   │   └── persistence.py                   # Save/load
│   │
│   ├── retrieval/                           # Search & retrieval
│   │   ├── init.py
│   │   ├── search.py                        # Semantic search
│   │   ├── ranking.py                       # Result ranking
│   │   └── filters.py                       # Filtering
│   │
│   ├── preprocessing/                       # Text processing
│   │   ├── init.py
│   │   ├── text_cleaner.py                  # ✅ MINIMAL (emoji saqlanadi)
│   │   ├── tokenizer.py                     # Tokenization
│   │   └── normalizer.py                    # Normalization
│   │
│   ├── evaluation/                          # Testing & metrics
│   │   ├── init.py
│   │   ├── metrics.py                       # Evaluation metrics
│   │   ├── testing.py                       # Test suites
│   │   └── benchmarks.py                    # Performance tests
│   │
│   └── utils/                               # Utilities
│       ├── init.py
│       ├── config.py                        # Configuration
│       ├── logger.py                        # Logging
│       └── helpers.py                       # Helper functions
│
├── logs/                                    # 📝 LOG FILES (auto-generated)
│   └── app.log
│
├── main.py                                  # 🚀 MAIN APPLICATION
├── run.py                                   # Runner script
├── config.py                                # App configuration
├── api_models.py                            # Pydantic models
├── database.py                              # ✅ Data loader (JK/JPK/KRIM)
├── startup.py
├── test_system.py                           # System tests
├── requirements.txt                         # Python dependencies
├── .env.example                             # Environment variables example
├── .gitignore                               # Git ignore rules
├── Makefile                                 # Build commands
├── Dockerfile                               # Docker image
├── docker-compose.yml                       # Docker compose
├── pytest.ini                               # Pytest config
├── setup.py                                 # Package setup
└── README.md   
```

## 📝 Example

```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "Qasddan odam o'ldirish", "top_k": 3}
)
print(response.json())
```

## ✅ Data Format

Documents kan have:
- `id`: Unique ID
- `asosiy_matn`: Main text
- `emoji_sarlavha`: Title with emoji (preserved!)
- `tushuntirish`: Explanation
- Other custom fields

## 🔧 Configuration

Edit `ML/utils/config.py` for custom settings.

---

**Author:** TergooAI Team Abdullayev Bobur 
**Version:** 3.0.0
