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
├── JPK/ PROTSESUAL HUJJATLAR            # Jinoyat-Protsessual Kodeks
└── MJTK/  # Kriminalistika
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


---

**Author:** TergooAI Team Abdullayev Bobur 
**Version:** 3.0.0
