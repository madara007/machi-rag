from __future__ import annotations
import os
from typing import List,  Literal, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware  
import uvicorn


from search import (
  Embedder, EmbedderConfig,
  JapaneseTokenizer,
  HybridRanker, HybridWeights,
  HybridSearchEngine,
)

model_name = "intfloat/multilingual-e5-base"
model_path = f"path_to_model/{model_name}"

engine = HybridSearchEngine(
  embedder=Embedder(EmbedderConfig(model_name=model_path, use_e5_prefix=True)),
  tokenizer=JapaneseTokenizer(),
  ranker=HybridRanker(HybridWeights(alpha_vec=0.6, alpha_bm25=0.4, normalize=True)),
)
engine.load()

app = FastAPI(title="Hybrid PDF Search API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 必要なら追加: "http://your-domain.example"
  ],
  allow_credentials=True,
  allow_methods=["*"],   # または ["POST", "OPTIONS"]
  allow_headers=["*"],   # または ["Content-Type", "Authorization"]
)

# ---------- リクエスト/レスポンス モデル ----------
class BuildFromTextsReq(BaseModel):
  texts: List[str]
  source: str = "api"

class SearchReq(BaseModel):
  query: str = Field(..., description="検索クエリ")

class Hit(BaseModel):
  score: float
  text: str
  meta: Dict[str, Any]

class SearchRes(BaseModel):
  hits: List[Hit]

# ---------- 検索 ----------
@app.post("/search", response_model=SearchRes)
def search(req: SearchReq):
  try:
    hits = engine.search_hybrid(req.query, k=5)
    out: List[Hit] = []
    for idx, score in hits:
      d = engine.doc_at(idx)
      out.append(Hit(score=score, text=d.text, meta=d.meta))                          
    return SearchRes(hits=out)
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))

# ---------- ローカル起動 ----------
if __name__ == "__main__":
  # 例: uvicorn app:app --reload --port 8000
  uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
