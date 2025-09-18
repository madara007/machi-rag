# clean_hybrid_pdf_search.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ========= Utils =========
def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
  return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def to_2d(x: np.ndarray) -> np.ndarray:
  return np.atleast_2d(np.asarray(x, dtype="float32"))

def minmax(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype="float32")
  if x.size == 0:
    return x
  mn, mx = float(x.min()), float(x.max())
  if mx - mn < 1e-12:
    return np.ones_like(x, dtype="float32")
  return (x - mn) / (mx - mn)

# ========= Document =========
@dataclass
class Document:
  text: str
  meta: Dict[str, Any]

# ========= Tokenizer (Japanese) =========
class JapaneseTokenizer:
  def __init__(self) -> None:
    self._tok = Tokenizer()

  def tokenize(self, text: str) -> List[str]:
    toks: List[str] = []
    for t in self._tok.tokenize(text):
      pos = t.part_of_speech.split(",")[0]
      if pos not in ("名詞", "動詞", "形容詞"):
        continue
      base = t.base_form if t.base_form and t.base_form != "*" else t.surface
      w = base.strip().lower()
      if len(w) >= 2:
        toks.append(w)
    return toks

# ========= Embedder (E5) =========
@dataclass
class EmbedderConfig:
  model_name: str = "intfloat/multilingual-e5-base"
  device: Optional[str] = None
  batch_size: int = 64
  use_e5_prefix: bool = True  # query:/passage:

class Embedder:
  def __init__(self, cfg: EmbedderConfig = EmbedderConfig()) -> None:
    self.cfg = cfg
    self.model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)

  @property
  def dim(self) -> int:
    return int(self.model.get_sentence_embedding_dimension())

  def _prepend(self, texts: Iterable[str], kind: str) -> List[str]:
    if not self.cfg.use_e5_prefix:
      return list(texts)
    prefix = f"{kind}: "
    return [prefix + t for t in texts]

  def encode_corpus(self, texts: List[str]) -> np.ndarray:
    inputs = self._prepend(texts, "passage")
    emb = self.model.encode(
      inputs, batch_size=self.cfg.batch_size, show_progress_bar=False, normalize_embeddings=False
    ).astype("float32")
    return l2_normalize(emb, axis=1)

  def encode_query(self, query: str) -> np.ndarray:
    q = self._prepend([query], "query")
    emb = self.model.encode(q, batch_size=1, show_progress_bar=False, normalize_embeddings=False).astype("float32")
    return l2_normalize(emb, axis=1)

# ========= FAISS (cosine via IP) =========
class FaissIndex:
  def __init__(self, dim: int) -> None:
    self.dim = dim
    self.index = faiss.IndexFlatIP(dim)  # 正規化済みベクトルならcos類似
    self.ntotal = 0

  def add(self, embeddings: np.ndarray) -> None:
    emb2d = to_2d(embeddings)
    if emb2d.shape[1] != self.dim:
      raise ValueError(f"Dim mismatch: expected {self.dim}, got {emb2d.shape[1]}")
    self.index.add(emb2d)
    self.ntotal = self.index.ntotal

  def search(self, query_vec: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q2d = to_2d(query_vec)
    k = min(k, self.index.ntotal) if self.index.ntotal > 0 else 0
    if k == 0:
      return np.zeros((q2d.shape[0], 0), dtype="float32"), np.zeros((q2d.shape[0], 0), dtype="int64")
    sims, ids = self.index.search(q2d, k)  # sims: 大きいほど良い
    return sims, ids

  def save(self, path: str) -> None:
    faiss.write_index(self.index, path)

  def load(self, path: str) -> None:
    self.index = faiss.read_index(path)
    self.dim = self.index.d
    self.ntotal = self.index.ntotal

# ========= BM25 =========
class BM25Index:
  def __init__(self, tokenizer: JapaneseTokenizer) -> None:
    self._tokenizer = tokenizer
    self._bm25: Optional[BM25Okapi] = None

  def build(self, docs: List[Document]) -> None:
    tokenized = [self._tokenizer.tokenize(d.text) for d in docs]
    self._bm25 = BM25Okapi(tokenized)

  def get_scores(self, query: str) -> np.ndarray:
    if self._bm25 is None:
      raise ValueError("BM25 is not built.")
    q = self._tokenizer.tokenize(query)
    scores = self._bm25.get_scores(q)
    return np.asarray(scores, dtype="float32")

# ========= Hybrid Ranker =========
@dataclass
class HybridWeights:
  alpha_vec: float = 0.6
  alpha_bm25: float = 0.4
  normalize: bool = True

class HybridRanker:
  def __init__(self, weights: HybridWeights = HybridWeights()) -> None:
    self.w = weights

  def combine(self, vec_scores: np.ndarray, bm25_scores: np.ndarray) -> np.ndarray:
    if self.w.normalize:
      v = minmax(vec_scores)
      b = minmax(bm25_scores)
    else:
      v, b = vec_scores, bm25_scores
    return self.w.alpha_vec * v + self.w.alpha_bm25 * b

# ========= PDF Parser =========
def parse_pdf_pages(path: str, max_pages: Optional[int] = None) -> List[Document]:
  """
  1ページ=1ドキュメント。meta に page(1始まり) と source を付与。
  LangChainのPyMuPDFLoaderを利用（日本語に強め）。
  """
  from langchain_community.document_loaders import PyMuPDFLoader
  raw_docs = PyMuPDFLoader(file_path=path).load()
  docs: List[Document] = []
  total = len(raw_docs) if max_pages is None else min(max_pages, len(raw_docs))
  for i in range(total):
    d = raw_docs[i]
    text = (d.page_content or "").replace("\n", " ").strip()
    page = d.metadata.get("page", i + 1)
    src = d.metadata.get("source", path)
    docs.append(Document(text=text, meta={"page": int(page), "source": src}))
  return docs

# ========= Persist paths =========
@dataclass
class PersistPaths:
  faiss_path: str = "faiss.index"
  docs_path: str = "docs.jsonl"   # JSONL: {"text": "...", "meta": {...}}

# ========= Search Engine (Facade) =========
class HybridSearchEngine:
  def __init__(
    self,
    embedder: Embedder,
    tokenizer: JapaneseTokenizer,
    ranker: HybridRanker,
    persist: Optional[PersistPaths] = None,
  ) -> None:
    self.embedder = embedder
    self.tokenizer = tokenizer
    self.ranker = ranker
    self.persist = persist or PersistPaths()
    self.docs: List[Document] = []
    self.vec_index: Optional[FaissIndex] = None
    self.bm25_index: Optional[BM25Index] = None

  # ----- Build -----
  def build_from_documents(self, docs: List[Document]) -> None:
    if not docs:
      raise ValueError("Empty documents.")
    # 空白ドキュメントを除外
    self.docs = [Document(text=d.text.strip(), meta=d.meta) for d in docs if d.text and d.text.strip()]

    texts = [d.text for d in self.docs]
    emb = self.embedder.encode_corpus(texts)
    self.vec_index = FaissIndex(self.embedder.dim)
    self.vec_index.add(emb)

    self.bm25_index = BM25Index(self.tokenizer)
    self.bm25_index.build(self.docs)

  def build_from_texts(self, texts: List[str], default_source: str = "inmemory") -> None:
    docs = [Document(text=t, meta={"page": i + 1, "source": default_source}) for i, t in enumerate(texts)]
    self.build_from_documents(docs)

  # ----- Persist -----
  def save(self) -> None:
    if self.vec_index is None:
      raise ValueError("Vector index not built.")
    self.vec_index.save(self.persist.faiss_path)
    with open(self.persist.docs_path, "w", encoding="utf-8") as f:
      for d in self.docs:
        f.write(json.dumps({"text": d.text, "meta": d.meta}, ensure_ascii=False) + "\n")

  def load(self) -> None:
    if not os.path.exists(self.persist.faiss_path):
      raise FileNotFoundError(self.persist.faiss_path)
    if not os.path.exists(self.persist.docs_path):
      raise FileNotFoundError(self.persist.docs_path)

    self.docs = []
    with open(self.persist.docs_path, "r", encoding="utf-8") as f:
      for ln in f:
        obj = json.loads(ln)
        self.docs.append(Document(text=obj["text"], meta=obj.get("meta", {})))

    self.vec_index = FaissIndex(dim=self.embedder.dim)
    self.vec_index.load(self.persist.faiss_path)

    self.bm25_index = BM25Index(self.tokenizer)
    self.bm25_index.build(self.docs)

  # ----- Search -----
  def search_vector(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
    if self.vec_index is None:
      raise ValueError("Vector index not built.")
    q = self.embedder.encode_query(query)
    sims, ids = self.vec_index.search(q, k)
    out: List[Tuple[int, float]] = []
    if ids.size == 0:
      return out
    for i, s in zip(ids[0], sims[0]):
      out.append((int(i), float(s)))  # cos類似
    return out

  def search_bm25(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
    if self.bm25_index is None:
      raise ValueError("BM25 index not built.")
    scores = self.bm25_index.get_scores(query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(int(i), float(s)) for i, s in ranked]

  def search_hybrid(self, query: str, k: int = 10, top_k_vec: int = 100, top_k_bm25: int = 200) -> List[Tuple[int, float]]:
    vec_hits = self.search_vector(query, k=min(top_k_vec, len(self.docs)))
    bm_hits  = self.search_bm25(query, k=min(top_k_bm25, len(self.docs)))

    vec_map = {i: s for i, s in vec_hits}
    bm_map  = {i: s for i, s in bm_hits}

    all_ids = sorted(set(vec_map) | set(bm_map))
    vec_scores = np.zeros(len(self.docs), dtype="float32")
    bm_scores  = np.zeros(len(self.docs), dtype="float32")
    for i in all_ids:
      if i in vec_map: vec_scores[i] = vec_map[i]
      if i in bm_map:  bm_scores[i]  = bm_map[i]

    hybrid = self.ranker.combine(vec_scores, bm_scores)
    ranked = sorted(((i, float(hybrid[i])) for i in all_ids), key=lambda x: x[1], reverse=True)[:k]
    return ranked

  # ----- Convenience -----
  def doc_at(self, i: int) -> Document:
    return self.docs[i]

# ========= Example =========
if __name__ == "__main__":
  from pathlib import Path

  # フォルダ内のPDFファイル一覧
  pdf_dir = Path("./dataset")
  pdf_files = sorted(pdf_dir.glob("*.pdf"))

  model_name = "intfloat/multilingual-e5-base"
  model_path = f"path_to_model/{model_name}"

  query = "姫路市 電話番号"
  for pdf_path in pdf_files:
    docs = parse_pdf_pages(pdf_path)  # 1ページ=1Doc(metaにpage,source)
    engine = HybridSearchEngine(
      embedder=Embedder(EmbedderConfig(model_name=model_path, use_e5_prefix=True)),
      tokenizer=JapaneseTokenizer(),
      ranker=HybridRanker(HybridWeights(alpha_vec=0.6, alpha_bm25=0.4, normalize=True)),
    )
    #engine.build_from_documents(docs=docs)
    engine.load()
    for i, score in engine.search_hybrid(query, k=5):
      d = engine.doc_at(i)
      print(f"{score:.4f} | p.{d.meta.get('page')} | {d.meta.get('source')} | {d.text[:60]}…")
