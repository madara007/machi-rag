import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
import openai
import faiss

# =========================================================
# 環境変数の読み込み
# =========================================================
load_dotenv()

# OpenAI 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# デフォルトの保存先
DEFAULT_FAISS_PATH = "faiss.index"
DEFAULT_TEXTS_PATH = "texts.jsonl"


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
  """行ベクトルごとにL2正規化。"""
  norm = np.linalg.norm(x, axis=axis, keepdims=True) + eps
  return x / norm


class HybridSearchEngine:
  """
  FAISS（コサイン類似/IndexFlatIP）と BM25 を加重合成するハイブリッド検索。
  - ベクトル: OpenAI Embeddings を使用（手動でL2正規化）
  - BM25: Janomeで日本語トークナイズ
  - 保存: FAISSインデックス＋texts.jsonl を保存して再ロード時のズレを回避
  """

  def __init__(self) -> None:
    self.texts: List[str] = []
    self.tokenizer = Tokenizer()
    self.bm25: Optional[BM25Okapi] = None
    self.index: Optional[faiss.IndexFlatIP] = None
    self.embedding_dim: int = 1536  # text-embedding-3-small の次元数

  # -------------------- Embedding --------------------

  def get_embeddings(self, texts: List[str]) -> np.ndarray:
    """複数テキストの埋め込み取得"""
    resp = openai.embeddings.create(
      model="text-embedding-3-small",
      input=texts
    )
    emb = np.array([d.embedding for d in resp.data], dtype="float32")
    return emb

  def get_query_embedding(self, query: str) -> np.ndarray:
    """単一クエリの埋め込み取得"""
    resp = openai.embeddings.create(
      model="text-embedding-3-small",
      input=[query]
    )
    return np.array(resp.data[0].embedding, dtype="float32")

  # -------------------- Ingest / Build --------------------

  def build_index_from_excel(
    self,
    excel_path: str,
    text_column: str = "question",
    score_column: str = "総合評価",
    threshold: int = 20,
    max_rows: Optional[int] = None,
    texts_out_path: str = DEFAULT_TEXTS_PATH,
  ) -> None:
    df = pd.read_excel(excel_path, nrows=max_rows) if max_rows else pd.read_excel(excel_path)
    if score_column in df.columns:
      df = df[df[score_column] > threshold]
    if text_column not in df.columns:
      raise ValueError(f"{text_column} 列が見つかりません")

    texts = df[text_column].astype(str).tolist()
    self.build_index(texts)
    self.save_texts(texts_out_path)

  def build_index(self, texts: List[str]) -> None:
    if not texts:
      raise ValueError("空のテキストリストです。")
    self.texts = texts

    embeddings = self.get_embeddings(texts)
    emb = l2_normalize(embeddings, axis=1)

    self.embedding_dim = emb.shape[1]
    self.index = faiss.IndexFlatIP(self.embedding_dim)
    self.index.add(emb)

    tokenized_corpus = [self._tokenize_ja(t) for t in texts]
    self.bm25 = BM25Okapi(tokenized_corpus)

  # -------------------- Save / Load --------------------

  def save_index(self, faiss_path: str = DEFAULT_FAISS_PATH) -> None:
    if self.index is None:
      raise ValueError("FAISS インデックスが未構築です。")
    faiss.write_index(self.index, faiss_path)

  def save_texts(self, texts_path: str = DEFAULT_TEXTS_PATH) -> None:
    if not self.texts:
      raise ValueError("保存するテキストがありません。")
    with open(texts_path, "w", encoding="utf-8") as f:
      for t in self.texts:
        f.write(t.replace("\n", " ") + "\n")

  def load_index(self, faiss_path: str = DEFAULT_FAISS_PATH, texts_path: str = DEFAULT_TEXTS_PATH) -> None:
    if not os.path.exists(faiss_path):
      raise FileNotFoundError(f"インデックスファイルが見つかりません: {faiss_path}")
    if not os.path.exists(texts_path):
      raise FileNotFoundError(f"textsファイルが見つかりません: {texts_path}")

    self.index = faiss.read_index(faiss_path)
    self.embedding_dim = self.index.d

    with open(texts_path, "r", encoding="utf-8") as f:
      self.texts = [ln.rstrip("\n") for ln in f]

    tokenized = [self._tokenize_ja(t) for t in self.texts]
    self.bm25 = BM25Okapi(tokenized)

  # -------------------- Search --------------------

  def search_by_vector(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
    if self.index is None:
      raise ValueError("FAISS インデックスが未構築です。")
    q = self.get_query_embedding(query)[None, :]
    q = l2_normalize(q, axis=1)
    k = min(k, self.index.ntotal)
    sims, ids = self.index.search(q, k)
    out: List[Tuple[int, float]] = []
    for i, sim in zip(ids[0], sims[0]):
      if i == -1:
        continue
      out.append((int(i), float(sim)))
    return out

  def search_by_bm25(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
    if self.bm25 is None:
      raise ValueError("BM25 が未構築です。")
    query_tokens = self._tokenize_ja(query)
    scores = self.bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(int(i), float(s)) for i, s in ranked]

  def search_hybrid(
    self,
    query: str,
    k: int = 5,
    vec_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_k_vec: int = 100,
    top_k_bm25: int = 200,
    return_with_scores: bool = False,
  ) -> List[str] | List[Tuple[str, float]]:
    vec_hits = self.search_by_vector(query, k=min(top_k_vec, max(1, len(self.texts))))
    bm_hits = self.search_by_bm25(query,  k=min(top_k_bm25, max(1, len(self.texts))))

    vec_scores_raw: Dict[int, float] = {i: s for i, s in vec_hits}
    bm_scores_raw:  Dict[int, float] = {i: s for i, s in bm_hits}

    vec_scores = self._normalize_scores(vec_scores_raw)
    bm_scores  = self._normalize_scores(bm_scores_raw)

    all_ids = set(vec_scores) | set(bm_scores)
    combo: List[Tuple[int, float]] = []
    for i in all_ids:
      score = vec_weight * vec_scores.get(i, 0.0) + bm25_weight * bm_scores.get(i, 0.0)
      combo.append((i, score))
    combo.sort(key=lambda x: x[1], reverse=True)
    top = combo[:k]

    if return_with_scores:
      return [(self.texts[i], float(s)) for i, s in top]
    return [self.texts[i] for i, _ in top]

  # -------------------- Utils --------------------

  def _normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
      return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
      return {i: 1.0 for i in scores}
    return {i: (v - mn) / (mx - mn) for i, v in scores.items()}

  def _tokenize_ja(self, text: str) -> List[str]:
    toks: List[str] = []
    for t in self.tokenizer.tokenize(text):
      pos = t.part_of_speech.split(",")[0]
      if pos not in ("名詞", "動詞", "形容詞"):
        continue
      base = t.base_form if t.base_form and t.base_form != "*" else t.surface
      w = base.strip().lower()
      if not w or len(w) < 2:
        continue
      if all(ch in ".,:;!?()[]{}<>\"'`~|\\/+-=_*^%$#@＆※・、。　 " for ch in w):
        continue
      toks.append(w)
    return toks

# -------------------- サンプル実行 --------------------
if __name__ == "__main__":
  """
  最小デモ:
  - 初回: Excelからフィルタ→インデックス構築→保存
  - 2回目以降: 保存物をロードして検索
  """
  excel_file = "QA抽出_大学メール_2025.4.21.xlsx"  # 置き換えてください
  text_col = "question"
  score_col = "総合評価"
  threshold = 20
  query_text = "講演会の申し込みページを作りたい。必要な項目を教えて"
  faiss_file = DEFAULT_FAISS_PATH
  texts_file = DEFAULT_TEXTS_PATH

  engine = HybridSearchEngine()

  if not (os.path.exists(faiss_file) and os.path.exists(texts_file)):
    print("[INFO] Excel から読み込み、インデックスを構築中...")
    engine.build_index_from_excel(
      excel_path=excel_file,
      text_column=text_col,
      score_column=score_col,
      threshold=threshold,
      texts_out_path=texts_file,
    )
    engine.save_index(faiss_file)
  else:
    print("[INFO] 保存済みインデックスを読み込み中...")
    engine.load_index(faiss_file, texts_file)

  print("[INFO] ハイブリッド検索…")
  results = engine.search_hybrid(
    query=query_text,
    k=10,
    vec_weight=0.5,
    bm25_weight=0.5,
    top_k_vec=100,
    top_k_bm25=200,
    return_with_scores=True,
  )
  for i, (text, score) in enumerate(results, 1):
    print(f"{i:02d}. {score:.3f} | {text[:100]}{'…' if len(text) > 100 else ''}")
