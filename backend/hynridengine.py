import os
import json
import faiss
import numpy as np
import openai
import pandas as pd
from dataclasses import dataclass
from dotenv import load_dotenv
from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi

from pypdf  import PdfReader
from sentence_transformers import SentenceTransformer

"""
FAISS（コサイン類似/IndexFlatIP）と BM25 を加重合成するハイブリッド検索。
- ベクトル: hugging face  Embeddings を使用
- BM25: Janomeで日本語トークナイズ
- 保存: FAISSインデックス＋texts.jsonl を保存して再ロード時のズレを回避
- intfloat/multilingual-e5-base → 768次元
"""



model_name = "intfloat/multilingual-e5-base"
model_path = f"path_to_model/{model_name}"
model = SentenceTransformer(model_path)

text  = ["ベクトル化したい文字列", "hogehoge"]

embedding = model.encode(text)

# --------------------------------------------------
# 2. FAISS インデックス作成（L2距離を使用）
#L2(ユークリッド距離)ベクトル次元の二乗の値を合計した平方根を求めるため長さを渡してる。
# np.array　shape 次元数の取得
"""
np.array
ValueError: not enough values to unpack (expected 2, got 1)
理由: arr.shape は (3,)（一次元）なので、2つの値（rows, cols）に分けられない。
二次元配列にしないとfaissに登録できない
atleast_2dを利用すると二次元配列の計算をしてくれるようになる
"""
#FAIssにベクトルを登録するさいに次元数インスタンスの引数に果たさないといけない。
_, d  = np.array(embedding).astype("float32").shape
index = faiss.IndexFlatL2(d)

index.add(np.array(embedding).astype('float32'))

# --------------------------------------------------
# 3. 日本語トークン化（Janome
tokenizer = Tokenizer()

tokenized_texts = [
  [token.surface for token in tokenizer.tokenize(t)]
  for t in text
]

#----------------------------------------------------
#Okapi Bm25 情報検索
#各文章においてその単語がどのくらい出現したのかを計算する
"""
BM25 のスコアは コーパス（文書）の単語の出方 と クエリ（検索語）の単語 の関係で決まります。
ただし「文字数」そのものではなく、単語数や単語の出現頻度と文書全体の長さが関係してきます。
"""
bm25 = BM25Okapi(tokenized_texts)

# 検索クエリの設定とトークン化
query_text = "テスト ベクトル"
tokenized_query = [token.surface for token in tokenizer.tokenize(query_text)]


# --------------------------------------------------
# 4. ベクトル検索（クエリをベクトル化し、FAISS で検索）
query_embedding = model.encode(query_text)
query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
"""
クエリベクトルとコーパス内の全ベクトルの距離を計算

距離が小さい順（または類似度が大きい順）に上位k件を返す

D = 距離のリスト, I = 文書番号のリスト
"""
D, I = index.search(query_vector, k=min(10, index.ntotal))

print("\n=== ベクトル検索結果 ===")
for rank, idx in enumerate(I[0],  start=1):
  print(f"順位 {rank}: テキスト='{text[idx]}', 距離={D[0][rank-1]:.4f}")


# --------------------------------------------------
# 5. キーワード検索（BM25）
print("\n=== キーワード検索（BM25）結果 ===")
bm25_scores = bm25.get_scores(tokenized_query)
indexed_scores = list(enumerate(bm25_scores))
indexed_scores.sort(key=lambda x: x[1], reverse=True)

print(indexed_scores)

for rank, (idx, score) in enumerate(indexed_scores[:10], start=1):
  print(f"順位 {rank}: テキスト='{text[idx]}', BM25スコア={score:.4f}")

# --------------------------------------------------
# 6. ハイブリッド検索（ベクトル + BM25 を統合）
print("\n=== ハイブリッド検索結果 ===")
hybrid_scores = []
for pos, idx in enumerate(I[0]):
  # ベクトルスコア（距離の逆数でスコア化）
  vec_score = 1 / (D[0][pos] + 1e-5)
  # BM25スコア（事前に算出済み）
  bm25_score = bm25_scores[idx]
  # ハイブリッドスコア（加重平均：ここでは 0.5:0.5）
  hybrid_score = 0.5 * vec_score + 0.5 * bm25_score
  hybrid_scores.append((idx, hybrid_score, D[0][pos], bm25_score))

# スコア順に並べ替え
hybrid_scores.sort(key=lambda x: x[1], reverse=True)
for rank, (idx, score, dist, bm) in enumerate(hybrid_scores, start=1):
  print(
    f"順位 {rank}: テキスト='{text[idx]}', "
    f"ハイブリッドスコア={score:.4f} (距離={dist:.4f}, BM25スコア={bm:.4f})"
  )
