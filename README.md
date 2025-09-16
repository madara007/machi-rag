# machi-rag 🏙️

**machi-rag** は、地方自治体のオープンデータを活用して  
行政サービスや防災情報を検索・回答する **RAGアプリ** です。  

## 特長
- 🔎 **RAG検索**：ベクトルDB（FAISS）を活用し、関連情報を検索して回答を生成  
- 🏛️ **行政×防災**：生活に役立つ行政情報と防災情報を統合  
- ⚡ **モダン構成**：BackendにPython(FastAPI)、FrontendにNext.js(React)を採用  

## 技術スタック
- **Frontend**: Next.js (React), TypeScript, Tailwind CSS  
- **Backend**: Python (FastAPI)  
- **ベクトルDB**: FAISS  
- **LLM**: 任意のOpenAI互換API または Hugging Faceモデル  

## 選定理由（Vector DB）
- **FAISS**: 軽量でローカル環境でも簡単に動かせるため採用。  
  小規模〜中規模のオープンデータ検索に適しており、ポートフォリオ用途に適しています。  

## 使い方（イメージ）
1. ユーザーが「児童手当 申請方法」や「○○市 避難所」などの質問を入力  
2. FAISSで関連するオープンデータを検索  
3. LLMに検索結果を渡して回答を生成  
4. 回答とともに **参照元リンク（オープンデータの出典）** を表示  
