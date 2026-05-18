"""考研智库 - ChromaDB 向量库模块

支持 PDF chunk 的 embedding 生成、双路检索（向量 + 关键词）
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class VectorStore:
    """ChromaDB 向量存储与检索"""

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        model_name: str = "shibing624/text2vec-base-chinese",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 embedding 模型
        print(f"加载 Embedding 模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="exam_chunks",
            metadata={"description": "考研知识库向量存储"},
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成 embedding"""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def add_chunks(self, df: pd.DataFrame) -> None:
        """将 DataFrame 中的 chunk 批量入库"""
        if df.empty:
            print("没有可入库的 chunks")
            return

        ids = df["chunk_id"].astype(str).tolist()
        texts = df["text"].fillna("").astype(str).tolist()
        metadatas = []
        for _, row in df.iterrows():
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            text = str(row.get("text", ""))
            metadatas.append({
                "source": str(metadata.get("source") or row.get("source") or ""),
                "page": int(metadata.get("page") or row.get("page") or 0),
                "chunk_id": str(metadata.get("chunk_id") or row.get("chunk_id") or ""),
                "text_preview": text[:200],
            })

        embeddings = self.embed_texts(texts)

        # 使用 upsert，避免重复上传同一文件时 ChromaDB 因重复 id 报错。
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"✓ 入库完成: {len(ids)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_source: Optional[str] = None,
    ) -> List[dict]:
        """语义检索：返回 top_k 最相关的 chunk"""
        query_embedding = self.embed_texts([query])[0]

        where_filter = {"source": filter_source} if filter_source else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # 转为相似度
            })
        return hits

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
    ) -> List[dict]:
        """混合检索：向量相似度 * alpha + 关键词命中 * (1-alpha)"""
        # 向量检索
        vector_results = self.search(query, top_k * 2)

        # 简单关键词匹配：统计 query 中词在 doc 中的出现次数
        query_terms = set(query)
        keyword_scores = {}
        for r in vector_results:
            text_lower = r["text"].lower()
            # 简单词匹配分数
            matches = sum(1 for term in query_terms if term in text_lower)
            keyword_scores[r["chunk_id"]] = matches / max(len(query_terms), 1)

        # 归一化
        max_kw = max(keyword_scores.values()) if keyword_scores else 1
        for r in vector_results:
            r["keyword_score"] = keyword_scores.get(r["chunk_id"], 0) / max_kw
            r["final_score"] = (
                alpha * r["score"] + (1 - alpha) * r["keyword_score"]
            )

        # 按最终分数排序
        vector_results.sort(key=lambda x: x["final_score"], reverse=True)
        return vector_results[:top_k]

    def delete_by_source(self, source: str) -> None:
        """删除指定来源的所有 chunks"""
        self.collection.delete(where={"source": source})
        print(f"✓ 已删除来源: {source}")

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """清空向量库"""
        self.client.delete_collection("exam_chunks")
        self.collection = self.client.get_or_create_collection("exam_chunks")
        print("✓ 向量库已清空")


if __name__ == "__main__":
    # python -m src.retrieval.vectorstore
    vs = VectorStore()
    print(f"向量库当前容量: {vs.count()} chunks")