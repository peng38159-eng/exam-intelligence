"""考研智库 - 文档采集与处理模块

支持 PDF/TXT 教材、历年真题、笔记的统一解析、分块、Embedding 入库
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
import pandas as pd
from tqdm import tqdm


class DocumentIngestion:
    """文档采集与预处理管道"""

    def __init__(
        self,
        upload_dir: str = "data/uploads",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._stats = {"files_processed": 0, "chunks_created": 0}

    def extract_text_from_pdf(self, pdf_path: str) -> List[dict]:
        """从 PDF 提取文本，按段落分块"""
        pages = []
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "page": page_num,
                    "text": text.strip(),
                    "source": os.path.basename(pdf_path),
                })
        doc.close()
        return pages

    def extract_text_from_txt(self, txt_path: str) -> List[dict]:
        """从 TXT 提取文本"""
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{
            "page": 1,
            "text": text.strip(),
            "source": os.path.basename(txt_path),
        }]

    def chunk_pages(self, pages: List[dict]) -> List[dict]:
        """将页面文本进一步分块，带 overlap"""
        chunks = []
        for page in pages:
            text = page["text"]
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                # 简单句子边界优化：尽量在句号处截断
                if end < len(text):
                    last_period = chunk_text.rfind("。")
                    if last_period > self.chunk_size * 0.6:
                        end = start + last_period + 1
                        chunk_text = text[start:end]

                chunk_id = hashlib.md5(
                    f"{page['source']}:{start}-{end}".encode()
                ).hexdigest()[:12]

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text.strip(),
                    "page": page["page"],
                    "source": page["source"],
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_id": chunk_id,
                    }
                })
                start = end - self.chunk_overlap

        self._stats["chunks_created"] += len(chunks)
        return chunks

    def process_file(self, file_path: str) -> List[dict]:
        """处理单个文件，返回 chunk 列表"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            pages = self.extract_text_from_pdf(file_path)
        elif ext == ".txt":
            pages = self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self.chunk_pages(pages)
        self._stats["files_processed"] += 1
        return chunks

    def process_directory(self, directory: str) -> List[dict]:
        """批量处理目录下所有支持的文件"""
        all_chunks = []
        dir_path = Path(directory)

        supported = ["*.pdf", "*.txt"]
        for pattern in supported:
            for file_path in tqdm(
                list(dir_path.glob(pattern)),
                desc=f"处理文件 ({pattern})"
            ):
                try:
                    chunks = self.process_file(str(file_path))
                    all_chunks.extend(chunks)
                    print(f"  ✓ {file_path.name}: {len(chunks)} chunks")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

        return all_chunks

    def get_stats(self) -> dict:
        return self._stats.copy()


def quick_ingest(
    upload_dir: str = "data/uploads",
    output_path: str = "data/chunks.parquet",
) -> pd.DataFrame:
    """快速入库入口：扫描上传目录，输出 chunk DataFrame"""
    ingester = DocumentIngestion(upload_dir=upload_dir)
    chunks = ingester.process_directory(upload_dir)

    if not chunks:
        print("未找到任何文件，请先将 PDF/TXT 放入 data/uploads/ 目录")
        return pd.DataFrame()

    df = pd.DataFrame(chunks)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n入库完成: {len(df)} chunks, 保存至 {output_path}")
    return df


if __name__ == "__main__":
    # python -m src.ingestion.loader
    df = quick_ingest()
    print(df.head(3))