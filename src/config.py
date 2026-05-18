"""考研智库 — 统一配置模块

所有路径/模型/参数集中管理，支持环境变量覆盖。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class Config:
    """全局配置"""

    # --- 路径 ---
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EXAM_DATA_DIR", PROJECT_ROOT / "data")
        )
    )
    upload_dir: str = ""
    vectorstore_dir: str = ""
    graph_path: str = ""
    review_data_path: str = ""

    # --- Embedding ---
    embedding_model: str = "shibing624/text2vec-base-chinese"

    # --- LLM ---
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "qwen3:8b"

    # --- 分块 ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- 检索 ---
    search_top_k: int = 5
    hybrid_alpha: float = 0.7  # 向量权重 vs 关键词权重

    def __post_init__(self):
        # 从环境可覆盖路径
        self.upload_dir = os.environ.get(
            "EXAM_UPLOAD_DIR", str(self.data_dir / "uploads")
        )
        self.vectorstore_dir = os.environ.get(
            "EXAM_VECTORSTORE_DIR", str(self.data_dir / "vectorstore")
        )
        self.graph_path = os.environ.get(
            "EXAM_GRAPH_PATH", str(self.data_dir / "graph" / "knowledge_graph.gml")
        )
        self.review_data_path = os.environ.get(
            "EXAM_REVIEW_PATH", str(self.data_dir / "review_schedule.pkl")
        )

        # 确保必备目录存在
        for d in [self.upload_dir, self.vectorstore_dir,
                  str(Path(self.graph_path).parent),
                  str(Path(self.review_data_path).parent)]:
            Path(d).mkdir(parents=True, exist_ok=True)


# 全局单例
_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> Config:
    global _config
    _config = Config()
    return _config
