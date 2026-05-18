# 使用指南与开发说明

## 推荐使用流程

1. 安装依赖：`pip install -r requirements.txt`
2. 启动本地模型：`ollama serve`，并拉取 `qwen3:8b` 或其他中文能力较强的模型。
3. 启动界面：`streamlit run src/ui/main.py`
4. 在「上传入库」中上传 PDF/TXT 资料。
5. 在「搜索问答」中提问，或在「题库练习」中按知识点生成练习题。

## 配置项

可通过环境变量覆盖默认配置（参见 `src/config.py`）：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `EXAM_DATA_DIR` | 数据根目录 | `./data` |
| `EXAM_UPLOAD_DIR` | 上传文件目录 | `./data/uploads` |
| `EXAM_VECTORSTORE_DIR` | 向量库目录 | `./data/vectorstore` |
| `EXAM_GRAPH_PATH` | 图谱文件路径 | `./data/graph/knowledge_graph.gml` |
| `EXAM_REVIEW_PATH` | 复习计划路径 | `./data/review_schedule.pkl` |

## 已知运行约束

- PDF 解析依赖 PyMuPDF；只处理 TXT 或运行轻量测试时不强制加载 PyMuPDF。
- ChromaDB 和 sentence-transformers 首次加载会下载或读取 embedding 模型，耗时较长。
- 当前默认 LLM 调用仍保留 Mock 响应，便于没有 Ollama 时演示；后续可把 `src/agents/workflow.py` 的 `call_llm` 替换为真实 `ChatOllama` 调用。
- TXT 文件自动检测编码（utf-8 / gbk / gb2312 / gb18030），无需手动转换。

## 质量检查

```bash
python -m compileall -q src tests
pytest -q
ruff check src tests
```

如果本机还没有测试工具：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## 数据目录

运行时会生成以下目录，默认不提交到 Git（已在 `.gitignore` 中排除）：

- `data/uploads/`：用户上传原始资料
- `data/vectorstore/`：ChromaDB 持久化向量库
- `data/graph/`：知识图谱持久化文件
- `data/cache/`：缓存文件

## API 概览

```python
from src.ingestion.loader import DocumentIngestion, quick_ingest
from src.retrieval.vectorstore import VectorStore
from src.graph.knowledge_graph import KnowledgeGraph
from src.agents.workflow import run_exam_query
from src.generator.review import ForgettingCurve, QuestionGenerator, ReviewAnalyzer
from src.config import get_config

# 配置
cfg = get_config()
print(cfg.embedding_model)  # shibing624/text2vec-base-chinese

# 文档入库
ingester = DocumentIngestion(upload_dir=cfg.upload_dir)
chunks = ingester.process_directory(cfg.upload_dir)

# 向量检索
vs = VectorStore(persist_dir=cfg.vectorstore_dir)
hits = vs.hybrid_search("什么是极限的ε-N定义？", top_k=5)

# 知识图谱
kg = KnowledgeGraph(graph_path=cfg.graph_path)
kg.add_relation("极限", "连续", "前置")
related = kg.get_related_concepts("极限", depth=2)

# Agent 工作流
result = run_exam_query("ε-N定义", subject="数学一", mode="answer")
print(result.reasoning_result)

# 遗忘曲线出题
fc = ForgettingCurve(review_data_path=cfg.review_data_path)
report = fc.get_mastery_report()

gen = QuestionGenerator(seed=42)
q = gen.generate_from_template("极限", "基础", subject="数学一")
```

## 后续优化方向

- 将 Mock LLM 切换为真实 Ollama，并提供模型选择配置。
- 将知识图谱关系抽取从正则启发式升级为 LLM/NER 抽取。
- 增加更多 408、数学一、英语、政治题库模板（已初步支持四科）。
- 给上传入库增加重复文件识别、增量更新和删除入口。
- GitHub Actions 已配置，自动运行 compileall、pytest、ruff。
