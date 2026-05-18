# 使用指南与开发说明

## 推荐使用流程

1. 安装依赖：`pip install -r requirements.txt`
2. 启动本地模型：`ollama serve`，并拉取 `qwen3:8b` 或其他中文能力较强的模型。
3. 启动界面：`streamlit run src/ui/main.py`
4. 在「上传入库」中上传 PDF/TXT 资料。
5. 在「搜索问答」中提问，或在「题库练习」中按知识点生成练习题。

## 已知运行约束

- PDF 解析依赖 PyMuPDF；只处理 TXT 或运行轻量测试时不强制加载 PyMuPDF。
- ChromaDB 和 sentence-transformers 首次加载会下载或读取 embedding 模型，耗时较长。
- 当前默认 LLM 调用仍保留 Mock 响应，便于没有 Ollama 时演示；后续可把 `src/agents/workflow.py` 的 `call_llm` 替换为真实 `ChatOllama` 调用。

## 质量检查

```bash
python -m compileall -q src tests
pytest -q
```

如果本机还没有测试工具：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## 数据目录

运行时会生成以下目录，默认不提交到 Git：

- `data/uploads/`：用户上传原始资料
- `data/vectorstore/`：ChromaDB 持久化向量库
- `data/graph/`：知识图谱持久化文件
- `data/cache/`：缓存文件

## 后续优化方向

- 将 Mock LLM 切换为真实 Ollama，并提供模型选择配置。
- 将知识图谱关系抽取从正则启发式升级为 LLM/NER 抽取。
- 增加更多 408、数学一、英语、政治题库模板。
- 给上传入库增加重复文件识别、增量更新和删除入口。
- 增加 GitHub Actions，自动运行 compileall、pytest、ruff。
