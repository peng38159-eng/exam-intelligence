# 考研智库（AI Exam Intelligence System）

基于本地 LLM + 知识图谱 + RAG 的考研备考系统。自动理解教材、发现知识点关联、生成针对性练习题。

## 核心特性

- **多模态知识管理**：PDF教材、历年真题、笔记统一入库
- **双路检索**：向量相似度 + 知识图谱路径推理
- **多Agent协作**：检索Agent、推理Agent、出题Agent、复盘Agent
- **遗忘曲线出题**：基于薄弱点智能生成练习题
- **100%本地运行**：Ollama + ChromaDB + NetworkX，零API费用

## 快速开始

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/exam-intelligence.git
cd exam-intelligence

# 一键安装依赖
pip install -r requirements.txt

# 启动 Ollama（确保本地运行）
ollama serve  # 默认 http://localhost:11434

# 启动 Web 界面
streamlit run src/ui/main.py
```

## 系统架构

详见 [docs/architecture.md](docs/architecture.md)

## 技术栈

- LLM推理: Ollama (qwen3/llama3)
- 向量库: ChromaDB
- 知识图谱: NetworkX
- Agent框架: LangGraph
- Web界面: Streamlit

## License

MIT