# 考研智库 — AI Exam Intelligence System

[![CI](https://github.com/peng38159-eng/exam-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/peng38159-eng/exam-intelligence/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 基于本地 LLM + 知识图谱 + RAG 的考研备考系统。
> 自动理解教材、发现知识点关联、生成针对性练习题。

## 核心特性

- **多模态知识管理**：PDF教材、历年真题、笔记统一入库
- **双路检索**：向量语义相似度 + 知识图谱路径推理
- **多Agent协作**：SearchAgent → ReasonAgent → GeneratorAgent → ReviewAgent
- **遗忘曲线出题**：基于艾宾浩斯遗忘曲线智能生成练习题
- **全科覆盖**：数学一 / 计算机408 / 英语一 / 政治
- **100%本地运行**：Ollama + ChromaDB + NetworkX + LangGraph，零API费用

## 系统预览

```
┌─────────────────────────────────────────────────────────────┐
│                   考研智库 - 搜索问答 Tab                      │
├─────────────────────────────────────────────────────────────┤
│  💬 输入问题: 什么是极限的ε-N定义？     [数学一 ▼]            │
│  模式: ○ 深度解答  ○ 出题练习  ● 完整流程                     │
│  [🚀 提交]                                                  │
├─────────────────────────────────────────────────────────────┤
│  📝 推理结果                                                │
│  根据ε-N定义，证明lim(n→∞) 1/n = 0。                         │
│  证明：∀ε>0，要使|1/n-0|<ε，即1/n<ε，只需n>1/ε...            │
│                                                             │
│  📖 引用来源                                                │
│  ▼ 同济高数教材 p.23                                         │
│  📝 练习题                                                  │
│  ▼ 第1题 [解答题][基础] 用ε-N定义证明...                     │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com)（用于本地 LLM 推理）

### 一键安装

```bash
# 克隆仓库
git clone https://github.com/peng38159-eng/exam-intelligence.git
cd exam-intelligence

# 运行安装脚本（会自动安装依赖、拉取 qwen3:8b 模型）
bash scripts/setup.sh

# 或者手动安装
pip install -r requirements.txt
```

### 启动

```bash
# 确保 Ollama 在运行
ollama serve

# 拉取推荐模型
ollama pull qwen3:8b

# 启动 Web 界面
streamlit run src/ui/main.py
```

访问 http://localhost:8501

### 使用流程

1. 侧边栏点击 **初始化向量库**
2. 切换到「上传入库」Tab，上传 PDF 教材或 TXT 笔记
3. 在「搜索问答」Tab 中提问，获取 RAG 增强回答
4. 在「题库练习」Tab 中按知识点生成练习题
5. 在「仪表盘」Tab 查看掌握度实时监控

## 项目结构

```
exam-intelligence/
├── src/
│   ├── config.py                 # 统一配置
│   ├── ingestion/loader.py       # PDF/TXT 解析与分块
│   ├── retrieval/vectorstore.py  # ChromaDB 向量库与混合检索
│   ├── graph/knowledge_graph.py  # NetworkX 知识图谱
│   ├── agents/workflow.py        # LangGraph 多Agent工作流
│   ├── generator/review.py       # 遗忘曲线出题与复盘
│   └── ui/main.py               # Streamlit 主界面
├── tests/test_core.py            # 核心逻辑测试
├── docs/                         # 文档
│   ├── architecture.md
│   └── usage.md
├── scripts/setup.sh              # 一键安装脚本
├── .github/workflows/ci.yml      # CI 流水线
├── pyproject.toml                # 项目配置
├── requirements.txt
└── README.md
```

## 技术架构

```
数据采集层     文档处理层      知识管理层       AI推理层        用户交互层
PDF/TXT  -->  分块+Embedding --> ChromaDB  --> LangGraph  --> Streamlit
真题/笔记                          │
                                 NetworkX知识图谱  --> 多Agent协作 --> 搜索/出题/复盘
```

详见 [docs/architecture.md](docs/architecture.md)

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| LLM推理 | Ollama | 本地运行，隐私安全，支持 qwen3/llama3 |
| 向量库 | ChromaDB | 轻量，Python 原生，元数据过滤强大 |
| 图谱 | NetworkX | 无需额外部署，可视化方便 |
| Agent | LangGraph | 状态机驱动，多 Agent 协作成熟方案 |
| 前端 | Streamlit | Python 直出 Web，开发效率极高 |

## 适用场景

| 场景 | 说明 |
|------|------|
| 考研备考 | 数学一/计算机408/英语一/政治全科知识管理 |
| 自学复习 | 任何需要系统化知识管理的考试 |
| AI学习项目 | 学习 LangGraph/RAG/知识图谱的实战项目 |
| 论文/科研 | 研究性资料的知识图谱构建与检索 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest -q

# 代码检查
ruff check src tests

# 语法编译验证
python -m compileall -q src tests
```

## 后续规划

- [ ] 将 Mock LLM 切换为真实 Ollama 调用
- [ ] 知识图谱关系抽取升级为 LLM/NER 抽取
- [ ] 增加 Web 端知识图谱可视化（Plotly）
- [ ] 支持增量更新与重复文件识别
- [ ] 增加更多 408 课程题库（数据结构/计组/OS/计网）

## License

MIT License — 详见 [LICENSE](LICENSE)
