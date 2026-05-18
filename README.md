# 考研智库（AI Exam Intelligence System）

> 基于本地 LLM + 知识图谱 + RAG 的考研备考系统。
> 自动理解教材、发现知识点关联、生成针对性练习题。

## 核心特性

- **多模态知识管理**：PDF教材、历年真题、笔记统一入库
- **双路检索**：向量相似度 + 知识图谱路径推理
- **多Agent协作**：SearchAgent → ReasonAgent → GeneratorAgent → ReviewAgent
- **遗忘曲线出题**：基于遗忘曲线（Ebbinghaus）智能生成练习题
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
- Ollama（用于本地 LLM 推理）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama

```bash
# 安装 Ollama（如果未安装）
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型（首次运行需要）
ollama pull qwen3:8b
# 或
ollama pull llama3

# 启动服务
ollama serve
```

### 3. 启动 Web 界面

```bash
streamlit run src/ui/main.py
```

访问 http://localhost:8501

### 4. 上传资料入库

1. 进入「上传入库」Tab
2. 上传 PDF 教材或 TXT 笔记
3. 点击「开始入库」
4. 系统自动分块 → Embedding → 入向量库 + 构建知识图谱

## 功能演示

### Tab 1: 搜索问答

基于 RAG + 知识图谱的双路检索，对复杂考研问题给出深度解答，并标注引用来源。

### Tab 2: 题库练习

基于遗忘曲线调度，优先从薄弱知识点出题，支持选择题/填空题/解答题多种题型。

### Tab 3: 知识图谱

可视化知识点之间的前置关系、包含关系、推导关系，帮助理解学科全貌。

### Tab 4: 上传入库

批量上传 PDF/TXT，自动解析、分块、入库，构建个人专属考研知识库。

### Tab 5: 仪表盘

实时掌握度监控、待复习列表、遗忘曲线可视化。

## 技术架构

```
数据采集层     文档处理层      知识管理层       AI推理层        用户交互层
PDF/TXT  -->  分块+Embedding --> ChromaDB  --> LangGraph  --> Streamlit
真题/笔记                          │
                                 NetworkX知识图谱  --> 多Agent协作 --> 搜索/出题/复盘
```

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/usage.md](docs/usage.md)

## 项目结构

```
exam-intelligence/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── docs/
│   ├── architecture.md          # 系统架构文档
│   └── usage.md                 # 使用指南与开发说明
├── tests/
│   └── test_core.py             # 核心逻辑回归测试
├── scripts/
│   └── setup.sh                 # 环境一键安装
├── data/                        # 数据目录（运行时创建）
│   ├── uploads/                 # 上传的原始文件
│   ├── vectorstore/             # ChromaDB 向量库
│   └── graph/                   # 知识图谱文件
└── src/
    ├── __init__.py
    ├── ingestion/
    │   └── loader.py            # PDF/TXT 解析与分块
    ├── retrieval/
    │   └── vectorstore.py       # ChromaDB 向量库与混合检索
    ├── graph/
    │   └── knowledge_graph.py   # NetworkX 知识图谱
    ├── agents/
    │   └── workflow.py          # LangGraph 多Agent工作流
    ├── generator/
    │   └── review.py            # 遗忘曲线出题与复盘
    └── ui/
        └── main.py             # Streamlit 主界面
```

## 适用场景

| 场景 | 说明 |
|------|------|
| 考研备考 | 数学一/计算机408/英语一/政治全科知识管理 |
| 自学复习 | 任何需要系统化知识管理的考试 |
| AI学习项目 | 学习 LangGraph/RAG/知识图谱的实战项目 |
| 论文/科研 | 研究性资料的知识图谱构建与检索 |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| LLM推理 | Ollama | 本地运行，隐私安全，支持 qwen3/llama3 |
| 向量库 | ChromaDB | 轻量，Python 原生，元数据过滤强大 |
| 图谱 | NetworkX | 无需额外部署，可视化方便 |
| Agent | LangGraph | 状态机驱动，多 Agent 协作成熟方案 |
| 前端 | Streamlit | Python 直出 Web，开发效率极高 |

## License

MIT License - 详见 [LICENSE](LICENSE)