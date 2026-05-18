"""考研智库 - LangGraph 多Agent协作工作流

四个专业Agent：
- SearchAgent：双路检索（向量+图谱），找到最相关的知识块
- ReasonAgent：多跳推理，综合多来源答案
- GeneratorAgent：基于遗忘曲线生成针对性练习题
- ReviewAgent：分析答题结果，更新知识图谱掌握度
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
try:
    from langgraph.graph import StateGraph, END
except Exception:  # 允许在未安装 LangGraph 时运行核心 Mock 流程和单元测试
    StateGraph = None
    END = "__end__"
try:
    from langchain_core.messages import AIMessage
except Exception:
    @dataclass
    class AIMessage:
        content: str
import json


# ============ Agent State ============

@dataclass
class AgentState:
    """多Agent共享状态"""
    user_query: str = ""
    subject: str = "通用"  # 数学一/408/英语/政治

    # SearchAgent 输出
    retrieved_chunks: list = field(default_factory=list)
    graph_paths: list = field(default_factory=list)

    # ReasonAgent 输出
    reasoning_result: str = ""
    cited_sources: list = field(default_factory=list)

    # GeneratorAgent 输出
    generated_questions: list = field(default_factory=list)

    # ReviewAgent 输出
    updated_mastery: dict = field(default_factory=dict)
    review_summary: str = ""

    # 中间追踪
    active_agent: Optional[str] = None
    messages: list = field(default_factory=list)
    error: Optional[str] = None


# ============ Agent Prompts ============

SEARCH_PROMPT = """你是一个专业的考研知识检索Agent。你的任务是根据用户问题，从知识库中精确检索相关内容。

用户问题：{user_query}
科目：{subject}

请执行双路检索：
1. 向量检索：找到语义最相关的知识片段
2. 图谱检索：找到与问题相关的概念及其关联路径

输出格式（JSON）：
{{
  "retrieved_chunks": [
    {{"text": "片段内容", "source": "来源", "page": 页码, "relevance": 0.0~1.0}}
  ],
  "graph_paths": [
    {{"path": ["概念1", "概念2", "概念3"], "relation": "关系类型"}}
  ],
  "reasoning_hints": "检索过程中发现的关键推理线索"
}}"""


REASON_PROMPT = """你是一个专业的考研推理Agent。你的任务是基于检索结果，为用户提供深度解答。

用户问题：{user_query}
科目：{subject}

检索到的知识片段：
{retrieved_chunks}

知识图谱关联路径：
{graph_paths}

请综合以上信息，提供：
1. 完整的深度解答（结合多个知识源）
2. 明确标注每个结论的来源（页码/来源）
3. 如果图谱路径存在，解释概念间的推理链条

输出格式（JSON）：
{{
  "answer": "完整解答...",
  "reasoning_chain": "推理链条说明...",
  "cited_sources": [
    {{"text": "引用的原文", "source": "来源", "page": 页码}}
  ],
  "confidence": 0.0~1.0,
  "gaps": "如果知识不足，明确说明哪些地方需要补充"
}}"""


GENERATOR_PROMPT = """你是一个专业的考研出题Agent。你的任务是基于遗忘曲线和用户的薄弱知识点，生成高质量练习题。

用户问题方向：{user_query}
科目：{subject}

当前掌握情况（知识点 → 掌握度）：
{mastery_info}

最近检索到的相关知识：
{recent_knowledge}

出题要求：
1. 优先从薄弱知识点出题
2. 题目要有区分度（基础题/中档题/难题比例合理）
3. 每道题要标注：题型、难度、考察知识点、参考答案

输出格式（JSON数组）：
{{
  "questions": [
    {{
      "id": 1,
      "type": "选择题/填空题/解答题",
      "difficulty": "基础/中档/困难",
      "topic": "考察的知识点",
      "question": "题目内容",
      "answer": "参考答案",
      "explanation": "解题思路"
    }}
  ]
}}"""


REVIEW_PROMPT = """你是一个专业的考研复盘Agent。你的任务是根据用户的答题结果，分析薄弱环节并更新知识图谱。

用户答题情况：
{answer_results}

原始问题：
{original_questions}

请分析：
1. 每道题的答题情况（正确/错误/部分正确）
2. 错误对应的知识点
3. 每个知识点的掌握度变化（基于对错情况）
4. 下次复习的优先级建议

输出格式（JSON）：
{{
  "analysis": [
    {{
      "question_id": 1,
      "correct": true/false,
      "weak_concepts": ["知识点1", "知识点2"],
      "mastery_delta": -0.2 ~ +0.2
    }}
  ],
  "updated_mastery": {{
    "知识点": 新掌握度(0.0~1.0)
  }},
  "review_priority": ["下次优先复习的知识点列表"],
  "summary": "整体复盘总结"
}}"""


# ============ Mock Agent 实现（可替换为真实 Ollama 调用）==========

def call_llm(prompt: str, model: str = "qwen3:8b") -> str:
    """调用本地 Ollama LLM

    实际项目中替换此处为：
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model, base_url="http://localhost:11434")
    return llm.invoke(prompt)
    """
    # TODO: 替换为真实 Ollama 调用
    # 目前返回结构化的模拟响应，用于开发测试
    return mock_llm_response(prompt)


def mock_llm_response(prompt: str) -> str:
    """模拟LLM响应（开发阶段使用）"""
    if "推理Agent" in prompt or "深度解答" in prompt:
        return json.dumps({
            "answer": "根据ε-N定义，证明lim(n→∞) 1/n = 0。\n\n证明：∀ε>0，要使|1/n - 0| < ε，即1/n < ε，只需n > 1/ε。\n\n取N = ⌈1/ε⌉，则当n > N时，有|1/n - 0| < ε，故lim(n→∞) 1/n = 0。\n\n关键点：ε的任意性是极限定义的核心。",
            "reasoning_chain": "从ε-N定义出发，通过不等式变换找到N的构造方法，体现了极限的ε-N语言本质。",
            "cited_sources": [
                {"text": "极限的ε-N定义", "source": "同济高数教材", "page": 23},
                {"text": "夹逼定理", "source": "同济高数教材", "page": 25},
            ],
            "confidence": 0.92,
            "gaps": ""
        }, ensure_ascii=False)
    elif "出题Agent" in prompt or "高质量练习题" in prompt:
        return json.dumps({
            "questions": [
                {
                    "id": 1,
                    "type": "解答题",
                    "difficulty": "基础",
                    "topic": "ε-N定义证明极限",
                    "question": "用ε-N定义证明：lim(n→∞) 2n/(n+1) = 2",
                    "answer": "∀ε>0，要使|2n/(n+1) - 2| = 2/(n+1) < ε，只需n > 2/ε - 1，取N = ⌈2/ε - 1⌉即可。",
                    "explanation": "先将表达式化简为标准形式，再反解n。"
                },
                {
                    "id": 2,
                    "type": "选择题",
                    "difficulty": "中档",
                    "topic": "夹逼定理应用",
                    "question": "已知x_n ≤ y_n ≤ z_n，且lim x_n = lim z_n = 1，则lim y_n = ?",
                    "answer": "1",
                    "explanation": "直接由夹逼定理得。"
                },
                {
                    "id": 3,
                    "type": "解答题",
                    "difficulty": "困难",
                    "topic": "综合证明",
                    "question": "证明：如果lim a_n = A，则lim (a_1 + a_2 + ... + a_n)/n = A",
                    "answer": "提示：利用 Stolz 公式或两边夹。",
                    "explanation": "这是重要的极限性质，可由Heine定理证明。"
                },
            ]
        }, ensure_ascii=False)
    elif "复盘Agent" in prompt or "答题结果" in prompt:
        return json.dumps({
            "analysis": [
                {"question_id": 1, "correct": True, "weak_concepts": [], "mastery_delta": 0.1},
                {"question_id": 2, "correct": True, "weak_concepts": [], "mastery_delta": 0.05},
                {"question_id": 3, "correct": False, "weak_concepts": ["Stolz公式", "Heine定理"], "mastery_delta": -0.15},
            ],
            "updated_mastery": {
                "ε-N定义": 0.85,
                "夹逼定理": 0.78,
                "Stolz公式": 0.35,
                "Heine定理": 0.30,
            },
            "review_priority": ["Stolz公式", "Heine定理", "极限性质综合"],
            "summary": "ε-N定义掌握较好，但Stolz公式和Heine定理是薄弱点，建议专项练习。"
        }, ensure_ascii=False)
    elif "SearchAgent" in prompt or "精确检索" in prompt:
        return json.dumps({
            "retrieved_chunks": [
                {"text": "极限的ε-N定义：设f(n)为数列，∀ε>0，∃N∈N+，当n>N时，恒有|f(n)-A|<ε，则称A为数列的极限。", "source": "同济高数教材", "page": 23, "relevance": 0.95},
                {"text": "夹逼定理：如果数列x_n, y_n, z_n满足x_n≤y_n≤z_n，且lim x_n = lim z_n = A，则lim y_n = A。", "source": "同济高数教材", "page": 25, "relevance": 0.88},
            ],
            "graph_paths": [
                {"path": ["数列", "极限", "连续", "导数"], "relation": "包含"},
            ],
            "reasoning_hints": "极限是微分和连续的基础，前置知识为数列。"
        }, ensure_ascii=False)
    return json.dumps({"error": "未知prompt类型"})


# ============ LangGraph 节点函数 ============

def search_node(state: AgentState) -> AgentState:
    """SearchAgent 节点"""
    state.active_agent = "SearchAgent"

    prompt = SEARCH_PROMPT.format(
        user_query=state.user_query,
        subject=state.subject,
    )

    try:
        response = call_llm(prompt)
        result = json.loads(response)

        state.retrieved_chunks = result.get("retrieved_chunks", [])
        state.graph_paths = result.get("graph_paths", [])
        state.messages.append(AIMessage(content=f"检索完成，找到{len(state.retrieved_chunks)}个相关片段"))
    except Exception as e:
        state.error = f"SearchAgent错误: {e}"

    return state


def reason_node(state: AgentState) -> AgentState:
    """ReasonAgent 节点"""
    state.active_agent = "ReasonAgent"

    chunks_text = "\n".join([
        f"[{c['source']} p.{c['page']}] {c['text']}"
        for c in state.retrieved_chunks
    ])
    paths_text = "\n".join([
        " → ".join(p["path"]) + f" ({p['relation']})"
        for p in state.graph_paths
    ])

    prompt = REASON_PROMPT.format(
        user_query=state.user_query,
        subject=state.subject,
        retrieved_chunks=chunks_text,
        graph_paths=paths_text or "无图谱路径",
    )

    try:
        response = call_llm(prompt)
        result = json.loads(response)

        state.reasoning_result = result.get("answer", "")
        state.cited_sources = result.get("cited_sources", [])
        state.messages.append(AIMessage(content="推理完成，生成深度解答"))
    except Exception as e:
        state.error = f"ReasonAgent错误: {e}"

    return state


def generate_node(state: AgentState) -> AgentState:
    """GeneratorAgent 节点"""
    state.active_agent = "GeneratorAgent"

    # 简化的掌握度信息（实际从图谱读取）
    mastery_info = json.dumps({
        "ε-N定义": 0.75,
        "夹逼定理": 0.70,
        "Stolz公式": 0.40,
    }, ensure_ascii=False)

    recent_knowledge = "\n".join([
        c["text"][:200] for c in state.retrieved_chunks[:3]
    ])

    prompt = GENERATOR_PROMPT.format(
        user_query=state.user_query,
        subject=state.subject,
        mastery_info=mastery_info,
        recent_knowledge=recent_knowledge,
    )

    try:
        response = call_llm(prompt)
        result = json.loads(response)

        state.generated_questions = result.get("questions", [])
        state.messages.append(AIMessage(content=f"生成{len(state.generated_questions)}道练习题"))
    except Exception as e:
        state.error = f"GeneratorAgent错误: {e}"

    return state


def review_node(state: AgentState) -> AgentState:
    """ReviewAgent 节点"""
    state.active_agent = "ReviewAgent"

    # 从 state 中获取答题结果
    answer_results = state.generated_questions  # 简化：直接用生成题目作为答题记录
    original = state.user_query

    prompt = REVIEW_PROMPT.format(
        answer_results=json.dumps(answer_results, ensure_ascii=False, indent=2),
        original_questions=original,
    )

    try:
        response = call_llm(prompt)
        result = json.loads(response)

        state.updated_mastery = result.get("updated_mastery", {})
        state.review_summary = result.get("summary", "")
        state.messages.append(AIMessage(content="复盘完成，更新知识图谱"))
    except Exception as e:
        state.error = f"ReviewAgent错误: {e}"

    return state


# ============ LangGraph 工作流 ============

def build_exam_workflow():
    """构建考研Agent工作流图"""
    if StateGraph is None:
        raise RuntimeError("LangGraph 未安装，请先执行 pip install -r requirements.txt")

    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("search", search_node)
    workflow.add_node("reason", reason_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)

    # 设置入口和边
    workflow.set_entry_point("search")
    workflow.add_edge("search", "reason")
    workflow.add_edge("reason", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", END)

    return workflow.compile()


def run_exam_query(
    query: str,
    subject: str = "数学一",
    mode: Literal["answer", "practice", "review"] = "answer",
) -> AgentState:
    """对外统一的查询入口

    Args:
        query: 用户问题
        subject: 科目
        mode: answer=深度解答, practice=出题练习, review=答题复盘

    Returns:
        AgentState: 包含各Agent执行结果
    """
    # 使用显式顺序执行作为稳定默认路径：避免 LangGraph 版本差异导致
    # dataclass state 被转换成 dict 后 UI 访问属性失败。
    state = AgentState(user_query=query, subject=subject)
    state = search_node(state)
    if state.error:
        return state

    state = reason_node(state)
    if state.error or mode == "answer":
        return state

    state = generate_node(state)
    if state.error or mode == "practice":
        return state

    state = review_node(state)
    return state


if __name__ == "__main__":
    # python -m src.agents.workflow
    print("=== 考研Agent工作流测试 ===")

    # 测试检索+推理
    result = run_exam_query("什么是极限的ε-N定义？", subject="数学一", mode="answer")
    print("\n推理结果：")
    print(result.reasoning_result[:300])
    print(f"\n引用来源：{len(result.cited_sources)} 条")