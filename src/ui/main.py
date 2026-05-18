"""考研智库 - Streamlit Web 主界面

功能：
- 搜索问答（检索 + 推理）
- 题库练习（AI出题 + 答题）
- 知识图谱可视化
- 掌握度仪表盘
- PDF上传入库
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 确保 src 模块可导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.loader import DocumentIngestion
from src.retrieval.vectorstore import VectorStore
from src.graph.knowledge_graph import KnowledgeGraph
from src.agents.workflow import run_exam_query
from src.generator.review import ForgettingCurve, ReviewAnalyzer, Question, QuestionGenerator

# ============ 页面配置 ============
st.set_page_config(
    page_title="考研智库 - AI Exam Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
    .question-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============ 会话状态 ============
def init_session_state():
    defaults = {
        "vectorstore": None,
        "knowledge_graph": None,
        "forgetting_curve": None,
        "chat_history": [],
        "current_tab": "search",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ============ 侧边栏 ============
def render_sidebar():
    with st.sidebar:
        st.markdown("### 📚 考研智库")
        st.markdown("—— AI 驱动的考研备考系统 ——\n")

        st.markdown("### 🔧 初始化")
        if st.button("🔄 初始化向量库", use_container_width=True):
            with st.spinner("加载中..."):
                try:
                    st.session_state.vectorstore = VectorStore()
                    st.session_state.knowledge_graph = KnowledgeGraph()
                    st.session_state.forgetting_curve = ForgettingCurve()
                    st.success(f"✓ 初始化成功！向量库: {st.session_state.vectorstore.count()} chunks")
                except Exception as e:
                    st.error(f"初始化失败: {e}")

        if st.button("🗑️ 清空向量库", use_container_width=True):
            try:
                vs = VectorStore()
                vs.clear()
                st.session_state.vectorstore = None
                st.success("已清空")
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        st.markdown("### 📊 状态")
        if st.session_state.vectorstore:
            st.metric("向量库容量", f"{st.session_state.vectorstore.count()} chunks")
        if st.session_state.knowledge_graph:
            summary = st.session_state.knowledge_graph.summary()
            st.metric("图谱节点", summary["nodes"])
            st.metric("图谱边数", summary["edges"])
        if st.session_state.forgetting_curve:
            report = st.session_state.forgetting_curve.get_mastery_report()
            st.metric("待复习", report.get("due_now", 0))

        st.markdown("---")
        st.markdown("### ℹ️ 关于")
        st.caption("基于 Ollama + ChromaDB + LangGraph 构建")
        st.caption("100% 本地运行，隐私安全")


# ============ Tab 1: 搜索问答 ============
def render_search_tab():
    st.markdown('<p class="sub-header">基于 RAG + 知识图谱的深度问答</p>', unsafe_allow_html=True)

    # 科目选择
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "💬 输入你的问题",
            placeholder="例如：什么是极限的ε-N定义？",
            label_visibility="collapsed",
        )
    with col2:
        subject = st.selectbox("科目", ["数学一", "408计算机", "英语一", "政治"], label_visibility="collapsed")

    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.radio("模式", ["深度解答", "出题练习", "完整流程"], horizontal=True)

    if query and st.button("🚀 提交", use_container_width=True):
        mode_map = {"深度解答": "answer", "出题练习": "practice", "完整流程": "review"}
        with st.spinner("Agent 思考中..."):
            result = run_exam_query(query, subject=subject, mode=mode_map[mode])

        st.markdown("---")
        st.markdown("### 📝 推理结果")

        # 推理答案
        if result.reasoning_result:
            st.markdown(result.reasoning_result)

        # 引用来源
        if result.cited_sources:
            st.markdown("#### 📖 引用来源")
            for src in result.cited_sources:
                with st.expander(f"{src['source']} p.{src['page']}"):
                    st.markdown(f"> {src['text']}")

        # 生成的题目
        if result.generated_questions:
            st.markdown("#### 📝 练习题")
            for q in result.generated_questions:
                with st.expander(f"第{q['id']}题 [{q['type']}] [{q['difficulty']}]"):
                    st.markdown(f"**题目**：{q['question']}")
                    st.markdown(f"**答案**：{q['answer']}")
                    if q.get('explanation'):
                        st.markdown(f"**思路**：{q['explanation']}")


# ============ Tab 2: 题库练习 ============
def render_practice_tab():
    st.markdown('<p class="sub-header">基于遗忘曲线的智能出题系统</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        topic = st.text_input("📌 知识点", placeholder="例如：极限的ε-N定义")
    with col2:
        difficulty = st.selectbox("难度", ["基础", "中档", "困难"])

    col1, col2 = st.columns([1, 1])
    with col1:
        num_questions = st.slider("题目数量", 1, 10, 3)
    with col2:
        generate_variants = st.checkbox("生成变形题")

    if topic and st.button("📝 开始练习", use_container_width=True):
        generator = QuestionGenerator()
        subject = "408" if any(k in topic for k in ["线性表", "树", "图", "排序", "操作系统", "网络"]) else "数学一"
        st.session_state.practice_questions = generator.generate_questions(
            topic=topic,
            difficulty=difficulty,
            num_questions=num_questions,
            subject=subject,
            include_variants=generate_variants,
        )
        st.success(f"已生成 {len(st.session_state.practice_questions)} 道练习题")

    # 展示练习题（从 session 读取）
    if "practice_questions" in st.session_state and st.session_state.practice_questions:
        st.markdown("---")
        st.markdown("### 练习进行中")

        for i, q in enumerate(st.session_state.practice_questions, 1):
            st.markdown(f"""
            <div class="question-card">
                <strong>第{i}题 [{q.type}] [{q.difficulty}]</strong><br>
                {q.question}
            </div>
            """, unsafe_allow_html=True)

            ans = st.text_area(f"你的答案 #{i}", key=f"ans_{i}", placeholder="写出你的答案...")
            correct = st.radio(f"第{i}题正确与否", ["✅ 正确", "❌ 错误"], horizontal=True, key=f"correct_{i}")

            st.session_state.practice_questions[i-1].user_answer = ans
            st.session_state.practice_questions[i-1].user_correct = (correct == "✅ 正确")

        if st.button("📊 提交复盘"):
            # 分析答题
            fc = st.session_state.forgetting_curve or ForgettingCurve()
            analyzer = ReviewAnalyzer()

            analyses = []
            for q in st.session_state.practice_questions:
                current_mastery = fc.schedule.get(q.topic, {}).get("mastery", 0.5)
                analysis = analyzer.analyze_answer(
                    question=q,
                    user_answer=getattr(q, "user_answer", ""),
                    is_correct=getattr(q, "user_correct", False),
                    concept_mastery=current_mastery,
                )
                analyses.append(analysis)

            # 生成分布报告
            correct_rate = sum(1 for a in analyses if a["correct"]) / len(analyses)
            report = analyzer.generate_report(analyses, correct_rate)

            st.markdown("---")
            st.markdown(report)

            # 清空练习题
            st.session_state.practice_questions = []


# ============ Tab 3: 知识图谱 ============
def render_graph_tab():
    st.markdown('<p class="sub-header">知识点关联可视化</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        concept = st.text_input("🔍 查询概念", placeholder="输入一个知识点，如：极限")
    with col2:
        depth = st.slider("查询深度", 1, 3, 1)

    if concept and st.button("查询关联"):
        kg = st.session_state.knowledge_graph or KnowledgeGraph()

        # 查找关联概念
        related = kg.get_related_concepts(concept, depth=depth)
        paths = kg.get_learning_order(concept)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔗 直接关联")
            if related:
                for neighbor, rel in related[:10]:
                    st.markdown(f"- **{neighbor}** ({rel})")
            else:
                st.info("未找到关联概念")

        with col2:
            st.markdown("#### 📖 前置知识链")
            if paths:
                for i, p in enumerate(paths[:8], 1):
                    st.markdown(f"{i}. {p}")

    # 图谱可视化（简化文字版）
    st.markdown("---")
    st.markdown("#### 🕸️ 知识图谱概览")

    kg = st.session_state.knowledge_graph or KnowledgeGraph()
    summary = kg.summary()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("节点数", summary["nodes"])
    with col2:
        st.metric("边数", summary["edges"])
    with col3:
        categories = summary.get("categories", {})
        st.metric("类别数", len(categories))

    if summary["nodes"] > 0:
        # 简单文字图谱
        st.markdown("#### 知识点网络（文字版）")
        nodes_data = kg.to_visualization_data()
        for node in nodes_data["nodes"][:20]:
            related_nodes = [e["target"] for e in nodes_data["edges"] if e["source"] == node["id"]][:3]
            if related_nodes:
                st.markdown(f"**{node['id']}** → {', '.join(related_nodes)}")


# ============ Tab 4: 上传入库 ============
def render_ingest_tab():
    st.markdown('<p class="sub-header">上传考研资料，自动入库并构建知识图谱</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "📂 选择 PDF 或 TXT 文件",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"已选择 {len(uploaded_files)} 个文件")

        # 保存文件
        save_dir = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for f in uploaded_files:
            save_path = save_dir / f.name
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            saved_paths.append(str(save_path))
            st.success(f"✓ 已保存: {f.name}")

        if st.button("⚙️ 开始入库", use_container_width=True):
            with st.spinner("文档处理中..."):
                # 解析文档
                ingester = DocumentIngestion(upload_dir=str(save_dir))
                all_chunks = ingester.process_directory(str(save_dir))

                if all_chunks:
                    # 入库向量库
                    import pandas as pd
                    df = pd.DataFrame(all_chunks)

                    vs = VectorStore()
                    vs.add_chunks(df)

                    # 构建图谱
                    kg = KnowledgeGraph()
                    kg.auto_build_from_chunks(df)
                    kg.save()

                    st.session_state.vectorstore = vs
                    st.session_state.knowledge_graph = kg

                    st.success(f"✓ 入库完成！{len(all_chunks)} chunks，已更新向量库和图谱")
                else:
                    st.warning("未提取到任何文本，请检查文件格式")


# ============ Tab 5: 仪表盘 ============
def render_dashboard_tab():
    st.markdown('<p class="sub-header">掌握度实时监控与复习提醒</p>', unsafe_allow_html=True)

    fc = st.session_state.forgetting_curve or ForgettingCurve()
    report = fc.get_mastery_report()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总知识点", report["total_concepts"])
    with col2:
        st.metric("平均掌握度", f"{report['avg_mastery']:.0%}")
    with col3:
        st.metric("待复习", report.get("due_now", 0))
    with col4:
        st.metric("优先级高", report.get("by_priority", {}).get("high", 0))

    # 待复习列表
    st.markdown("---")
    st.markdown("### ⏰ 待复习列表")

    due = fc.get_due_concepts()
    if due:
        for item in due[:10]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item["priority"], "⚪")
                st.markdown(f"{priority_emoji} **{item['concept']}**")
            with col2:
                st.caption(f"掌握度: {item['mastery']:.0%}")
            with col3:
                if st.button("复习", key=f"review_{item['concept']}"):
                    st.info(f"跳转到「{item['concept']}」的练习题")
                    st.session_state.current_tab = "practice"
                    st.rerun()
    else:
        st.success("🎉 目前没有需要复习的内容，保持得很好！")

    # 掌握度分布图
    if report["total_concepts"] > 0:
        st.markdown("---")
        st.markdown("### 📈 掌握度分布")

        priorities = report.get("by_priority", {})
        chart_data = {
            "状态": ["已掌握 (≥70%)", "学习中 (40-70%)", "薄弱 (<40%)"],
            "数量": [
                priorities.get("low", 0),
                priorities.get("medium", 0),
                priorities.get("high", 0),
            ]
        }
        st.bar_chart(chart_data, x="状态", y="数量")


# ============ 主函数 ============
def main():
    render_sidebar()

    st.markdown('<p class="main-header">📚 考研智库</p>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🔍 搜索问答",
        "📝 题库练习",
        "🕸️ 知识图谱",
        "📤 上传入库",
        "📊 仪表盘",
    ])

    with tabs[0]:
        render_search_tab()
    with tabs[1]:
        render_practice_tab()
    with tabs[2]:
        render_graph_tab()
    with tabs[3]:
        render_ingest_tab()
    with tabs[4]:
        render_dashboard_tab()


if __name__ == "__main__":
    main()