from datetime import datetime, timedelta
from pathlib import Path

from src.ingestion.loader import DocumentIngestion
from src.graph.knowledge_graph import KnowledgeGraph
from src.generator.review import ForgettingCurve, QuestionGenerator
from src.agents.workflow import run_exam_query, AgentState


def test_chunk_pages_makes_progress_with_overlap():
    ingester = DocumentIngestion(chunk_size=10, chunk_overlap=3)
    chunks = ingester.chunk_pages([
        {"page": 1, "source": "demo.txt", "text": "abcdefghijklmnopqrstuvwxyz"}
    ])
    assert chunks
    assert all(chunk["text"] for chunk in chunks)
    assert len(chunks) < 10


def test_knowledge_graph_save_load_and_depth(tmp_path: Path):
    graph_path = tmp_path / "knowledge_graph.gml"
    kg = KnowledgeGraph(graph_path=str(graph_path), verbose=False)
    kg.add_relation("极限", "连续", "前置")
    kg.add_relation("连续", "导数", "前置")
    kg.save()

    loaded = KnowledgeGraph(graph_path=str(graph_path), verbose=False)
    assert loaded.summary()["nodes"] == 3
    assert ("连续", "前置") in loaded.get_related_concepts("极限", depth=1)
    assert ("导数", "前置") not in loaded.get_related_concepts("极限", depth=1)
    assert ("导数", "前置") in loaded.get_related_concepts("极限", depth=2)


def test_question_generator_templates_are_real_questions():
    generator = QuestionGenerator(seed=1)
    for subject, topics in generator.question_bank.items():
        for topic in topics:
            question = generator.generate_from_template(topic, "基础", subject=subject)
            assert question.question
            assert "..." not in question.question
            assert question.answer


def test_forgetting_curve_due_sorting(tmp_path: Path):
    fc = ForgettingCurve(review_data_path=str(tmp_path / "review.pkl"))
    now = datetime.now()
    fc.schedule = {
        "弱点": {
            "next_review_at": (now - timedelta(minutes=30)).isoformat(),
            "mastery": 0.2,
            "priority": "high",
        },
        "一般": {
            "next_review_at": (now - timedelta(minutes=10)).isoformat(),
            "mastery": 0.6,
            "priority": "medium",
        },
    }
    due = fc.get_due_concepts()
    assert [item["concept"] for item in due] == ["弱点", "一般"]


def test_run_exam_query_returns_agent_state():
    result = run_exam_query("什么是极限？", subject="数学一", mode="practice")
    assert isinstance(result, AgentState)
    assert result.reasoning_result
    assert result.generated_questions
    assert not result.error
