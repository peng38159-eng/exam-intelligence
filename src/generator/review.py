"""考研智库 - AI 出题与复盘模块

基于遗忘曲线（Ebbinghaus Curve）的智能练习系统：
- 分析用户薄弱点，按遗忘周期安排复习
- 支持选择题、填空题、解答题多种题型
- 自动生成相似题、变形题
"""

import random
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import pickle


# ============ 遗忘曲线模型 ============

EBBINGHAUS_RETENTION = {
    # 时间(分钟) → 记忆保留率
    1: 0.98,
    5: 0.91,
    30: 0.81,
    60: 0.70,
    480: 0.58,
    1440: 0.44,   # 1天后
    10080: 0.36,  # 7天后
    43200: 0.20,  # 30天后
}


class ForgettingCurve:
    """遗忘曲线调度器

    核心原理：记忆会在遗忘临界点时最容易被唤醒，
    通过在临界点前安排复习，可以最大化记忆效率。
    """

    def __init__(self, review_data_path: str = "data/review_schedule.pkl"):
        self.review_data_path = Path(review_data_path)
        self.review_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.schedule = self._load_schedule()

    def _load_schedule(self) -> Dict:
        if self.review_data_path.exists():
            with open(self.review_data_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_schedule(self) -> None:
        with open(self.review_data_path, "wb") as f:
            pickle.dump(self.schedule, f)

    def calculate_next_review(
        self,
        concept: str,
        correct: bool,
        current_mastery: float = 0.5,
    ) -> Dict:
        """计算下一次复习时间

        Args:
            concept: 知识点名称
            correct: 本次答题是否正确
            current_mastery: 当前掌握度 (0.0~1.0)

        Returns:
            {
                "next_review_at": datetime,
                "interval_minutes": int,
                "new_mastery": float,
                "priority": "high/medium/low"
            }
        """
        # 记忆稳定指数：答对则提升，答错则下降
        if correct:
            mastery_delta = 0.05 * (1 - current_mastery)  # 越不熟提升越多
        else:
            mastery_delta = -0.15 * current_mastery  # 越熟退步越少

        new_mastery = max(0.0, min(1.0, current_mastery + mastery_delta))

        # 计算复习间隔（分钟）
        if not correct:
            interval = 1  # 答错立即复习
        else:
            # 答对：根据掌握度延长间隔
            # 掌握度0.9+ → 30天，0.7+ → 7天，0.5+ → 1天，<0.5 → 更频繁
            interval_map = [
                (0.90, 43200),
                (0.70, 10080),
                (0.50, 1440),
                (0.30, 480),
                (0.00, 60),
            ]
            interval = 60  # 默认
            for threshold, mins in interval_map:
                if new_mastery >= threshold:
                    interval = mins
                    break

        next_review_at = datetime.now() + timedelta(minutes=interval)

        # 优先级
        if not correct or new_mastery < 0.4:
            priority = "high"
        elif new_mastery < 0.7:
            priority = "medium"
        else:
            priority = "low"

        return {
            "next_review_at": next_review_at,
            "interval_minutes": interval,
            "new_mastery": new_mastery,
            "priority": priority,
        }

    def update_concept(self, concept: str, correct: bool, current_mastery: float) -> Dict:
        """更新某个知识点的复习计划"""
        result = self.calculate_next_review(concept, correct, current_mastery)
        self.schedule[concept] = {
            "last_reviewed": datetime.now().isoformat(),
            "next_review_at": result["next_review_at"].isoformat(),
            "mastery": result["new_mastery"],
            "interval_minutes": result["interval_minutes"],
            "priority": result["priority"],
            "review_count": self.schedule.get(concept, {}).get("review_count", 0) + 1,
        }
        self._save_schedule()
        return result

    def get_due_concepts(self, before_minutes: int = 0) -> List[Dict]:
        """获取已到期的知识点（需要复习）"""
        now = datetime.now()
        due = []
        for concept, data in self.schedule.items():
            next_review = datetime.fromisoformat(data["next_review_at"])
            if next_review <= now + timedelta(minutes=before_minutes):
                due.append({
                    "concept": concept,
                    "mastery": data["mastery"],
                    "priority": data["priority"],
                    "overdue_minutes": int((now - next_review).total_seconds() / 60),
                })
        # 按优先级和掌握度排序
        due.sort(key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 3),
            x["mastery"]
        ))
        return due

    def get_mastery_report(self) -> Dict:
        """生成整体掌握度报告"""
        if not self.schedule:
            return {"total_concepts": 0, "avg_mastery": 0.0, "by_priority": {}}

        masteries = [d["mastery"] for d in self.schedule.values()]
        priorities = [d["priority"] for d in self.schedule.values()]

        report = {
            "total_concepts": len(self.schedule),
            "avg_mastery": round(sum(masteries) / len(masteries), 3),
            "by_priority": {
                "high": sum(1 for p in priorities if p == "high"),
                "medium": sum(1 for p in priorities if p == "medium"),
                "low": sum(1 for p in priorities if p == "low"),
            },
            "due_now": len(self.get_due_concepts()),
            "due_soon": len(self.get_due_concepts(before_minutes=60)),
        }
        return report


# ============ 题目生成器 ============

@dataclass
class Question:
    id: int
    type: str  # 选择题/填空题/解答题
    difficulty: str  # 基础/中档/困难
    topic: str
    question: str
    answer: str
    explanation: str = ""
    variants: List[str] = field(default_factory=list)
    user_answer: str = ""
    user_correct: bool = False


class QuestionGenerator:
    """智能题目生成器

    支持从知识库出题、题目变形、选项干扰项生成
    """

    def __init__(self, seed: Optional[int] = None):
        self._random = random.Random(seed)
        self.question_bank = self._load_bank()

    def _load_bank(self) -> Dict:
        # 加载内置题库（可扩展为从向量库检索生成）
        return {
            "英语一": {
                "阅读理解": [
                    {
                        "type": "选择题",
                        "difficulty": "中档",
                        "template": "Read the passage and answer: What is the author's main argument regarding {topic}?",
                        "answer_template": "The author argues that {topic} …（定位主题句并结合段落首尾句推断主旨）。",
                    },
                    {
                        "type": "选择题",
                        "difficulty": "基础",
                        "template": "The word \"{word}\" in paragraph 2 most likely means ______.",
                        "answer_template": "根据上下文线索（同义复现 / 反义对比 / 举例说明）推断词义。",
                    },
                ],
                "翻译": [
                    {
                        "type": "解答题",
                        "difficulty": "困难",
                        "template": "Translate the following sentence into Chinese: \"{sentence}\"",
                        "answer_template": "拆长句 → 找主干 → 处理定语从句/状语 → 调整语序为中文表达习惯。",
                    },
                ],
                "写作": [
                    {
                        "type": "解答题",
                        "difficulty": "中档",
                        "template": "Write an essay on the topic: \"{essay_topic}\" (about 160-200 words).",
                        "answer_template": "三段式：引入主题 + 论点展开（2-3个分论点）+ 总结升华。注意使用衔接词。",
                    },
                ],
            },
            "政治": {
                "马原": [
                    {
                        "type": "解答题",
                        "difficulty": "中档",
                        "template": "用矛盾的普遍性与特殊性辩证关系原理，分析 {scenario}。",
                        "answer_template": "普遍性寓于特殊性之中，通过特殊性表现出来；二者在一定条件下相互转化。结合材料具体分析。",
                    },
                    {
                        "type": "选择题",
                        "difficulty": "基础",
                        "template": "辩证唯物主义认为，意识的本质是 ______。",
                        "answer_template": "意识是人脑对客观存在的主观映象（反映），体现了物质决定意识、意识反作用于物质的辩证关系。",
                    },
                ],
                "毛中特": [
                    {
                        "type": "解答题",
                        "difficulty": "中档",
                        "template": "结合 {policy}，论述新时代坚持和发展中国特色社会主义的基本方略。",
                        "answer_template": "从'十四个坚持'中选择若干相关要点，结合政策背景与目标展开论述。",
                    },
                ],
                "史纲": [
                    {
                        "type": "选择题",
                        "difficulty": "基础",
                        "template": "{event} 的历史意义包括哪些方面？",
                        "answer_template": "从政治、经济、思想文化、国际影响四个维度归纳。",
                    },
                ],
                "思修": [
                    {
                        "type": "解答题",
                        "difficulty": "基础",
                        "template": "结合材料，谈谈如何在 {context} 中践行社会主义核心价值观。",
                        "answer_template": "从个人层面（爱国敬业诚信友善）延展到社会/国家层面，结合具体行为。",
                    },
                ],
            },
            "数学一": {
                "极限": [
                    {
                        "type": "解答题",
                        "difficulty": "基础",
                        "template": "用 ε-N 定义证明：lim(n→∞) {a_n} = {L}",
                        "answer_template": "证明思路：化简 |{a_n} - {L}|，再由 ε 反解 N。若 n>N 时恒小于 ε，则极限成立。",
                    },
                    {
                        "type": "填空题",
                        "difficulty": "基础",
                        "template": "若对任意 ε>0，存在 N，使 n>N 时 |a_n-A|<ε，则 A 称为数列 a_n 的______。",
                        "answer_template": "极限。关键是‘任意 ε’和‘存在 N’两个量词顺序不能颠倒。",
                    },
                ],
                "连续": [
                    {
                        "type": "解答题",
                        "difficulty": "中档",
                        "template": "说明函数在点 x0 连续需要同时满足哪三个条件，并举一个不连续的反例。",
                        "answer_template": "三个条件：f(x0) 有定义、lim(x→x0)f(x) 存在、二者相等。反例可取分段函数在 x0 左右极限不等。",
                    },
                ],
                "导数": [
                    {
                        "type": "解答题",
                        "difficulty": "中档",
                        "template": "用导数定义求 f(x)=x² 在 x={x0} 处的导数。",
                        "answer_template": "f'({x0})=lim(h→0)[({x0}+h)²-{x0}²]/h=lim(h→0)(2{x0}+h)=2{x0}。",
                    },
                ],
            },
            "408": {
                "线性表": [
                    {
                        "type": "简答题",
                        "difficulty": "基础",
                        "template": "比较顺序表和链表在随机访问、插入删除、存储开销上的差异。",
                        "answer_template": "顺序表随机访问 O(1)，插入删除需移动元素；链表随机访问 O(n)，插入删除改指针即可但有指针存储开销。",
                    },
                ],
                "树": [
                    {
                        "type": "计算题",
                        "difficulty": "中档",
                        "template": "一棵二叉树有 n0 个叶结点，则度为 2 的结点数是多少？说明理由。",
                        "answer_template": "n2=n0-1。由二叉树边数 n-1 与度数和 n1+2n2 联立可得。",
                    },
                ],
            },
        }

    def generate_from_template(
        self,
        topic: str,
        difficulty: str,
        subject: str = "数学一",
        **kwargs,
    ) -> Question:
        """从模板生成题目"""
        templates = self.question_bank.get(subject, {}).get(topic, [])
        if not templates and subject != "数学一":
            templates = self.question_bank.get("数学一", {}).get(topic, [])
        if not templates:
            return self._generate_placeholder(topic, difficulty)

        template = self._random.choice(templates)
        # 所有可能的占位符提供安全默认值
        safe_kwargs = {
            "a_n": "1/n",
            "L": "0",
            "x0": "1",
            "topic": topic,
            "word": "ambiguous",
            "sentence": "The quick brown fox jumps over the lazy dog.",
            "essay_topic": topic,
            "scenario": "当前社会热点问题",
            "policy": "某项国家政策",
            "event": "某历史事件",
            "context": "日常生活",
        }
        safe_kwargs.update(kwargs)

        # 使用 format_map 而不引发 KeyError（缺失占位符保留原样）
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        question_text = template["template"].format_map(_SafeDict(safe_kwargs))
        answer_text = template["answer_template"].format_map(_SafeDict(safe_kwargs))

        return Question(
            id=self._random.randint(1000, 9999),
            type=template["type"],
            difficulty=difficulty or template["difficulty"],
            topic=topic,
            question=question_text,
            answer=answer_text,
        )

    def generate_variants(
        self,
        original: Question,
        num_variants: int = 2,
    ) -> List[Question]:
        """生成题目的变形题（更换参数/问法）"""
        variants = []
        for i in range(num_variants):
            # 简单变形：更换数列/函数形式，保持难度
            var = Question(
                id=original.id * 10 + i,
                type=original.type,
                difficulty=original.difficulty,
                topic=original.topic,
                question=f"[变形{i+1}] {original.question}",  # 简化版，实际用LLM改写
                answer=original.answer,
                explanation=f"与原题同类，考察{original.topic}的理解",
            )
            variants.append(var)
        return variants

    def generate_similar(
        self,
        topic: str,
        difficulty: str,
        exclude_ids: List[int] = None,
        subject: str = "数学一",
    ) -> Question:
        """生成同知识点相似题"""
        exclude_ids = exclude_ids or []
        for _ in range(20):
            question = self.generate_from_template(topic, difficulty, subject=subject)
            if question.id not in exclude_ids:
                return question
        return self._generate_placeholder(topic, difficulty)

    def generate_questions(
        self,
        topic: str,
        difficulty: str,
        num_questions: int = 3,
        subject: str = "数学一",
        include_variants: bool = False,
    ) -> List[Question]:
        """批量生成练习题，供 Streamlit 练习页直接调用。"""
        questions: List[Question] = []
        used_ids: List[int] = []
        for _ in range(max(1, num_questions)):
            question = self.generate_similar(topic, difficulty, used_ids, subject=subject)
            used_ids.append(question.id)
            questions.append(question)
            if include_variants:
                questions.extend(self.generate_variants(question, num_variants=1))
        return questions[: max(1, num_questions)]

    def _generate_placeholder(self, topic: str, difficulty: str) -> Question:
        return Question(
            id=self._random.randint(1000, 9999),
            type="解答题",
            difficulty=difficulty,
            topic=topic,
            question=f"请解答关于「{topic}」的{difficulty}难度题目",
            answer="（请参考教材或向教师请教）",
        )


# ============ 复盘分析器 ============

class ReviewAnalyzer:
    """答题复盘分析器"""

    def __init__(self):
        self.forgetting_curve = ForgettingCurve()

    def analyze_answer(
        self,
        question: Question,
        user_answer: str,
        is_correct: bool,
        concept_mastery: float = 0.5,
    ) -> Dict:
        """分析单次答题，更新掌握度"""
        # 更新遗忘曲线
        update_result = self.forgetting_curve.update_concept(
            concept=question.topic,
            correct=is_correct,
            current_mastery=concept_mastery,
        )

        analysis = {
            "question_id": question.id,
            "topic": question.topic,
            "correct": is_correct,
            "mastery_change": update_result["new_mastery"] - concept_mastery,
            "new_mastery": update_result["new_mastery"],
            "next_review_in_minutes": update_result["interval_minutes"],
            "priority": update_result["priority"],
        }
        return analysis

    def generate_report(
        self,
        analyses: List[Dict],
        overall_correct_rate: float,
    ) -> str:
        """生成复盘报告"""
        topics_mastered = [a for a in analyses if a["new_mastery"] >= 0.8]
        topics_weak = [a for a in analyses if a["new_mastery"] < 0.5]

        report = f"""## 复盘报告

**整体正确率：{overall_correct_rate:.1%}**

### 已掌握知识点 ({len(topics_mastered)}个)
{', '.join([a['topic'] for a in topics_mastered]) if topics_mastered else '无'}

### 薄弱知识点 ({len(topics_weak)}个) — 建议优先复习
"""
        for a in topics_weak:
            report += f"- **{a['topic']}**：掌握度 {a['new_mastery']:.0%}，下次复习 {a['next_review_in_minutes']}分钟后\n"

        report += "\n### 复习建议\n"
        if topics_weak:
            report += f"1. 优先攻克：{', '.join([a['topic'] for a in topics_weak[:3]])}\n"
        report += "2. 利用碎片时间回顾已掌握知识点，防止遗忘\n"
        report += "3. 建议配合教材进行专项练习\n"

        return report


if __name__ == "__main__":
    # python -m src.generator.review
    fc = ForgettingCurve()
    print(f"当前复习计划: {fc.get_mastery_report()}")