"""考研智库 - 知识图谱模块

基于 NetworkX 构建知识点概念图谱，支持实体识别、关系抽取和图上路径推理
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx


class KnowledgeGraph:
    """知识点知识图谱：基于 NetworkX 实现"""

    def __init__(self, graph_path: str = "data/graph/knowledge_graph.gml", verbose: bool = True):
        self.graph_path = Path(graph_path)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        pkl_path = self.graph_path.with_suffix(".pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.graph = pickle.load(f)
            if self.verbose:
                print(f"✓ 知识图谱已加载: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
        elif self.graph_path.exists():
            self.graph = nx.read_gml(str(self.graph_path))
            if self.verbose:
                print(f"✓ 知识图谱已加载: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")
        else:
            self.graph = nx.DiGraph()
            if self.verbose:
                print("新建空知识图谱")

        # 预定义关系类型
        self.RELATION_TYPES = {
            "包含": "包含",
            "前置": "前置知识",
            "推导": "推导关系",
            "对比": "对比关系",
            "应用": "应用场景",
            "等价": "等价概念",
            "属从": "从属关系",
        }

    def add_concept(self, concept: str, category: str = "通用", **attrs) -> None:
        """添加概念节点"""
        if concept not in self.graph:
            self.graph.add_node(
                concept,
                category=category,
                mastery_level=0.0,
                review_count=0,
                last_reviewed=None,
                **attrs
            )

    def add_relation(
        self,
        from_node: str,
        to_node: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """添加概念间关系边"""
        if from_node not in self.graph:
            self.add_concept(from_node)
        if to_node not in self.graph:
            self.add_concept(to_node)

        edge_data = {
            "relation": relation_type,
            "weight": weight,
        }
        self.graph.add_edge(from_node, to_node, **edge_data)

    def add_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取概念实体（简单启发式提取）

        实际项目中应替换为 spaCy NER 或 LLM 抽取
        这里使用关键词模式匹配识别教材常见概念词
        """
        # 常见考研概念模式
        patterns = [
            "第[一二三四五六七八九十\\d]+章",
            "[\\u4e00-\\u9fa5]+定理",
            "[\\u4e00-\\u9fa5]+定律",
            "[\\u4e00-\\u9fa5]+公式",
            "[\\u4e00-\\u9fa5]+原理",
            "[\\u4e00-\\u9fa5]+法则",
            "[\\u4e00-\\u9fa5]+定义",
            "[\\u4e00-\\u9fa5]+性质",
            "[\\u4e00-\\u9fa5]+推论",
            "极限|连续|导数|微分|积分|级数|行列式|矩阵|向量|概率",
            "函数|数列|多项式|三角函数|指数函数|对数函数",
            "极限|微分|积分|微分方程|多元函数|重积分|曲线积分|曲面积分",
        ]

        import re
        found = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                concept = m.strip()
                if len(concept) >= 2:
                    self.add_concept(concept, category="知识点")
                    found.append(concept)

        return list(set(found))

    def auto_build_from_chunks(self, df) -> None:
        """从已入库的 chunks 自动构建图谱

        思路：相邻 chunk 的标题/关键词视为关联概念
        """
        prev_concepts = []
        for _, row in df.iterrows():
            text = row.get("text", "")
            concepts = self.add_entities_from_text(text)

            # 同一章节内相邻 chunk 的概念互相连接
            for c in concepts:
                for prev in prev_concepts:
                    if c != prev:
                        # 默认关系为"相关"
                        self.add_relation(prev, c, relation_type="相关", weight=0.5)

            prev_concepts = concepts[-5:]  # 滑动窗口保留最近5个概念

        if self.verbose:
            print(f"图谱构建完成: {len(self.graph.nodes)} 节点, {len(self.graph.edges)} 边")

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> List[List[str]]:
        """查找两个概念间的最短路径（用于多跳推理问答）"""
        try:
            paths = list(nx.all_shortest_paths(
                self.graph, source=source, target=target, cutoff=max_depth
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_related_concepts(self, concept: str, depth: int = 1) -> List[Tuple[str, str]]:
        """获取某概念的相关概念及其关系

        返回指定深度内的所有相邻概念（出边 + 入边），
        按深度从小到大排序。
        """
        if concept not in self.graph:
            return []

        related = []
        # 出边（当前概念 → 邻居）—— 不限制 depth
        for neighbor in self.graph.successors(concept):
            edge_data = self.graph.get_edge_data(concept, neighbor)
            rel_type = edge_data.get("relation", "相关") if edge_data else "相关"
            related.append((neighbor, rel_type))

        # 入边（前驱 → 当前概念），用 "前驱" 标记方向
        for neighbor in self.graph.predecessors(concept):
            if neighbor not in {r[0] for r in related}:
                edge_data = self.graph.get_edge_data(neighbor, concept)
                rel_type = edge_data.get("relation", "相关") if edge_data else "相关"
                related.append((neighbor, f"前驱·{rel_type}"))

        # 若 depth>1，按 BFS 扩展
        if depth > 1:
            extended = list(related)
            # BFS 从 concept 出发搜 depth 步
            for _ in range(depth - 1):
                frontier: set[str] = set()
                for neighbor, _rel in extended:
                    for succ in self.graph.successors(neighbor):
                        frontier.add(succ)
                    for pred in self.graph.predecessors(neighbor):
                        frontier.add(pred)
                frontier.discard(concept)
                for node in frontier:
                    if node not in {r[0] for r in related} and node not in {r[0] for r in extended}:
                        extended.append((node, "间接关联"))
                related = extended

        return related[:20]  # 最多返回20个

    def get_learning_order(self, concept: str) -> List[str]:
        """获取某概念的前置知识链（拓扑排序）"""
        if concept not in self.graph:
            return [concept]

        # 找所有入度为0的节点（最基础知识）
        ancestors = nx.ancestors(self.graph, concept)
        if not ancestors:
            return [concept]

        # 拓扑排序
        subgraph = self.graph.subgraph(ancestors | {concept})
        try:
            order = list(nx.topological_sort(subgraph))
            return order
        except nx.NetworkXError:
            return list(ancestors) + [concept]

    def update_mastery(self, concept: str, mastery_delta: float) -> None:
        """更新概念掌握度"""
        if concept in self.graph.nodes:
            current = self.graph.nodes[concept].get("mastery_level", 0.0)
            new_level = max(0.0, min(1.0, current + mastery_delta))
            self.graph.nodes[concept]["mastery_level"] = new_level
            self.graph.nodes[concept]["review_count"] += 1

    def save(self) -> None:
        """保存图谱到文件"""
        # NetworkX GML 不支持某些中文，用不同后缀
        with open(self.graph_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self.graph, f)
        if self.verbose:
            print(f"✓ 图谱已保存至 {self.graph_path.with_suffix('.pkl')}")

    def load(self) -> None:
        """从文件加载图谱"""
        pkl_path = self.graph_path.with_suffix(".pkl")
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.graph = pickle.load(f)
            if self.verbose:
                print(f"✓ 图谱已加载: {len(self.graph.nodes)} 节点")

    def to_visualization_data(self) -> Dict:
        """导出为 D3.js / PyVis 可视化格式"""
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": node,
                "category": attrs.get("category", "通用"),
                "mastery": attrs.get("mastery_level", 0.0),
            })

        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": attrs.get("relation", "相关"),
                "weight": attrs.get("weight", 1.0),
            })

        return {"nodes": nodes, "edges": edges}

    def summary(self) -> Dict:
        categories: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            category = data.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1
        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "categories": categories,
        }


if __name__ == "__main__":
    kg = KnowledgeGraph()
    print(f"当前图谱: {kg.summary()}")