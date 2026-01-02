"""
Гибридный Knowledge Retriever.

Стратегия:
1. Быстрый поиск по ключевым словам (детерминированный)
2. Fallback на эмбеддинги если не нашли (опционально)
3. Возвращает только релевантные факты
"""

import re
from typing import List, Optional, Tuple
from .base import KnowledgeSection
from .data import WIPON_KNOWLEDGE


# Маппинг интентов на категории
INTENT_TO_CATEGORY = {
    "price_question": ["pricing"],
    "question_features": ["features", "products"],
    "question_integrations": ["integrations"],
    "objection_competitor": ["competitors", "benefits"],
    "objection_price": ["pricing", "benefits"],
    "agreement": ["pricing", "support", "contacts"],
    "greeting": [],  # Не нужны факты
    "rejection": [],  # Не нужны факты
}


class KnowledgeRetriever:
    """Гибридный retriever: keywords + embeddings (опционально)"""

    def __init__(self, use_embeddings: bool = False):
        self.kb = WIPON_KNOWLEDGE
        self.use_embeddings = use_embeddings
        self.embedder = None
        self.np = None

        if use_embeddings:
            self._init_embeddings()

    def _init_embeddings(self):
        """Инициализация модели эмбеддингов (опционально)"""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.np = np
            # Маленькая русская модель ~100MB
            self.embedder = SentenceTransformer('cointegrated/rubert-tiny2')

            # Индексируем все секции
            texts = [s.facts for s in self.kb.sections]
            embeddings = self.embedder.encode(texts)
            for i, section in enumerate(self.kb.sections):
                section.embedding = embeddings[i].tolist()

            print(f"[KnowledgeRetriever] Indexed {len(texts)} sections with embeddings")
        except ImportError:
            print("[KnowledgeRetriever] sentence-transformers not installed, using keywords only")
            self.use_embeddings = False

    def retrieve(
        self,
        message: str,
        intent: str = None,
        state: str = None,
        top_k: int = 2
    ) -> str:
        """
        Найти релевантные факты.

        Args:
            message: Сообщение пользователя
            intent: Классифицированный интент
            state: Текущее состояние state machine
            top_k: Максимум секций для возврата

        Returns:
            Строка с фактами или пустая строка
        """
        message_lower = message.lower()

        # Шаг 1: Сужаем область поиска по интенту
        categories = INTENT_TO_CATEGORY.get(intent, [])
        if categories:
            candidate_sections = []
            for cat in categories:
                candidate_sections.extend(self.kb.get_by_category(cat))
        else:
            candidate_sections = self.kb.sections

        if not candidate_sections:
            return ""

        # Шаг 2: Поиск по ключевым словам
        scored_sections = self._keyword_search(message_lower, candidate_sections)

        # Шаг 3: Если не нашли и есть эмбеддинги — семантический поиск
        if not scored_sections and self.use_embeddings:
            scored_sections = self._semantic_search(message, candidate_sections, top_k)

        # Шаг 4: Сортируем по score и priority
        scored_sections.sort(key=lambda x: (x[0], x[1].priority), reverse=True)

        # Шаг 5: Возвращаем топ-K
        results = []
        for score, section in scored_sections[:top_k]:
            if score > 0:
                results.append(section.facts.strip())

        return "\n\n---\n\n".join(results) if results else ""

    def _keyword_search(
        self,
        message_lower: str,
        sections: List[KnowledgeSection]
    ) -> List[Tuple[float, KnowledgeSection]]:
        """Поиск по ключевым словам"""
        results = []

        for section in sections:
            score = 0.0
            for keyword in section.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in message_lower:
                    score += 1
                    # Бонус за точное совпадение слова
                    if re.search(rf'\b{re.escape(keyword_lower)}\b', message_lower, re.IGNORECASE):
                        score += 0.5

            if score > 0:
                results.append((score, section))

        return results

    def _semantic_search(
        self,
        message: str,
        sections: List[KnowledgeSection],
        top_k: int
    ) -> List[Tuple[float, KnowledgeSection]]:
        """Семантический поиск по эмбеддингам"""
        if not self.embedder or not self.np:
            return []

        query_emb = self.embedder.encode(message)
        results = []

        for section in sections:
            if section.embedding:
                # Косинусное сходство
                section_emb = self.np.array(section.embedding)
                score = self.np.dot(query_emb, section_emb) / (
                    self.np.linalg.norm(query_emb) * self.np.linalg.norm(section_emb)
                )
                if score > 0.4:  # Порог релевантности
                    results.append((float(score), section))

        return sorted(results, key=lambda x: x[0], reverse=True)[:top_k]

    def get_company_info(self) -> str:
        """Получить базовую информацию о компании"""
        return f"{self.kb.company_name}: {self.kb.company_description}"


# Singleton для использования в generator.py
_retriever = None


def get_retriever(use_embeddings: bool = False) -> KnowledgeRetriever:
    """Получить инстанс retriever'а"""
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever(use_embeddings=use_embeddings)
    return _retriever
