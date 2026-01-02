"""
Модуль базы знаний Wipon.
"""

from .base import KnowledgeBase, KnowledgeSection
from .retriever import KnowledgeRetriever, get_retriever

__all__ = [
    "KnowledgeBase",
    "KnowledgeSection",
    "KnowledgeRetriever",
    "get_retriever",
]
