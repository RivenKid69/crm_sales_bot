#!/usr/bin/env python3
"""
ПОЛНЫЙ СТРЕСС-ТЕСТ ЧАТБОТА WIPON

Тестирует ВСЕ компоненты системы:
- Intent classification
- State machine transitions
- Knowledge retrieval
- Response generation
- Pain extraction
- Полные диалоговые сценарии

Запуск:
    python scripts/full_bot_stress_test.py
    python scripts/full_bot_stress_test.py --verbose
    python scripts/full_bot_stress_test.py --fix  # автоисправление
"""

import sys
import os
import json
import asyncio
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Импорты компонентов бота
try:
    from src.knowledge.retriever import KnowledgeRetriever, get_retriever
    from src.knowledge.data import WIPON_KNOWLEDGE
    from src.bot.intent_classifier import classify_intent
    from src.bot.state_machine import ConversationStateMachine, ConversationState
    from src.bot.pain_extractor import extract_pains
    HAS_BOT_COMPONENTS = True
except ImportError as e:
    print(f"⚠️  Не все компоненты доступны: {e}")
    HAS_BOT_COMPONENTS = False


# =============================================================================
# ТИПЫ ПЕРСОН (как реальные пользователи общаются)
# =============================================================================

class PersonaType(Enum):
    """Типы персон для тестирования"""
    FORMAL = "formal"           # Вежливый, формальный
    CASUAL = "casual"           # Разговорный, простой
    AGGRESSIVE = "aggressive"   # Агрессивный, требовательный
    CONFUSED = "confused"       # Растерянный, много вопросов
    SKEPTIC = "skeptic"         # Скептик, много возражений
    IMPATIENT = "impatient"     # Нетерпеливый, короткие сообщения
    TECHIE = "techie"           # Технарь, много терминов
    NEWBIE = "newbie"           # Новичок, ничего не знает
    PRICE_FOCUSED = "price"     # Фокус только на цене
    SILENT = "silent"           # Молчун, односложные ответы


# =============================================================================
# ДИАЛОГОВЫЕ СЦЕНАРИИ
# =============================================================================

@dataclass
class DialogTurn:
    """Один ход диалога"""
    user_message: str
    expected_intent: Optional[str] = None
    expected_state: Optional[str] = None
    expected_topics: List[str] = field(default_factory=list)
    should_contain: List[str] = field(default_factory=list)
    should_not_contain: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DialogScenario:
    """Полный сценарий диалога"""
    name: str
    persona: PersonaType
    turns: List[DialogTurn]
    description: str = ""


# =============================================================================
# СЦЕНАРИИ ДИАЛОГОВ
# =============================================================================

DIALOG_SCENARIOS = [
    # =========================================================================
    # СЦЕНАРИЙ 1: Классическая продажа (формальный клиент)
    # =========================================================================
    DialogScenario(
        name="classic_sale_formal",
        persona=PersonaType.FORMAL,
        description="Классический путь: приветствие → вопросы → цена → согласие",
        turns=[
            DialogTurn(
                user_message="Добрый день! Меня интересует автоматизация торговли.",
                expected_intent="greeting",
                expected_state="greeting",
                description="Формальное приветствие"
            ),
            DialogTurn(
                user_message="Расскажите, пожалуйста, какие решения вы предлагаете?",
                expected_intent="question_features",
                expected_topics=["overview"],
                should_contain=["Wipon"],
                description="Вопрос о продуктах"
            ),
            DialogTurn(
                user_message="У меня магазин продуктов, 2 точки. Что подойдёт?",
                expected_intent="question_features",
                expected_topics=["who_is_it_for", "tariffs_detailed"],
                description="Уточнение потребности"
            ),
            DialogTurn(
                user_message="Сколько это будет стоить?",
                expected_intent="price_question",
                expected_topics=["tariffs", "tariffs_detailed"],
                should_contain=["тариф", "Lite"],
                description="Вопрос о цене"
            ),
            DialogTurn(
                user_message="Хорошо, давайте попробуем. Как начать?",
                expected_intent="agreement",
                should_contain=["заявк", "контакт"],
                description="Согласие"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 2: Скептик с возражениями
    # =========================================================================
    DialogScenario(
        name="skeptic_objections",
        persona=PersonaType.SKEPTIC,
        description="Скептик: много возражений, сомнения",
        turns=[
            DialogTurn(
                user_message="Ну и что это за программа?",
                expected_intent="question_features",
                description="Скептическое начало"
            ),
            DialogTurn(
                user_message="А чем вы лучше iiko?",
                expected_intent="objection_competitor",
                expected_topics=["vs_others"],
                should_contain=["iiko", "дешевле"],
                description="Сравнение с конкурентом"
            ),
            DialogTurn(
                user_message="Дорого наверное...",
                expected_intent="objection_price",
                expected_topics=["tariffs", "free"],
                should_contain=["бесплатн"],
                description="Возражение о цене"
            ),
            DialogTurn(
                user_message="А данные точно не потеряются?",
                expected_intent="question_features",
                expected_topics=["data_protection"],
                should_contain=["бэкап", "шифрован", "резерв"],
                description="Сомнение о безопасности"
            ),
            DialogTurn(
                user_message="А если сервера упадут?",
                expected_intent="question_features",
                expected_topics=["data_protection"],
                should_contain=["SLA", "99"],
                description="Сомнение о надёжности"
            ),
            DialogTurn(
                user_message="Ладно, убедили. Оставлю заявку.",
                expected_intent="agreement",
                description="Согласие после возражений"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 3: Нетерпеливый клиент (короткие сообщения)
    # =========================================================================
    DialogScenario(
        name="impatient_short",
        persona=PersonaType.IMPATIENT,
        description="Нетерпеливый: очень короткие сообщения",
        turns=[
            DialogTurn(
                user_message="прайс",
                expected_intent="price_question",
                expected_topics=["tariffs"],
                description="Одно слово - прайс"
            ),
            DialogTurn(
                user_message="касса",
                expected_intent="question_features",
                expected_topics=["wipon_kassa"],
                description="Одно слово - касса"
            ),
            DialogTurn(
                user_message="бесплатно?",
                expected_intent="price_question",
                expected_topics=["free"],
                description="Одно слово - бесплатно"
            ),
            DialogTurn(
                user_message="каспи",
                expected_intent="question_integrations",
                expected_topics=["marketplaces"],
                description="Одно слово - каспи"
            ),
            DialogTurn(
                user_message="ок",
                expected_intent="agreement",
                description="Короткое согласие"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 4: Технарь (много технических вопросов)
    # =========================================================================
    DialogScenario(
        name="techie_detailed",
        persona=PersonaType.TECHIE,
        description="Технарь: технические вопросы",
        turns=[
            DialogTurn(
                user_message="Какие API endpoints предоставляете?",
                expected_intent="question_features",
                expected_topics=["common_questions"],
                should_contain=["API", "Pro"],
                description="Вопрос об API"
            ),
            DialogTurn(
                user_message="Интеграция с 1С есть? REST или SOAP?",
                expected_intent="question_integrations",
                expected_topics=["1c"],
                should_contain=["1С"],
                description="Вопрос о 1С"
            ),
            DialogTurn(
                user_message="Какие сканеры поддерживаете? HID или serial?",
                expected_intent="question_features",
                expected_topics=["hardware"],
                should_contain=["HID", "USB"],
                description="Вопрос об оборудовании"
            ),
            DialogTurn(
                user_message="Минимальные системные требования для desktop версии?",
                expected_intent="question_features",
                expected_topics=["system_requirements"],
                should_contain=["Windows", "ГБ"],
                description="Системные требования"
            ),
            DialogTurn(
                user_message="TLS какой версии? Шифрование данных?",
                expected_intent="question_features",
                expected_topics=["data_protection"],
                should_contain=["TLS", "AES"],
                description="Безопасность"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 5: Новичок (ничего не понимает)
    # =========================================================================
    DialogScenario(
        name="newbie_confused",
        persona=PersonaType.NEWBIE,
        description="Новичок: базовые вопросы",
        turns=[
            DialogTurn(
                user_message="Здравствуйте, я вообще не понимаю как это работает",
                expected_intent="greeting",
                description="Растерянное приветствие"
            ),
            DialogTurn(
                user_message="Мне нужна касса? Или ККМ? В чём разница?",
                expected_intent="question_features",
                expected_topics=["wipon_kassa", "common_questions"],
                description="Базовый вопрос о кассе"
            ),
            DialogTurn(
                user_message="А без интернета работать можно?",
                expected_intent="question_features",
                expected_topics=["common_questions"],
                should_contain=["офлайн", "интернет"],
                description="Вопрос об офлайне"
            ),
            DialogTurn(
                user_message="Я ИП на упрощёнке, это подойдёт?",
                expected_intent="question_features",
                expected_topics=["who_is_it_for", "wipon_tis"],
                should_contain=["ИП", "упрощ"],
                description="Вопрос о соответствии"
            ),
            DialogTurn(
                user_message="А научите пользоваться?",
                expected_intent="question_features",
                expected_topics=["help"],
                should_contain=["обучен"],
                description="Вопрос об обучении"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 6: Алкогольный магазин (специфика)
    # =========================================================================
    DialogScenario(
        name="alcohol_shop",
        persona=PersonaType.CASUAL,
        description="Владелец алкогольного магазина",
        turns=[
            DialogTurn(
                user_message="У меня магазин алкоголя, нужна проверка марок",
                expected_intent="question_features",
                expected_topics=["wipon_pro"],
                should_contain=["УКМ", "алкоголь", "Pro"],
                description="Запрос на проверку алкоголя"
            ),
            DialogTurn(
                user_message="какой штраф если не проверять?",
                expected_intent="question_features",
                expected_topics=["wipon_pro"],
                should_contain=["штраф", "МРП"],
                description="Вопрос о штрафах"
            ),
            DialogTurn(
                user_message="сколько стоит wipon pro?",
                expected_intent="price_question",
                expected_topics=["wipon_pro_pricing"],
                should_contain=["12,000", "15,000"],
                description="Цена Wipon Pro"
            ),
            DialogTurn(
                user_message="на телефон поставить можно?",
                expected_intent="question_features",
                expected_topics=["wipon_pro", "mobile_app"],
                should_contain=["смартфон", "телефон"],
                description="Мобильная версия"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 7: Миграция с другой системы
    # =========================================================================
    DialogScenario(
        name="migration_scenario",
        persona=PersonaType.FORMAL,
        description="Переход с другой системы",
        turns=[
            DialogTurn(
                user_message="Мы сейчас на 1С, хотим перейти на облако",
                expected_intent="question_features",
                expected_topics=["switching", "1c"],
                should_contain=["перенос", "импорт"],
                description="Запрос на миграцию"
            ),
            DialogTurn(
                user_message="Данные можно перенести? Товары, остатки?",
                expected_intent="question_features",
                expected_topics=["switching"],
                should_contain=["импорт", "Excel"],
                description="Вопрос о переносе данных"
            ),
            DialogTurn(
                user_message="Сколько времени займёт переход?",
                expected_intent="question_features",
                expected_topics=["switching"],
                should_contain=["1-3 дня", "день"],
                description="Сроки миграции"
            ),
            DialogTurn(
                user_message="Поможете с настройкой?",
                expected_intent="question_features",
                expected_topics=["help", "switching"],
                should_contain=["помощь", "поддержк"],
                description="Помощь с переходом"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 8: Партнёрство
    # =========================================================================
    DialogScenario(
        name="partnership_inquiry",
        persona=PersonaType.FORMAL,
        description="Запрос на партнёрство",
        turns=[
            DialogTurn(
                user_message="Здравствуйте, хочу стать вашим партнёром",
                expected_intent="question_features",
                expected_topics=["partners"],
                should_contain=["партнёр"],
                description="Запрос на партнёрство"
            ),
            DialogTurn(
                user_message="Какие условия для реселлеров?",
                expected_intent="question_features",
                expected_topics=["partners"],
                should_contain=["реселлер", "комисси"],
                description="Условия партнёрства"
            ),
            DialogTurn(
                user_message="Как оформить сотрудничество?",
                expected_intent="question_features",
                expected_topics=["partners"],
                should_contain=["заявк", "wipon.kz"],
                description="Оформление партнёрства"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 9: Агрессивный клиент
    # =========================================================================
    DialogScenario(
        name="aggressive_client",
        persona=PersonaType.AGGRESSIVE,
        description="Агрессивный клиент, требования",
        turns=[
            DialogTurn(
                user_message="Почему так дорого?! У конкурентов дешевле!",
                expected_intent="objection_price",
                expected_topics=["tariffs", "free", "vs_others"],
                description="Агрессивное возражение о цене"
            ),
            DialogTurn(
                user_message="Мне СРОЧНО нужна касса! Сегодня!",
                expected_intent="question_features",
                expected_topics=["wipon_kassa"],
                description="Срочный запрос"
            ),
            DialogTurn(
                user_message="Ваш support вообще отвечает?!",
                expected_intent="question_features",
                expected_topics=["help"],
                should_contain=["поддержк"],
                description="Жалоба на поддержку"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 10: Разговорный стиль с опечатками
    # =========================================================================
    DialogScenario(
        name="casual_typos",
        persona=PersonaType.CASUAL,
        description="Разговорный стиль, опечатки, сленг",
        turns=[
            DialogTurn(
                user_message="прив, скока стоит прога?",
                expected_intent="price_question",
                expected_topics=["tariffs"],
                description="Приветствие + цена с опечатками"
            ),
            DialogTurn(
                user_message="а каспы подключить можно?",
                expected_intent="question_integrations",
                expected_topics=["marketplaces"],
                description="Kaspi с опечаткой"
            ),
            DialogTurn(
                user_message="скинь прас лист плз",
                expected_intent="price_question",
                expected_topics=["tariffs"],
                description="Прайс с опечатками"
            ),
            DialogTurn(
                user_message="ну ок, погнали",
                expected_intent="agreement",
                description="Сленговое согласие"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 11: Смешанный язык (русский + английский)
    # =========================================================================
    DialogScenario(
        name="mixed_language",
        persona=PersonaType.TECHIE,
        description="Смешанный русский + английский",
        turns=[
            DialogTurn(
                user_message="Hi, нужна POS система для retail",
                expected_intent="question_features",
                expected_topics=["overview", "wipon_kassa"],
                description="Англ+рус приветствие"
            ),
            DialogTurn(
                user_message="integration с kaspi marketplace есть?",
                expected_intent="question_integrations",
                expected_topics=["marketplaces"],
                description="Англ+рус интеграция"
            ),
            DialogTurn(
                user_message="price list скиньте please",
                expected_intent="price_question",
                expected_topics=["tariffs"],
                description="Англ+рус прайс"
            ),
            DialogTurn(
                user_message="ok, let's try. как оплатить?",
                expected_intent="agreement",
                description="Англ+рус согласие"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 12: Региональный клиент
    # =========================================================================
    DialogScenario(
        name="regional_client",
        persona=PersonaType.CASUAL,
        description="Клиент из региона",
        turns=[
            DialogTurn(
                user_message="Вы в Караганде работаете?",
                expected_intent="question_features",
                expected_topics=["coverage"],
                should_contain=["Караганд", "Казахстан"],
                description="Вопрос о регионе"
            ),
            DialogTurn(
                user_message="Оборудование доставите?",
                expected_intent="question_features",
                expected_topics=["coverage"],
                should_contain=["доставк"],
                description="Вопрос о доставке"
            ),
            DialogTurn(
                user_message="А если что-то сломается, кто чинит?",
                expected_intent="question_features",
                expected_topics=["help"],
                should_contain=["поддержк", "удалённ"],
                description="Вопрос о сервисе"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 13: Отказ и возврат
    # =========================================================================
    DialogScenario(
        name="rejection_return",
        persona=PersonaType.SKEPTIC,
        description="Отказ, потом возврат",
        turns=[
            DialogTurn(
                user_message="Сколько стоит?",
                expected_intent="price_question",
                expected_topics=["tariffs"],
                description="Вопрос о цене"
            ),
            DialogTurn(
                user_message="Нет, дорого, не интересно",
                expected_intent="rejection",
                description="Отказ"
            ),
            DialogTurn(
                user_message="Хотя подождите, а бесплатная версия есть?",
                expected_intent="price_question",
                expected_topics=["free"],
                description="Возврат к разговору"
            ),
            DialogTurn(
                user_message="Ладно, попробую бесплатную",
                expected_intent="agreement",
                description="Согласие на бесплатную версию"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 14: Много вопросов подряд
    # =========================================================================
    DialogScenario(
        name="rapid_questions",
        persona=PersonaType.IMPATIENT,
        description="Много вопросов без ожидания ответа",
        turns=[
            DialogTurn(
                user_message="Сколько стоит? Есть интеграция с каспи? Работает офлайн?",
                expected_intent="price_question",  # или question_features
                expected_topics=["tariffs", "marketplaces", "common_questions"],
                description="Три вопроса в одном"
            ),
            DialogTurn(
                user_message="А на айфон есть? И принтер чеков подключить можно? Сканер какой нужен?",
                expected_intent="question_features",
                expected_topics=["mobile_app", "hardware"],
                description="Ещё три вопроса"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 15: Длинная история перед вопросом
    # =========================================================================
    DialogScenario(
        name="long_story",
        persona=PersonaType.CONFUSED,
        description="Длинная история, потом вопрос",
        turns=[
            DialogTurn(
                user_message="""Здравствуйте, у меня такая ситуация. Я открыл небольшой магазин
                продуктов в Алматы полгода назад. Сначала вёл учёт в Excel, но это очень неудобно.
                Потом попробовал 1С, но там нужен программист для настройки, это дорого.
                Сейчас ищу что-то простое, чтобы можно было самому разобраться.
                У меня 2 точки, планирую открыть третью. Нужна касса, учёт товаров,
                желательно интеграция с Kaspi. Что посоветуете?""",
                expected_intent="question_features",
                expected_topics=["overview", "tariffs_detailed", "marketplaces"],
                description="Длинная история + вопрос"
            ),
            DialogTurn(
                user_message="И ещё важно чтобы можно было с телефона смотреть продажи",
                expected_intent="question_features",
                expected_topics=["mobile_app"],
                description="Дополнение к истории"
            ),
        ]
    ),

    # =========================================================================
    # СЦЕНАРИЙ 16: Только эмоции
    # =========================================================================
    DialogScenario(
        name="emotional_only",
        persona=PersonaType.AGGRESSIVE,
        description="Эмоциональные сообщения",
        turns=[
            DialogTurn(
                user_message="ПОМОГИТЕ!!!",
                expected_intent="greeting",  # или question_features
                description="Крик о помощи"
            ),
            DialogTurn(
                user_message="Срочно нужна касса, проверка сегодня!",
                expected_intent="question_features",
                expected_topics=["wipon_kassa"],
                description="Срочный запрос"
            ),
            DialogTurn(
                user_message="Спасибо огромное, вы меня спасли!",
                expected_intent="agreement",
                description="Благодарность"
            ),
        ]
    ),
]


# =============================================================================
# ТЕСТОВЫЕ ФРАЗЫ ДЛЯ INTENT CLASSIFIER
# =============================================================================

INTENT_TEST_CASES = [
    # Приветствия
    ("Привет", "greeting"),
    ("Здравствуйте", "greeting"),
    ("Добрый день", "greeting"),
    ("Хай", "greeting"),
    ("Салам", "greeting"),

    # Вопросы о цене
    ("Сколько стоит?", "price_question"),
    ("Какая цена?", "price_question"),
    ("Прайс", "price_question"),
    ("Почём?", "price_question"),
    ("скока", "price_question"),

    # Вопросы о функциях
    ("Что умеет программа?", "question_features"),
    ("Какие функции?", "question_features"),
    ("Расскажите о системе", "question_features"),
    ("Что вы предлагаете?", "question_features"),

    # Вопросы об интеграциях
    ("Работаете с Kaspi?", "question_integrations"),
    ("Есть интеграция с 1С?", "question_integrations"),
    ("Подключается к банку?", "question_integrations"),

    # Возражения о цене
    ("Дорого", "objection_price"),
    ("Слишком дорого", "objection_price"),
    ("Нет денег", "objection_price"),
    ("Дороговато", "objection_price"),

    # Возражения о конкурентах
    ("А чем лучше iiko?", "objection_competitor"),
    ("У Poster дешевле", "objection_competitor"),
    ("Почему не 1С?", "objection_competitor"),

    # Согласие
    ("Хорошо, беру", "agreement"),
    ("Давайте", "agreement"),
    ("Оставлю заявку", "agreement"),
    ("Окей, согласен", "agreement"),
    ("ок", "agreement"),

    # Отказ
    ("Нет, спасибо", "rejection"),
    ("Не интересно", "rejection"),
    ("Не нужно", "rejection"),
    ("Отстаньте", "rejection"),
]


# =============================================================================
# ТЕСТОВЫЕ ФРАЗЫ ДЛЯ PAIN EXTRACTION
# =============================================================================

PAIN_TEST_CASES = [
    ("У меня всё в Excel, это неудобно", ["учёт в excel", "неудобно"]),
    ("Постоянные потери на складе", ["потери", "склад"]),
    ("Нет времени на отчёты", ["время", "отчёты"]),
    ("Сложно контролировать продавцов", ["контроль", "продавцы"]),
    ("Боюсь штрафов за алкоголь", ["штраф", "алкоголь"]),
    ("Каспи заказы теряются", ["kaspi", "заказы", "теряются"]),
    ("Не знаю сколько товара на складе", ["товар", "склад", "не знаю"]),
    ("Инвентаризация занимает 2 дня", ["инвентаризация", "время"]),
]


# =============================================================================
# КЛАСС ТЕСТИРОВЩИКА
# =============================================================================

class BotStressTester:
    """Полный стресс-тестировщик бота"""

    def __init__(self, verbose: bool = False, auto_fix: bool = False):
        self.verbose = verbose
        self.auto_fix = auto_fix
        self.retriever = KnowledgeRetriever(use_embeddings=False)
        self.results = {
            "intent": {"passed": 0, "failed": 0, "failures": []},
            "knowledge": {"passed": 0, "failed": 0, "failures": []},
            "dialog": {"passed": 0, "failed": 0, "failures": []},
            "pain": {"passed": 0, "failed": 0, "failures": []},
        }
        self.fix_suggestions = []

    def run_all_tests(self):
        """Запуск всех тестов"""
        print("=" * 70)
        print("ПОЛНЫЙ СТРЕСС-ТЕСТ ЧАТБОТА WIPON")
        print("=" * 70)
        print()

        # 1. Тест Intent Classifier
        if HAS_BOT_COMPONENTS:
            self.test_intent_classifier()
        else:
            print("⚠️  Intent Classifier недоступен, пропускаем")

        # 2. Тест Knowledge Retriever
        self.test_knowledge_retriever()

        # 3. Тест диалоговых сценариев
        self.test_dialog_scenarios()

        # 4. Тест Pain Extractor
        if HAS_BOT_COMPONENTS:
            self.test_pain_extractor()
        else:
            print("⚠️  Pain Extractor недоступен, пропускаем")

        # Итоги
        self.print_summary()

        # Предложения по исправлению
        if self.fix_suggestions:
            self.print_fix_suggestions()

    def test_intent_classifier(self):
        """Тест классификатора интентов"""
        print("\n" + "=" * 50)
        print("ТЕСТ INTENT CLASSIFIER")
        print("=" * 50)

        for message, expected_intent in INTENT_TEST_CASES:
            try:
                result = classify_intent(message)
                actual_intent = result.get("intent", "unknown")

                if actual_intent == expected_intent:
                    self.results["intent"]["passed"] += 1
                    if self.verbose:
                        print(f"✓ \"{message}\" → {actual_intent}")
                else:
                    self.results["intent"]["failed"] += 1
                    self.results["intent"]["failures"].append({
                        "message": message,
                        "expected": expected_intent,
                        "actual": actual_intent
                    })
                    print(f"✗ \"{message}\" → {actual_intent} (ожидалось: {expected_intent})")
                    self.fix_suggestions.append({
                        "component": "intent_classifier",
                        "issue": f"'{message}' классифицировано как '{actual_intent}' вместо '{expected_intent}'",
                        "suggestion": f"Добавить паттерн для '{expected_intent}' в patterns.py"
                    })
            except Exception as e:
                self.results["intent"]["failed"] += 1
                print(f"✗ \"{message}\" → ОШИБКА: {e}")

        total = self.results["intent"]["passed"] + self.results["intent"]["failed"]
        print(f"\nIntent Classifier: {self.results['intent']['passed']}/{total}")

    def test_knowledge_retriever(self):
        """Тест retriever'а знаний"""
        print("\n" + "=" * 50)
        print("ТЕСТ KNOWLEDGE RETRIEVER")
        print("=" * 50)

        # Используем кейсы из stress_test_knowledge.py
        from scripts.stress_test_knowledge import TEST_CASES

        for tc in TEST_CASES:
            facts = self.retriever.retrieve(tc.query, top_k=3)

            found_topics = []
            for section in WIPON_KNOWLEDGE.sections:
                if section.facts.strip() in facts:
                    found_topics.append(section.topic)

            found = any(t in found_topics for t in tc.expected_topics)

            if found:
                self.results["knowledge"]["passed"] += 1
                if self.verbose:
                    print(f"✓ [{tc.category}] \"{tc.query[:40]}...\"")
            else:
                self.results["knowledge"]["failed"] += 1
                self.results["knowledge"]["failures"].append({
                    "query": tc.query,
                    "expected": tc.expected_topics,
                    "found": found_topics
                })
                print(f"✗ [{tc.category}] \"{tc.query[:40]}...\"")
                print(f"  Expected: {tc.expected_topics}, Found: {found_topics}")

                # Предложение исправления
                self.fix_suggestions.append({
                    "component": "knowledge_data",
                    "issue": f"Запрос '{tc.query}' не находит {tc.expected_topics}",
                    "suggestion": f"Добавить keywords из запроса в секцию {tc.expected_topics[0]}"
                })

        total = self.results["knowledge"]["passed"] + self.results["knowledge"]["failed"]
        print(f"\nKnowledge Retriever: {self.results['knowledge']['passed']}/{total}")

    def test_dialog_scenarios(self):
        """Тест диалоговых сценариев"""
        print("\n" + "=" * 50)
        print("ТЕСТ ДИАЛОГОВЫХ СЦЕНАРИЕВ")
        print("=" * 50)

        for scenario in DIALOG_SCENARIOS:
            print(f"\n--- {scenario.name} ({scenario.persona.value}) ---")
            print(f"    {scenario.description}")

            scenario_passed = True

            for turn in scenario.turns:
                # Тест knowledge retrieval
                facts = self.retriever.retrieve(turn.user_message, top_k=2)

                # Проверка ожидаемых топиков
                if turn.expected_topics:
                    found_topics = []
                    for section in WIPON_KNOWLEDGE.sections:
                        if section.facts.strip() in facts:
                            found_topics.append(section.topic)

                    topic_found = any(t in found_topics for t in turn.expected_topics)

                    if not topic_found:
                        scenario_passed = False
                        print(f"  ✗ \"{turn.user_message[:40]}...\"")
                        print(f"    Topics: expected {turn.expected_topics}, found {found_topics}")
                        self.fix_suggestions.append({
                            "component": "knowledge_data",
                            "issue": f"Сценарий '{scenario.name}': '{turn.user_message}' не находит {turn.expected_topics}",
                            "suggestion": f"Проверить keywords в секции {turn.expected_topics[0]}"
                        })
                    elif self.verbose:
                        print(f"  ✓ \"{turn.user_message[:40]}...\"")

                # Проверка should_contain
                if turn.should_contain:
                    for phrase in turn.should_contain:
                        if phrase.lower() not in facts.lower():
                            scenario_passed = False
                            print(f"  ✗ \"{turn.user_message[:40]}...\"")
                            print(f"    Должно содержать '{phrase}', но не найдено")
                            self.fix_suggestions.append({
                                "component": "knowledge_data",
                                "issue": f"Ответ на '{turn.user_message}' не содержит '{phrase}'",
                                "suggestion": f"Проверить facts в релевантной секции"
                            })

            if scenario_passed:
                self.results["dialog"]["passed"] += 1
                print(f"  → Сценарий PASSED")
            else:
                self.results["dialog"]["failed"] += 1
                print(f"  → Сценарий FAILED")

        total = self.results["dialog"]["passed"] + self.results["dialog"]["failed"]
        print(f"\nДиалоговые сценарии: {self.results['dialog']['passed']}/{total}")

    def test_pain_extractor(self):
        """Тест экстрактора боли"""
        print("\n" + "=" * 50)
        print("ТЕСТ PAIN EXTRACTOR")
        print("=" * 50)

        for message, expected_pains in PAIN_TEST_CASES:
            try:
                pains = extract_pains(message)
                pains_lower = [p.lower() for p in pains]

                found = any(
                    any(ep in p for p in pains_lower)
                    for ep in expected_pains
                )

                if found or pains:  # Если нашёл хоть что-то
                    self.results["pain"]["passed"] += 1
                    if self.verbose:
                        print(f"✓ \"{message[:40]}...\" → {pains}")
                else:
                    self.results["pain"]["failed"] += 1
                    print(f"✗ \"{message[:40]}...\"")
                    print(f"  Expected keywords: {expected_pains}, Found: {pains}")
            except Exception as e:
                self.results["pain"]["failed"] += 1
                print(f"✗ \"{message[:40]}...\" → ОШИБКА: {e}")

        total = self.results["pain"]["passed"] + self.results["pain"]["failed"]
        print(f"\nPain Extractor: {self.results['pain']['passed']}/{total}")

    def print_summary(self):
        """Итоговая статистика"""
        print("\n" + "=" * 70)
        print("ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 70)

        total_passed = sum(r["passed"] for r in self.results.values())
        total_failed = sum(r["failed"] for r in self.results.values())
        total = total_passed + total_failed

        print(f"""
┌────────────────────────┬─────────┬─────────┬─────────┐
│ Компонент              │ Passed  │ Failed  │    %    │
├────────────────────────┼─────────┼─────────┼─────────┤""")

        for name, stats in self.results.items():
            p = stats["passed"]
            f = stats["failed"]
            t = p + f
            pct = (100 * p / t) if t > 0 else 0
            status = "✓" if f == 0 else "✗"
            print(f"│ {status} {name:<19} │ {p:>7} │ {f:>7} │ {pct:>6.1f}% │")

        print(f"""├────────────────────────┼─────────┼─────────┼─────────┤
│ ВСЕГО                  │ {total_passed:>7} │ {total_failed:>7} │ {100*total_passed/total if total else 0:>6.1f}% │
└────────────────────────┴─────────┴─────────┴─────────┘
""")

        if total_failed == 0:
            print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ!")
        else:
            print(f"⚠️  НАЙДЕНО {total_failed} ПРОБЛЕМ")

    def print_fix_suggestions(self):
        """Вывод предложений по исправлению"""
        print("\n" + "=" * 70)
        print("ПРЕДЛОЖЕНИЯ ПО ИСПРАВЛЕНИЮ")
        print("=" * 70)

        # Группируем по компонентам
        by_component = {}
        for fix in self.fix_suggestions:
            comp = fix["component"]
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(fix)

        for comp, fixes in by_component.items():
            print(f"\n[{comp}]")
            for i, fix in enumerate(fixes[:10], 1):  # Максимум 10 на компонент
                print(f"  {i}. {fix['issue']}")
                print(f"     → {fix['suggestion']}")

            if len(fixes) > 10:
                print(f"  ... и ещё {len(fixes) - 10} проблем")


def main():
    parser = argparse.ArgumentParser(description="Полный стресс-тест бота")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--fix", "-f", action="store_true", help="Автоисправление (TODO)")
    args = parser.parse_args()

    tester = BotStressTester(verbose=args.verbose, auto_fix=args.fix)
    tester.run_all_tests()

    return 0 if sum(r["failed"] for r in tester.results.values()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
