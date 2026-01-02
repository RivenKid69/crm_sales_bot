"""
State Machine — управление состояниями диалога

Поддерживает:
- Базовый flow: greeting → qualification → presentation → close
- SPIN Selling flow: greeting → spin_situation → spin_problem → spin_implication → spin_need_payoff → presentation → close
"""

from typing import Tuple, Dict, Optional
from config import SALES_STATES, QUESTION_INTENTS


# SPIN-фазы и их порядок
SPIN_PHASES = ["situation", "problem", "implication", "need_payoff"]

# Состояния SPIN
SPIN_STATES = {
    "situation": "spin_situation",
    "problem": "spin_problem",
    "implication": "spin_implication",
    "need_payoff": "spin_need_payoff",
}

# SPIN-интенты которые указывают на прогресс в соответствующей фазе
SPIN_PROGRESS_INTENTS = {
    "situation_provided": "situation",
    "problem_revealed": "problem",
    "implication_acknowledged": "implication",
    "need_expressed": "need_payoff",
}


class StateMachine:
    def __init__(self):
        self.state = "greeting"
        self.collected_data = {}
        self.spin_phase = None  # Текущая SPIN-фаза (если в SPIN flow)

    def reset(self):
        self.state = "greeting"
        self.collected_data = {}
        self.spin_phase = None

    def update_data(self, data: Dict):
        """Сохраняем извлечённые данные"""
        for key, value in data.items():
            if value:
                self.collected_data[key] = value

    def _get_current_spin_phase(self) -> Optional[str]:
        """Определяем текущую SPIN-фазу по состоянию"""
        config = SALES_STATES.get(self.state, {})
        return config.get("spin_phase")

    def _get_next_spin_state(self, current_phase: str) -> Optional[str]:
        """Определяем следующее SPIN-состояние"""
        if current_phase not in SPIN_PHASES:
            return None

        current_idx = SPIN_PHASES.index(current_phase)
        if current_idx < len(SPIN_PHASES) - 1:
            next_phase = SPIN_PHASES[current_idx + 1]
            return SPIN_STATES.get(next_phase)
        return "presentation"  # После need_payoff идёт presentation

    def _check_spin_data_complete(self, config: Dict) -> bool:
        """Проверяем собраны ли данные для текущей SPIN-фазы"""
        required = config.get("required_data", [])
        if not required:
            return True  # Если нет обязательных данных — считаем завершённой

        for field in required:
            if not self.collected_data.get(field):
                return False
        return True

    def _should_skip_spin_phase(self, phase: str) -> bool:
        """Определяем можно ли пропустить SPIN-фазу (для ускорения)"""
        # Implication и Need-Payoff можно пропустить если клиент уже готов
        if phase in ["implication", "need_payoff"]:
            # Если клиент уже выразил сильный интерес
            if self.collected_data.get("high_interest"):
                return True
            # Если уже есть желаемый результат
            if phase == "need_payoff" and self.collected_data.get("desired_outcome"):
                return True
        return False

    def apply_rules(self, intent: str) -> Tuple[str, str]:
        """
        Определяем действие и следующее состояние

        Returns: (action, next_state)
        """
        config = SALES_STATES.get(self.state, {})

        # Финальное состояние
        if config.get("is_final"):
            return "final", self.state

        # =====================================================================
        # ПРИОРИТЕТ 0: Вопросы требуют ответа!
        # Если клиент задаёт вопрос — сначала отвечаем, потом продолжаем
        # =====================================================================
        if intent in QUESTION_INTENTS:
            transitions = config.get("transitions", {})
            if intent in transitions:
                next_state = transitions[intent]
            else:
                next_state = self.state

            return "answer_question", next_state

        # =====================================================================
        # ПРИОРИТЕТ 1: Rejection — всегда обрабатываем немедленно
        # =====================================================================
        if intent == "rejection":
            transitions = config.get("transitions", {})
            if "rejection" in transitions:
                next_state = transitions["rejection"]
                return f"transition_to_{next_state}", next_state

        # =====================================================================
        # ПРИОРИТЕТ 2: SPIN-специфичная логика
        # =====================================================================
        spin_phase = self._get_current_spin_phase()

        if spin_phase:
            # Проверяем SPIN-специфичные интенты для перехода
            if intent in SPIN_PROGRESS_INTENTS:
                intent_phase = SPIN_PROGRESS_INTENTS[intent]
                # Если интент соответствует текущей или следующей фазе — это прогресс
                if intent_phase == spin_phase or \
                   (SPIN_PHASES.index(intent_phase) > SPIN_PHASES.index(spin_phase) if intent_phase in SPIN_PHASES and spin_phase in SPIN_PHASES else False):
                    # Проверяем можно ли перейти дальше
                    transitions = config.get("transitions", {})
                    if intent in transitions:
                        next_state = transitions[intent]
                        return f"transition_to_{next_state}", next_state

            # Автоматический переход если данные собраны
            if self._check_spin_data_complete(config):
                transitions = config.get("transitions", {})
                if "data_complete" in transitions:
                    next_state = transitions["data_complete"]
                    # Проверяем можно ли пропустить следующую фазу
                    next_config = SALES_STATES.get(next_state, {})
                    next_phase = next_config.get("spin_phase")
                    if next_phase and self._should_skip_spin_phase(next_phase):
                        # Пропускаем и идём дальше
                        skip_transitions = next_config.get("transitions", {})
                        if "data_complete" in skip_transitions:
                            next_state = skip_transitions["data_complete"]
                    return f"transition_to_{next_state}", next_state

        # =====================================================================
        # ПРИОРИТЕТ 2: Специальные правила текущего состояния
        # =====================================================================
        rules = config.get("rules", {})
        if intent in rules:
            return rules[intent], self.state

        # =====================================================================
        # ПРИОРИТЕТ 3: Переходы по интенту
        # =====================================================================
        transitions = config.get("transitions", {})
        if intent in transitions:
            next_state = transitions[intent]
            return f"transition_to_{next_state}", next_state

        # =====================================================================
        # ПРИОРИТЕТ 4: Проверка data_complete для non-SPIN состояний
        # =====================================================================
        required = config.get("required_data", [])
        if required:
            missing = [f for f in required if not self.collected_data.get(f)]
            if not missing and "data_complete" in transitions:
                next_state = transitions["data_complete"]
                return f"transition_to_{next_state}", next_state

        # =====================================================================
        # ПРИОРИТЕТ 5: Автопереход (для greeting)
        # =====================================================================
        if "any" in transitions:
            next_state = transitions["any"]
            return f"transition_to_{next_state}", next_state

        # =====================================================================
        # Дефолт: остаёмся в текущем состоянии
        # =====================================================================
        # Для SPIN состояний используем соответствующий промпт
        if spin_phase:
            return self.state, self.state

        return "continue_current_goal", self.state

    def process(self, intent: str, extracted_data: Dict = None) -> Dict:
        """Обработать интент, вернуть результат"""
        prev_state = self.state

        if extracted_data:
            self.update_data(extracted_data)

        action, next_state = self.apply_rules(intent)
        self.state = next_state

        # Обновляем spin_phase
        self.spin_phase = self._get_current_spin_phase()

        config = SALES_STATES.get(self.state, {})
        required = config.get("required_data", [])
        missing = [f for f in required if not self.collected_data.get(f)]

        # Собираем optional данные для SPIN
        optional = config.get("optional_data", [])
        optional_missing = [f for f in optional if not self.collected_data.get(f)]

        return {
            "action": action,
            "prev_state": prev_state,
            "next_state": next_state,
            "goal": config.get("goal", ""),
            "collected_data": self.collected_data.copy(),
            "missing_data": missing,
            "optional_data": optional_missing,
            "is_final": config.get("is_final", False),
            "spin_phase": self.spin_phase,
        }


if __name__ == "__main__":
    sm = StateMachine()
    
    # Тест
    print("=== Тест State Machine ===\n")
    
    tests = [
        ("greeting", {}),
        ("price_question", {}),
        ("info_provided", {"company_size": 15}),
        ("info_provided", {"pain_point": "теряем клиентов"}),
        ("agreement", {}),
    ]
    
    for intent, data in tests:
        result = sm.process(intent, data)
        print(f"Intent: {intent}")
        print(f"  {result['prev_state']} → {result['next_state']}")
        print(f"  Action: {result['action']}")
        print(f"  Data: {result['collected_data']}\n")