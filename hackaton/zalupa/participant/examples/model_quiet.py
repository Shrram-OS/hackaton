"""Референс «мовчазна модель»: ніколи не б'є тривогу.

Потрібна не як рішення, а як підлога. Вона дає TPR = 0 при FPR = 0, і будь-яка
метрика, яка не відрізняє від неї справжню модель, виміряна неправильно.
"""
import sys
from pathlib import Path


class Model:
    def start_shot(self, meta):
        pass

    def predict(self, t, channels):
        return 0.0


if __name__ == "__main__":
    # Запуск на заліку: toksym score --shots 200 --seed 7 -- python3 model_quiet.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from toksym_model import serve

    serve(Model())
