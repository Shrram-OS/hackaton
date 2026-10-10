"""Референс «частка гринвальдівської межі».

Найпростіше, що взагалі можна зробити з підручника — але вже не в один рядок,
бо готової частки гринвальдівської межі в даних немає. Її треба скласти:

  1. Середня густина. Інтерферометр міряє не її, а лінійний інтеграл уздовж
     хорди. Ділимо на довжину хорди — 2·κ·a.
  2. Струм плазми з поясу Роговського.
  3. n_GW = I / (π a²), далі відношення.

Жодного навчання, жодної підгонки, жодної історії — тільки геометрія машини,
яка й так відома. Це орієнтир, а не рішення: канали зашумлені, калібрування
зсунуте на кожен розряд, а густина — не єдина межа, до якої розряд може
підійти.
"""
import sys
from pathlib import Path
import math

A_MINOR = 0.95
KAPPA = 1.70
CHORD_M = 2.0 * A_MINOR * KAPPA
GREENWALD_AREA = math.pi * A_MINOR ** 2


class Model:
    def start_shot(self, meta):
        self.ema = 0.0
        self.seen = False

    def predict(self, t, channels):
        ip = channels.get("mag_ip_MA", float("nan"))
        chord = channels.get("ifr_chord_core", float("nan"))
        # Канал має право відмовити: тоді тримаємо попереднє значення.
        if ip != ip or chord != chord or ip < 0.3:
            return self.ema

        ne_bar = chord / CHORD_M
        f_gw = ne_bar * GREENWALD_AREA / max(ip, 1e-6)
        p = 1.0 / (1.0 + math.exp(-(f_gw - 0.95) / 0.07))

        # Згладжування: миттєвий кидок інтерферометра не має бути тривогою.
        self.ema = p if not self.seen else self.ema + 0.02 * (p - self.ema)
        self.seen = True
        return self.ema


if __name__ == "__main__":
    # Запуск на заліку: toksym score --shots 200 --seed 7 -- python3 model_fgw.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from toksym_model import serve

    serve(Model())
