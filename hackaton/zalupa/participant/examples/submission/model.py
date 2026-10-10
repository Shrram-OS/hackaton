"""Приклад здачі: та сама базова модель, але з вагами у файлі.

Це повний комплект, який приймає залік. У теці три файли:

    toksym.json    маніфест: що запускати і без чого воно не працює
    model.py       цей файл
    weights.json   коефіцієнти

Перевірити перед здачею:

    toksym check examples/submission

Прогнати:

    toksym score --submission examples/submission --shots 200 --seed 1 --lead 0.03

Різниця з `model_baseline.py` одна: там коефіцієнти зашиті в код, тут вони
читаються з файлу. Саме так приходить навчена модель — код описує форму
рішення, файл ваг несе числа, які дало навчання.

Що саме рахує модель і звідки взялися ці числа — у докстрінгу
`examples/model_baseline.py`.
"""
import json
import math
import sys
from pathlib import Path

# ── Геометрія машини. Відома й опублікована, вгадувати нічого. ──
A_MINOR = 0.95                       # малий радіус, м
KAPPA = 1.70                         # витягнутість
CHORD_M = 2.0 * A_MINOR * KAPPA      # довжина центральної хорди, м
GREENWALD_AREA = math.pi * A_MINOR ** 2

# ── Ваги ──
#
# Поруч із цим файлом, а не в коді. Так виглядає навчена модель: код описує
# форму рішення, файл ваг — конкретні числа, які дало навчання. Тут вони
# зміряні вручну, але місце в них те саме.
#
# Шлях будується від самого файлу, а не від поточної теки. Обидва варіанти
# працюють (програма запускає модель саме з її теки), але `__file__`
# працюватиме й тоді, коли ви прогоните модель звідкись іще.
with open(Path(__file__).resolve().parent / "weights.json", encoding="utf-8") as f:
    W = json.load(f)

LM_WARN = W["lm_warn"]
LM_SOFT = W["lm_soft"]
F_GW_WARN = W["f_gw_warn"]
F_GW_SOFT = W["f_gw_soft"]
A_FAST = W["alpha_lm"]
A_SLOW = W["alpha_f_gw"]
IP_MIN_MA = W["ip_min_MA"]


def sigmoid(x, x0, soft):
    """Плавний перехід 0 → 1 навколо x0. Різкий поріг давав би смикання."""
    return 1.0 / (1.0 + math.exp(-(x - x0) / soft))


class Model:
    def start_shot(self, meta):
        # Стан скидається на кожен розряд: нести згладжене значення з
        # попереднього — це підглядати в чуже минуле.
        self.lm = 0.0
        self.f_gw = 0.0
        self.seen = False
        self.peak = 0.0

    def predict(self, t, channels):
        ip = channels.get("mag_ip_MA")
        chord = channels.get("ifr_chord_core")

        # Прилад має право відмовити — у каналі буде nan. Тоді тримаємо
        # попередню оцінку, а не вважаємо, що величина впала до нуля.
        if not _ok(ip) or not _ok(chord) or ip < IP_MIN_MA:
            return self.peak

        f_gw = (chord / CHORD_M) * GREENWALD_AREA / max(ip, 1e-6)

        # Модуль: сідловій петлі байдуже, у який бік повернулася мода.
        lm_raw = channels.get("mag_saddle_lm")
        lm = abs(lm_raw) if _ok(lm_raw) else self.lm

        if not self.seen:
            self.f_gw, self.lm = f_gw, lm
            self.seen = True
        else:
            self.f_gw += A_SLOW * (f_gw - self.f_gw)
            self.lm += A_FAST * (lm - self.lm)

        # Два незалежні шляхи до тривоги. Складаємо як «жоден не спрацював»:
        # сума переоцінювала б збіг двох слабких сигналів, максимум — навпаки,
        # ігнорував би його зовсім.
        a = sigmoid(self.lm, LM_WARN, LM_SOFT)
        b = sigmoid(self.f_gw, F_GW_WARN, F_GW_SOFT)
        p = 1.0 - (1.0 - a) * (1.0 - b)

        # Тривога — подія, а не стан: якщо розряд уже підходив до межі, це
        # сталося. Правило заліку бере найперший перетин порога, тож тримати
        # максимум чесніше за миттєве значення.
        self.peak = max(self.peak, p)
        return self.peak


def _ok(v):
    """Чи це справжнє число. nan означає відмову приладу на цьому кроці."""
    return v is not None and v == v


if __name__ == "__main__":
    # Запуск на заліку: toksym score --shots 200 --seed 1 -- python3 model_baseline.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from toksym_model import serve

    serve(Model())
