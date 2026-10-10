"""Обгортка, яка перетворює звичайний клас на модель для заліку.

Контракт моделі — два методи:

    class Model:
        def start_shot(self, meta): ...        # новий розряд
        def predict(self, t, channels): ...    # ймовірність зриву, 0…1

`channels` — словник «ім'я каналу → число». Канал має право відмовити: тоді
значення буде `nan`, і це не помилка, а те, що станеться на справжній
установці. Історію збирає сама модель: наступного кроку повернутися до
попереднього вже не можна.

Щоб це запустилося на заліку, лишається один рядок:

    if __name__ == "__main__":
        from toksym_model import serve
        serve(Model())

І запуск:

    toksym score --shots 200 --seed 7 -- python3 model.py

Свій протокол (для тих, хто пише не на Python) описаний у README: один рядок
JSON униз, одне число вгору, суворо по черзі.
"""
import json
import sys

PROTOCOL = 1


def log(*parts):
    """Надрукувати щось для себе, не зламавши протокол.

    Звичайний `print` пішов би в канал відповідей, і залік зупинився б на
    «модель відповіла не числом». Рядок з `#` на початку вважається логом і
    переспрямовується в stderr бінарника.
    """
    print("#", *parts, flush=True)


def serve(model, stdin=None, stdout=None):
    """Крутити цикл заліку, поки бінарник не скаже «до побачення»."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    def say(line):
        # flush після кожного рядка обов'язковий: за блочної буферизації
        # відповідь застрягла б у буфері, а залік побачив би мовчання і
        # зупинився б з «модель не стартувала».
        stdout.write(f"{line}\n")
        stdout.flush()

    names = []
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        ev = json.loads(raw)
        kind = ev.get("type")

        if kind == "hello":
            got = ev.get("protocol")
            if got != PROTOCOL:
                raise SystemExit(
                    f"версія протоколу {got} замість {PROTOCOL}: "
                    "обгортка й бінарник з різних наборів"
                )
            say(json.dumps({"type": "ready"}))

        elif kind == "shot_start":
            # Склад апаратури оголошує бінарник. Обгортка його не дублює:
            # інакше два списки неминуче розійшлися б.
            names = list(ev["channels"])
            if hasattr(model, "start_shot"):
                meta = {k: ev[k] for k in ("shot_id", "dt", "duration", "scenario")}
                model.start_shot(meta)

        elif kind == "step":
            # null у потоці — це відмова приладу; для моделі це nan.
            channels = {
                n: (float("nan") if v is None else float(v))
                for n, v in zip(names, ev["v"])
            }
            say(repr(float(model.predict(ev["t"], channels))))

        elif kind == "shot_end":
            # Міток тут немає навмисно: підказувати по ходу прогону нічого.
            if hasattr(model, "end_shot"):
                model.end_shot(ev["shot_id"])

        elif kind == "bye":
            return
