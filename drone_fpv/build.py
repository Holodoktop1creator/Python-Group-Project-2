"""Собирает самостоятельный index.html для «Шахин FPV» из game.html.

game.html — содержимое страницы без <html>/<head>: его публикуют как Artifact.
index.html — готовый файл: открой в браузере на компьютере или выложи на хостинг.
"""
from pathlib import Path

here = Path(__file__).resolve().parent
body = (here / "game.html").read_text(encoding="utf-8")
head = (
    "<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, maximum-scale=1, "
    "user-scalable=no, viewport-fit=cover\">\n"
)
split = body.index("<canvas")
(here / "index.html").write_text(
    head + body[:split] + "</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n",
    encoding="utf-8",
)
print("index.html собран")
