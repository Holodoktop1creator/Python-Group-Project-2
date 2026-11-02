"""Собирает самостоятельный index.html (с <head> и viewport) из game.html.

game.html — содержимое страницы без <html>/<head>: его публикуют как Artifact.
index.html — готовый файл, который можно открыть на телефоне или завернуть в APK.
"""
from pathlib import Path

here = Path(__file__).resolve().parent
body = (here / "game.html").read_text(encoding="utf-8")
head = (
    "<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, maximum-scale=1, "
    "user-scalable=no, viewport-fit=cover\">\n"
)
# <title>, <meta>, <link>, <style> из начала game.html уходят в <head>
split = body.index("<canvas")
(here / "index.html").write_text(
    head + body[:split] + "</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n",
    encoding="utf-8",
)
print("index.html собран")
