#!/usr/bin/env python3
"""Serve the MFE 230ZB group deck.

    pip install flask
    python presentation/app.py               ->  http://127.0.0.1:5055   (opens the browser)
    python presentation/app.py --no-browser

The deck is a set of static files in deck/, so it also runs without Flask:
open deck/index.html in Chrome, or `python -m http.server 5055 --directory deck`.

    http://127.0.0.1:5055/?still=1    every slide at its last step, one under the other
                                     (the pre-read; print it to PDF from Chrome)
"""
import os
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "deck")
PORT = int(os.environ.get("PORT", "5055"))

try:
    from flask import Flask, Response, send_from_directory
except ImportError:
    print("Flask is not installed. Either `pip install flask`, or open deck/index.html directly, or run:\n"
          f"    python -m http.server {PORT} --directory deck")
    sys.exit(1)

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(DECK, "index.html")


@app.route("/<path:path>")
def files(path):
    if path == "js/portfolio.js" and not os.path.exists(os.path.join(DECK, path)):
        # the stock-level ranks are never committed: without them slide 4.3 shows its fallback
        return Response("/* no stock-level ranks on this machine */\n", mimetype="application/javascript")
    return send_from_directory(DECK, path)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Deck: {url}   one click per slide (its animations play by themselves); Left = back, R = replay,")
    print("      G + number to jump, I index, F full screen, T timer, B black")
    print(f"Pre-read: {url}?still=1  (print to PDF from Chrome)")
    app.run(host="127.0.0.1", port=PORT, debug=False)
