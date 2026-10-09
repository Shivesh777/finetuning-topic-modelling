#!/usr/bin/env python3
"""Serve section 3 (NoLBERT fine-tuning) as a standalone deck.

    pip install flask
    python section3/app.py               ->  http://127.0.0.1:5056   (opens the browser)
    python section3/app.py --no-browser

Four slides: 3.1 Find the partner, 3.2 Supervision trick, 3.3 What it read,
3.4 Reshape the space. /reshape.html is the interactive companion page for 3.4.
Static files, so it also runs without Flask:
`python -m http.server 5056 --directory section3`.
"""
import os
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "5056"))

try:
    from flask import Flask, send_from_directory
except ImportError:
    print("Flask is not installed. Either `pip install flask`, or open index.html directly, or run:\n"
          f"    python -m http.server {PORT} --directory section3")
    sys.exit(1)

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/<path:path>")
def files(path):
    return send_from_directory(HERE, path)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Section 3: {url}   one click per slide; Left = back, R = replay")
    print(f"Companion page for 3.4: {url}reshape.html")
    app.run(host="127.0.0.1", port=PORT, debug=False)
