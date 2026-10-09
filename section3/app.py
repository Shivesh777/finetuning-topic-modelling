#!/usr/bin/env python3
"""Serve the deck with Kevin's section-3 revisions.

    pip install flask
    python section3/app.py               ->  http://127.0.0.1:5056   (opens the browser)
    python section3/app.py --no-browser

The full team deck with section 3 revised: a 3.0 intro animation (the pair,
the space reshaped), 3.1 without the padding label, 3.2 with the Dec 2019
fine-tuning window. Static files, so it also runs without Flask:
`python -m http.server 5056 --directory section3`.
"""
import os
import sys
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", "5056"))

try:
    from flask import Flask, Response, send_from_directory
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
    if path in ("js/portfolio.js", "js/similarity.js") and not os.path.exists(os.path.join(HERE, path)):
        # generated files, never committed: the slides that use them show their fallbacks
        return Response("/* not on this machine */\n", mimetype="application/javascript")
    return send_from_directory(HERE, path)


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Deck (section-3 revisions): {url}   one click per slide; Left = back, R = replay")
    print(f"Companion page for the reshape idea: {url}reshape.html")
    app.run(host="127.0.0.1", port=PORT, debug=False)
