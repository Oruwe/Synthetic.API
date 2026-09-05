"""Mock legacy shipping portal.

Deliberately old-school, server-rendered HTML with no real API — this is the
"legacy portal that doesn't have an API" the Web-Navigator agent has to
operate like a human would: log in, click through, read the dashboard.

Selectors are kept stable and explicit (data-testid / data-field attributes)
on purpose: the Web-Navigator agent's extractor.py is designed to bind to
these exact attributes rather than guessing from visual layout, which is the
whole point of a "legacy portal that got a synthetic API bolted on."
"""

import json
import os
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("PORTAL_SECRET_KEY", "dev-only-not-a-real-secret")

PORTAL_USERNAME = os.environ.get("PORTAL_USERNAME", "admin")
PORTAL_PASSWORD = os.environ.get("PORTAL_PASSWORD", "admin123")

DATA_PATH = Path(__file__).parent / "data" / "orders.json"


def load_orders() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.get("/")
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


@app.get("/login")
def login():
    return render_template("login.html", error=None)


@app.post("/login")
def do_login():
    username = request.form.get("username", "")
    password = request.form.get("password", "")
    if username == PORTAL_USERNAME and password == PORTAL_PASSWORD:
        session["logged_in"] = True
        return redirect(url_for("dashboard"))
    return render_template("login.html", error="Invalid username or password"), 401


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    orders = load_orders()
    return render_template("dashboard.html", orders=orders)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
