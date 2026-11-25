"""Local demo UI. Binds to 127.0.0.1 and calls demo.ask().

    python3 demo_ui.py
    make demo

The first Qwen call on an M1 can take half a minute, so startup sends one
tiny chat request to load the model, and the page uses a 120s fetch
instead of streaming.
"""

import os
import sys

from flask import Flask, jsonify, render_template, request

import demo
import engine
import policy_rag

HOST = "127.0.0.1"
PORT = int(os.environ.get("DEMO_PORT", "5000"))
ASK_TIMEOUT_SECONDS = 120


def warmup() -> None:
    """One tiny Qwen call so the first demo question is not a cold load."""
    model = engine.CHAT_MODEL
    print(f"Warming {model}…", flush=True)
    try:
        engine.chat_answer("Reply with the single word ok.", "ok", max_tokens=4)
    except Exception as exc:
        raise SystemExit(
            f"Pre-warm failed. Start Ollama and pull {model}.\n{exc}"
        ) from exc
    print("Ready.", flush=True)


def create_app(*, prewarm: bool = False) -> Flask:
    """Build the app. Tests pass prewarm=False so Ollama is never called."""
    if prewarm:
        warmup()
    app = Flask(__name__)

    @app.get("/")
    def index():
        roles = policy_rag.known_roles()
        return render_template(
            "index.html",
            roles=roles,
            timeout_seconds=ASK_TIMEOUT_SECONDS,
        )

    @app.post("/ask")
    def ask_view():
        payload = request.get_json(silent=True) or {}
        question = payload.get("question", "")
        role = payload.get("role", "")
        try:
            result = demo.ask(question, role)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    return app


def main() -> int:
    app = create_app(prewarm=True)
    print(f"Open http://{HOST}:{PORT}  (this machine only)", flush=True)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
