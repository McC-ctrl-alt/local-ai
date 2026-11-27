"""Local demo for Curator.

GET / serves ui/index.html. POST /ask reads {"question", "role"} and returns
the answer, citations, and trail from policy_rag.consult(). The page is the
only place that changes how an answer looks. The server listens on
127.0.0.1 and does not call anything else to answer a question.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import engine
import policy_rag


HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_BODY = 64 * 1024
ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "ui" / "index.html"


def warm_models() -> None:
    """Load the reranker and both Ollama models before the first question.

    policy_rag keeps the reranker, so consult() does not load it again.
    One embed call and one short chat call bring nomic-embed-text and the
    answer model into Ollama. Reading the corpus builds the embedding cache
    when it is cold. A dummy rerank pays the first ONNX run at startup.
    """
    reranker = policy_rag.get_reranker()
    reranker.predict([("warmup", "policy")], batch_size=8)
    pdfs = engine.list_pdfs(policy_rag.CORPUS_DIR)
    if pdfs:
        engine.load_corpus(pdfs)
    engine.embed("warmup")
    engine.chat_answer("Reply with ok.", "ok", max_tokens=1)


def make_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind the demo on localhost. The host is not configurable."""
    server = ThreadingHTTPServer((HOST, port), CuratorHandler)
    server.daemon_threads = True
    return server


class CuratorHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self._route() != "/":
            self.send_error(404)
            return
        try:
            body = INDEX_PATH.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", _page_policy())
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self._route() != "/ask":
            self.send_error(404)
            return
        payload = self._read_json()
        if payload is None:
            return
        question = payload.get("question")
        role = payload.get("role")
        if (
            not isinstance(question, str)
            or not isinstance(role, str)
            or not question.strip()
            or not role.strip()
        ):
            self._send_json(400, {"error": "question and role are required."})
            return
        try:
            result = policy_rag.consult(question, role)
        except SystemExit as error:
            self._send_json(500, {"error": _exit_message(error)})
            return
        except Exception as error:
            message = str(error).strip() or "Curator could not answer."
            self._send_json(500, {"error": message})
            return
        self._send_json(
            200,
            {
                "answer": result["answer"],
                "citations": result["citations"],
                "trail": result["trail"],
                "stage": result["stage"],
            },
        )

    def _route(self) -> str:
        return self.path.split("?", 1)[0]

    def _read_json(self) -> dict[str, Any] | None:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length) if raw_length is not None else None
        except ValueError:
            length = None
        if length is None or length < 0 or length > MAX_BODY:
            self._send_json(400, {"error": "Expected a JSON body."})
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Expected a JSON body."})
            return None
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Expected a JSON object."})
            return None
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def _page_policy() -> str:
    return (
        "default-src 'none'; "
        "style-src 'unsafe-inline'; "
        "script-src 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'none'; "
        "base-uri 'none'; "
        "form-action 'self'"
    )


def _exit_message(error: SystemExit) -> str:
    if isinstance(error.code, str) and error.code.strip():
        return error.code.strip()
    return "Curator could not answer."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the local Curator demo.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    print("Warming models…", flush=True)
    try:
        warm_models()
    except Exception as error:
        print(f"Model warmup skipped: {error}", file=sys.stderr)
    except SystemExit as error:
        print(f"Model warmup skipped: {_exit_message(error)}", file=sys.stderr)
    server = make_server(args.port)
    url = f"http://{HOST}:{server.server_address[1]}"
    print(f"Curator demo at {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
