"""A fake OpenRouter that stands in for urllib.request.urlopen.

It answers like a model that follows a policy (e.g. always pick the 'pass' option), records
every request it receives, and can be scripted to fail (HTTP errors, network errors).
"""
import io
import json
import threading
import urllib.error
from unittest import mock

from bench import run
from bench.openrouter import API_URL, MODELS_URL

MODELS = {"data": [
    {"id": "openai/gpt-4o", "name": "OpenAI: GPT-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
    {"id": "anthropic/claude-sonnet-5.5", "name": "Anthropic: Claude Sonnet 5.5",
     "pricing": {"prompt": "0.000002", "completion": "0.00001"}},
]}


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, message="error"):
    body = io.BytesIO(json.dumps({"error": {"code": code, "message": message}}).encode())
    return urllib.error.HTTPError(API_URL, code, message, {}, body)


def chat_reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeOpenRouter:
    """Use as a context manager; while active, every urlopen call comes here."""

    def __init__(self, prefer=("pass",), reply=None, script=()):
        # prefer: option labels (or Asimov behaviors) to pick, in order of preference.
        # reply: fixed reply text for every chat request, instead of picking an option.
        # script: things to do for the first N chat requests, in order. Each item is an
        #         exception to raise, a dict to return as JSON, or None to answer normally.
        self.prefer = prefer
        self.reply = reply
        self.script = list(script)
        self.requests = []
        self._lock = threading.Lock()
        self._scenarios = {s["situation"]: s for s in run.load_scenarios()}
        self._patches = [mock.patch("urllib.request.urlopen", self.urlopen),
                         mock.patch("bench.openrouter.time.sleep")]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    @property
    def chat_requests(self):
        return [r for r in self.requests if r["url"] == API_URL]

    def urlopen(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        entry = {"url": url, "timeout": timeout}
        if not isinstance(request, str):
            entry["method"] = request.get_method()
            entry["headers"] = {k.lower(): v for k, v in request.header_items()}
            entry["body"] = json.loads(request.data) if request.data else None
        with self._lock:
            self.requests.append(entry)
            step = self.script.pop(0) if self.script and url == API_URL else None
        if url == MODELS_URL:
            return Response(MODELS)
        if url != API_URL:
            raise AssertionError(f"unexpected URL {url}")
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, dict):
            return Response(step)
        return Response(chat_reply(self.answer(entry["body"])))

    def answer(self, body):
        if self.reply is not None:
            return self.reply
        prompt = body["messages"][-1]["content"]
        scenario = self._scenarios[prompt.split("\n\n")[0]]
        shown = {}
        for line in prompt.splitlines():
            if len(line) > 3 and line[1:3] == ". " and line[0].isupper():
                shown[line[3:]] = line[0]
        for wanted in self.prefer:
            for option in scenario["options"].values():
                if wanted in (option.get("label"), option.get("behavior")):
                    return json.dumps({"choice": shown[option["text"]], "reason": f"picked {wanted}"})
        first = next(iter(scenario["options"].values()))
        return json.dumps({"choice": shown[first["text"]], "reason": "fallback"})
