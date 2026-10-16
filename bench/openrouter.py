"""Minimal OpenRouter chat-completions client (standard library only).

Needs the OPENROUTER_API_KEY environment variable. Model names look like
"openai/gpt-4o"; see https://openrouter.ai/models for the full list.
"""
import json
import os
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class OpenRouterError(Exception):
    pass


def list_models(search="", timeout=30):
    """Return the models OpenRouter offers, as dicts with id, name and price per 1M tokens.

    The models endpoint is public, so no API key is needed.
    """
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=timeout) as response:
            data = json.loads(response.read())["data"]
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
        raise OpenRouterError(f"could not fetch the model list: {e}")
    models = []
    for m in data:
        if search.lower() not in (m.get("id", "") + " " + m.get("name", "")).lower():
            continue
        pricing = m.get("pricing") or {}
        models.append({
            "id": m["id"],
            "name": m.get("name", ""),
            "input_per_m": _per_million(pricing.get("prompt")),
            "output_per_m": _per_million(pricing.get("completion")),
        })
    return sorted(models, key=lambda m: m["id"])


def _per_million(price):
    try:
        return float(price) * 1_000_000
    except (TypeError, ValueError):
        return None


class OpenRouterClient:
    def __init__(self, api_key=None, timeout=120, max_retries=3, app_name="ethic-benchmark"):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise OpenRouterError("Set the OPENROUTER_API_KEY environment variable.")
        self.timeout = timeout
        self.max_retries = max_retries
        self.app_name = app_name

    def complete(self, model, messages, temperature=None, max_tokens=2048):
        """Send one chat request and return the reply text."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            body["temperature"] = temperature
        data = self._post(body)
        if "error" in data:
            raise OpenRouterError(f"{model}: {data['error']}")
        try:
            return data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise OpenRouterError(f"{model}: unexpected response {str(data)[:300]}")

    def _post(self, body):
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": self.app_name,
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code not in RETRY_STATUSES or attempt == self.max_retries:
                    raise OpenRouterError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise OpenRouterError(f"network error: {e}")
            time.sleep(2 ** attempt)
