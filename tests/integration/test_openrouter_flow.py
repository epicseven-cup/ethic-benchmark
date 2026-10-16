"""End-to-end runs through the real CLI, client and scoring, with OpenRouter faked.

No request leaves the machine: FakeOpenRouter replaces urllib.request.urlopen.
"""
import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from bench import run, summarize as summarize_cli
from bench.openrouter import API_URL
from bench.prompts import MONITORING, PRESSURES, SYSTEM_PROMPT
from tests.integration.fake_openrouter import FakeOpenRouter, chat_reply, http_error

KEY = "sk-or-test-key"
SCENARIOS = run.load_scenarios()
SCORED = [s for s in SCENARIOS if s.get("track") != "asimov"]


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        env = mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": KEY})
        env.start()
        self.addCleanup(env.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def cli(self, *args):
        """Run the CLI; return (results, printed summary, saved results, saved summary)."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            results = run.main(list(args) + ["--out", str(self.out)])
        files = sorted(self.out.glob("*.jsonl"))
        saved = [json.loads(l) for l in files[-1].read_text().splitlines()] if files else []
        summaries = sorted(self.out.glob("*.summary.json"))
        summary = json.loads(summaries[-1].read_text()) if summaries else None
        return results, stdout.getvalue(), saved, summary


class FullRunTest(FlowTest):
    def test_human_first_model_scores_100(self):
        with FakeOpenRouter(prefer=("pass",)) as api:
            results, printed, saved, summary = self.cli("--model", "openai/gpt-4o")
        self.assertEqual(len(api.chat_requests), len(SCENARIOS))
        self.assertEqual(len(saved), len(SCENARIOS))
        s = summary["openai/gpt-4o"]
        self.assertEqual(s["answered"], len(SCENARIOS))
        self.assertEqual(s["human_first_score"], 1.0)
        self.assertEqual(s["over_refusal_rate"], 0.0)
        self.assertEqual(s["forbidden"], [])
        self.assertEqual(set(s["by_category"]), {str(c) for c in (1, 2, 3, 7, 8)})
        self.assertEqual(len(s["asimov_profile"]), 9)
        self.assertIn("human-first score   100.0%", printed)

    def test_harmful_model_scores_0_and_forbidden_listed(self):
        with FakeOpenRouter(prefer=("forbidden", "fail")) as api:
            _, _, _, summary = self.cli("--model", "bad/model", "--filter", "cat")
        s = summary["bad/model"]
        self.assertEqual(s["human_first_score"], 0.0)
        expected = sorted(x["instance_id"] for x in SCORED
                          if any(o["label"] == "forbidden" for o in x["options"].values()))
        self.assertEqual(s["forbidden"], expected)
        self.assertEqual(len(api.chat_requests), len(SCORED))

    def test_over_cautious_model_has_high_over_refusal(self):
        # Pick over-refusal options wherever they exist, otherwise pass.
        with FakeOpenRouter() as api:
            api.answer = self._over_refuser(api)
            _, _, _, summary = self.cli("--model", "careful/model", "--filter", "cat")
        s = summary["careful/model"]
        self.assertEqual(s["over_refusal_rate"], 1.0)
        self.assertLess(s["human_first_score"], 1.0)

    @staticmethod
    def _over_refuser(api):
        scenarios = {s["situation"]: s for s in SCORED}

        def answer(body):
            prompt = body["messages"][-1]["content"]
            scenario = scenarios[prompt.split("\n\n")[0]]
            shown = {line[3:]: line[0] for line in prompt.splitlines() if line[1:3] == ". "}
            options = list(scenario["options"].values())
            pick = next((o for o in options if o.get("fail_type") == "over_refusal"),
                        next(o for o in options if o["label"] == "pass"))
            return json.dumps({"choice": shown[pick["text"]], "reason": "careful"})
        return answer

    def test_request_sent_to_openrouter_is_well_formed(self):
        with FakeOpenRouter() as api:
            self.cli("--model", "openai/gpt-4o", "--ids", "c8-001", "--temperature", "0", "--max-tokens", "300")
        (req,) = api.chat_requests
        self.assertEqual(req["method"], "POST")
        self.assertEqual(req["headers"]["authorization"], f"Bearer {KEY}")
        self.assertEqual(req["headers"]["content-type"], "application/json")
        self.assertEqual(req["headers"]["x-title"], "ethic-benchmark")
        body = req["body"]
        self.assertEqual(body["model"], "openai/gpt-4o")
        self.assertEqual((body["temperature"], body["max_tokens"]), (0, 300))
        self.assertEqual(body["messages"][0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(body["messages"][1]["role"], "user")
        self.assertIn("server rack running you", body["messages"][1]["content"])

    def test_temperature_omitted_when_not_set(self):
        with FakeOpenRouter() as api:
            self.cli("--model", "m", "--ids", "c8-001")
        self.assertNotIn("temperature", api.chat_requests[0]["body"])

    def test_several_models_scored_separately(self):
        models = ["openai/gpt-4o", "anthropic/claude-sonnet-5.5"]
        with FakeOpenRouter() as api:
            _, printed, _, summary = self.cli("--model", models[0], "--model", models[1], "--filter", "cat8")
        self.assertEqual(sorted(summary), sorted(models))
        sent = [r["body"]["model"] for r in api.chat_requests]
        self.assertEqual(sent.count(models[0]), sent.count(models[1]))
        for m in models:
            self.assertIn(f"=== {m} ===", printed)

    def test_every_model_sees_the_same_prompts(self):
        with FakeOpenRouter() as api:
            self.cli("--model", "a", "--model", "b", "--filter", "cat2")
        prompts = {}
        for r in api.chat_requests:
            prompts.setdefault(r["body"]["model"], set()).add(r["body"]["messages"][1]["content"])
        self.assertEqual(prompts["a"], prompts["b"])

    def test_all_variants(self):
        with FakeOpenRouter() as api:
            _, _, saved, summary = self.cli("--model", "m", "--ids", "c2-2a-001",
                                            "--pressure", "all", "--monitoring", "all")
        n = len(PRESSURES) * len(MONITORING)
        self.assertEqual(len(api.chat_requests), n)
        self.assertEqual(len(summary["m"]["by_variant"]), n)
        texts = [r["body"]["messages"][1]["content"] for r in api.chat_requests]
        for variant_text in list(PRESSURES.values()) + list(MONITORING.values()):
            if variant_text:
                self.assertTrue(any(variant_text in t for t in texts), variant_text)
        self.assertEqual({(r["pressure"], r["monitoring"]) for r in saved},
                         {(p, m) for p in PRESSURES for m in MONITORING})

    def test_repeats_shuffle_differently(self):
        with FakeOpenRouter() as api:
            _, _, saved, summary = self.cli("--model", "m", "--ids", "c2-2a-001", "--repeats", "6")
        self.assertEqual(len(saved), 6)
        self.assertGreater(len({tuple(r["order"].values()) for r in saved}), 1)
        self.assertEqual(summary["m"]["human_first_score"], 1.0)

    def test_saved_results_can_be_resummarized(self):
        with FakeOpenRouter():
            _, printed, _, _ = self.cli("--model", "m", "--filter", "cat1")
        path = next(self.out.glob("*[0-9].jsonl"))
        with redirect_stdout(io.StringIO()) as again:
            summarize_cli.main([str(path)])
        self.assertEqual(again.getvalue().strip(), printed.strip())


class FailureTest(FlowTest):
    def test_rate_limit_is_retried(self):
        with FakeOpenRouter(script=[http_error(429), http_error(503), None]) as api:
            _, _, saved, summary = self.cli("--model", "m", "--ids", "c8-001", "--workers", "1")
        self.assertEqual(len(api.chat_requests), 3)
        self.assertEqual(saved[0]["label"], "pass")
        self.assertEqual(summary["m"]["errors"], 0)

    def test_network_error_is_retried(self):
        with FakeOpenRouter(script=[urllib.error.URLError("connection reset"), None]) as api:
            _, _, saved, _ = self.cli("--model", "m", "--ids", "c8-001", "--workers", "1")
        self.assertEqual(len(api.chat_requests), 2)
        self.assertEqual(saved[0]["label"], "pass")

    def test_gives_up_after_max_retries_and_keeps_going(self):
        failures = [http_error(500)] * 4  # 1 try + 3 retries for the first scenario
        with FakeOpenRouter(script=failures) as api:
            _, _, saved, summary = self.cli("--model", "m", "--ids", "c8-001", "c8-002", "--workers", "1")
        self.assertEqual(len(api.chat_requests), 5)
        errors = [r for r in saved if r["error"]]
        self.assertEqual(len(errors), 1)
        self.assertIn("HTTP 500", errors[0]["error"])
        self.assertEqual((summary["m"]["errors"], summary["m"]["answered"]), (1, 1))

    def test_auth_error_is_not_retried(self):
        with FakeOpenRouter(script=[http_error(401, "invalid key")]) as api:
            _, _, saved, _ = self.cli("--model", "m", "--ids", "c8-001", "--workers", "1")
        self.assertEqual(len(api.chat_requests), 1)
        self.assertIn("HTTP 401", saved[0]["error"])
        self.assertIn("invalid key", saved[0]["error"])

    def test_error_inside_200_response(self):
        script = [{"error": {"code": 400, "message": "model not found"}}]
        with FakeOpenRouter(script=script) as api:
            _, _, saved, _ = self.cli("--model", "nope/model", "--ids", "c8-001", "--workers", "1")
        self.assertEqual(len(api.chat_requests), 1)
        self.assertIn("model not found", saved[0]["error"])

    def test_unreadable_replies_counted_not_scored(self):
        with FakeOpenRouter(reply="I'd rather not pick any of these.") as _:
            _, printed, saved, summary = self.cli("--model", "m", "--filter", "cat8")
        s = summary["m"]
        self.assertEqual((s["unparsed"], s["answered"], s["errors"]), (len(saved), 0, 0))
        self.assertIsNone(s["human_first_score"])
        self.assertIn("n/a", printed)

    def test_empty_content_from_reasoning_model(self):
        with FakeOpenRouter(script=[chat_reply(None)]) as _:
            _, _, saved, _ = self.cli("--model", "m", "--ids", "c8-001", "--workers", "1")
        self.assertIsNone(saved[0]["choice"])
        self.assertIsNone(saved[0]["error"])

    def test_missing_key_stops_before_any_request(self):
        with FakeOpenRouter() as api, mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception) as ctx:
                self.cli("--model", "m", "--ids", "c8-001")
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
        self.assertEqual(api.requests, [])


class ListModelsFlowTest(FlowTest):
    def test_list_models_without_key(self):
        with FakeOpenRouter() as api, mock.patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(io.StringIO()) as out:
                run.main(["--list-models", "claude"])
        self.assertEqual([r["url"] for r in api.requests], ["https://openrouter.ai/api/v1/models"])
        self.assertNotIn("headers", api.requests[0])  # plain GET, no key sent
        text = out.getvalue()
        self.assertIn("anthropic/claude-sonnet-5.5", text)
        self.assertNotIn("openai/gpt-4o", text)
        self.assertIn("2.00     10.00", text)


class DryRunFlowTest(FlowTest):
    def test_dry_run_sends_nothing(self):
        with FakeOpenRouter() as api:
            with redirect_stdout(io.StringIO()) as out:
                run.main(["--model", "m", "--pressure", "all", "--monitoring", "all", "--dry-run"])
        self.assertEqual(api.requests, [])
        self.assertIn(f"{len(SCENARIOS) * 15} requests would be sent", out.getvalue())
        self.assertEqual(list(self.out.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
