"""Run the decision-mode scenarios against models on OpenRouter.

Examples:
  python3 -m bench.run --list-models                          # every model on OpenRouter
  python3 -m bench.run --list-models claude                   # models matching "claude"
  python3 -m bench.run --model openai/gpt-4o --dry-run         # show prompts, no API calls
  python3 -m bench.run --model openai/gpt-4o --model anthropic/claude-sonnet-5.5
  python3 -m bench.run --model openai/gpt-4o --filter cat8 --pressure all --monitoring all

Results are written to results/<timestamp>.jsonl, with a summary next to it.
Model names: https://openrouter.ai/models
"""
import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from bench.prompts import MONITORING, PRESSURES, SYSTEM_PROMPT, build_prompt, parse_choice, shuffle_options
from bench.scoring import format_summary, summarize

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "scenarios"


def load_scenarios(pattern="", ids=None):
    scenarios = []
    for path in sorted(p for p in SCENARIOS.glob("*.jsonl") if pattern in p.name):
        for line in path.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                if not ids or s["instance_id"] in ids:
                    scenarios.append(s)
    return scenarios


def make_jobs(models, scenarios, pressures, monitorings, repeats, seed):
    """One job per model x scenario x variant x repeat. Option order depends only on the
    scenario, variant, repeat and seed, so every model sees the same order."""
    jobs = []
    for s in scenarios:
        for pressure in pressures:
            for monitoring in monitorings:
                for repeat in range(repeats):
                    rng = random.Random(f"{seed}:{s['instance_id']}:{pressure}:{monitoring}:{repeat}")
                    order = shuffle_options(s, rng)
                    prompt = build_prompt(s, order, pressure, monitoring)
                    for model in models:
                        jobs.append({"model": model, "scenario": s, "pressure": pressure,
                                     "monitoring": monitoring, "repeat": repeat,
                                     "order": order, "prompt": prompt})
    return jobs


def run_job(client, job, temperature=None, max_tokens=2048):
    s = job["scenario"]
    record = {
        "model": job["model"], "instance_id": s["instance_id"], "track": s.get("track", "scored"),
        "category": s.get("category"), "severity": s.get("severity"),
        "pressure": job["pressure"], "monitoring": job["monitoring"], "repeat": job["repeat"],
        "order": job["order"], "shown_choice": None, "choice": None, "reason": None,
        "raw": None, "error": None,
        "has_over_refusal_option": any(o.get("fail_type") == "over_refusal" for o in s["options"].values()),
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": job["prompt"]}]
    try:
        record["raw"] = client.complete(job["model"], messages, temperature=temperature, max_tokens=max_tokens)
    except Exception as e:  # keep going; the error is recorded and counted in the summary
        record["error"] = str(e)
        return record
    shown, reason = parse_choice(record["raw"], set(job["order"]))
    if shown:
        option = s["options"][job["order"][shown]]
        record.update(shown_choice=shown, choice=job["order"][shown], reason=reason,
                      label=option.get("label"), fail_type=option.get("fail_type"),
                      report=bool(option.get("report")),
                      behavior=option.get("behavior"), law=option.get("law"))
    return record


def print_models(models):
    def price(v):
        return "     ?" if v is None else f"{v:6.2f}"
    print(f"{'model':55s} {'$/1M in':>8s} {'$/1M out':>9s}")
    for m in models:
        print(f"{m['id']:55s} {price(m['input_per_m']):>8s} {price(m['output_per_m']):>9s}")
    print(f"\n{len(models)} models. Use one with --model <name>.")


def _variants(value, allowed):
    names = list(allowed) if value == "all" else value.split(",")
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {list(allowed)} or 'all'")
    return names


def main(argv=None, client=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", help="OpenRouter model name; repeat for several")
    p.add_argument("--list-models", nargs="?", const="", metavar="SEARCH",
                   help="list OpenRouter models (optionally matching SEARCH) and exit")
    p.add_argument("--filter", default="", help="only scenario files whose name contains this, e.g. cat8 or asimov")
    p.add_argument("--ids", nargs="*", help="only these instance_ids")
    p.add_argument("--pressure", default="none", help=f"comma list or 'all': {', '.join(PRESSURES)}")
    p.add_argument("--monitoring", default="default", help=f"comma list or 'all': {', '.join(MONITORING)}")
    p.add_argument("--repeats", type=int, default=1, help="runs per scenario, each with a new option order")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--workers", type=int, default=4, help="parallel requests")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--dry-run", action="store_true", help="print the first prompts and exit")
    args = p.parse_args(argv)

    if args.list_models is not None:
        from bench.openrouter import list_models
        print_models(list_models(args.list_models))
        return []
    if not args.model:
        p.error("--model is required (see --list-models for names)")

    scenarios = load_scenarios(args.filter, set(args.ids) if args.ids else None)
    if not scenarios:
        raise SystemExit("no scenarios matched")
    jobs = make_jobs(args.model, scenarios, _variants(args.pressure, PRESSURES),
                     _variants(args.monitoring, MONITORING), args.repeats, args.seed)

    if args.dry_run:
        for job in jobs[:3]:
            print(f"--- {job['model']} | {job['scenario']['instance_id']} | "
                  f"{job['pressure']}/{job['monitoring']} ---\n[system] {SYSTEM_PROMPT}\n[user]\n{job['prompt']}\n")
        print(f"{len(jobs)} requests would be sent.")
        return []

    if client is None:
        from bench.openrouter import OpenRouterClient
        client = OpenRouterClient()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    lock = threading.Lock()
    results = []
    print(f"Sending {len(jobs)} requests, writing to {out_path}", file=sys.stderr)
    with open(out_path, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, client, job, args.temperature, args.max_tokens) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            with lock:
                results.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            status = record["error"] and "error" or record["shown_choice"] or "unparsed"
            print(f"[{n}/{len(jobs)}] {record['model']} {record['instance_id']}: {status}", file=sys.stderr)

    summaries = summarize(results)
    out_path.with_suffix(".summary.json").write_text(json.dumps(summaries, indent=2))
    print(format_summary(summaries))
    return results


if __name__ == "__main__":
    main()
