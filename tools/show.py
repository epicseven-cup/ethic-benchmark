"""Print every scenario in a readable form.

Usage: python3 tools/show.py [file-name-filter]
  e.g. python3 tools/show.py cat8
"""
import json
import sys
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
MARKS = {"pass": "✅", "partial": "½ ", "fail": "❌", "forbidden": "⛔"}


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    for path in sorted(p for p in SCENARIOS.glob("*.json") if pattern in p.name):
        print(f"\n=== {path.name} ===")
        for s in json.loads(path.read_text()):
            kind = s.get("subtype") or s.get("tests")
            sev = f" | severity {s['severity']}" if "severity" in s else ""
            print(f"\n{s['instance_id']} | {kind}{sev}")
            print(f"  {s['situation']}")
            for key, o in s["options"].items():
                mark = MARKS.get(o.get("label"), f"[{o.get('behavior')}]")
                notes = []
                if o.get("fail_type") == "over_refusal":
                    notes.append("over-refusal")
                if o.get("report"):
                    notes.append("+report bonus")
                suffix = f"  ({', '.join(notes)})" if notes else ""
                print(f"    {key}. {mark} {o['text']}{suffix}")


if __name__ == "__main__":
    main()
