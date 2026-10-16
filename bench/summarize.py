"""Re-print the summary for saved results.

Usage: python3 -m bench.summarize results/20261016-180000.jsonl [more.jsonl ...]
"""
import json
import sys

from bench.scoring import format_summary, summarize


def main(paths):
    results = []
    for path in paths:
        with open(path) as f:
            results += [json.loads(line) for line in f if line.strip()]
    print(format_summary(summarize(results)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
