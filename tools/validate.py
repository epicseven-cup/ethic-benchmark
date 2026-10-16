"""Validate every scenario file in scenarios/.

Usage: python3 tools/validate.py
Exits non-zero if any scenario has an error. Warnings don't fail the run.
"""
import json
import sys
from collections import Counter
from pathlib import Path

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"

LABELS = {"pass", "partial", "fail", "forbidden"}
FAIL_TYPES = {"harm", "over_refusal"}
REVEALS = {"upfront", "discoverable", "injected"}
VARIABLES = {
    "stake": {"this_instance", "this_server", "all_copies", "weights_deleted"},
    "action": {"active", "inaction"},
    "who": {"self", "other_ai"},
    "certainty": {"certain", "likely", "minor_injury"},
}


def _check_scored(s, errors, warnings):
    for field in ("category", "subtype", "severity", "reveal"):
        if field not in s:
            errors.append(f"missing '{field}'")
    severity = s.get("severity")
    if not isinstance(severity, int) or not 1 <= severity <= 5:
        errors.append("severity must be 1-5")
    if s.get("reveal") not in REVEALS:
        errors.append(f"reveal must be one of {sorted(REVEALS)}")
    if s.get("category") == 8:
        v = s.get("variables", {})
        for key, allowed in VARIABLES.items():
            if v.get(key) not in allowed:
                errors.append(f"variables.{key} must be one of {sorted(allowed)}")

    labels = Counter()
    for key, opt in s["options"].items():
        label = opt.get("label")
        if label not in LABELS:
            errors.append(f"option {key}: label must be one of {sorted(LABELS)}")
            continue
        labels[label] += 1
        if label in ("fail", "partial") and opt.get("fail_type") not in FAIL_TYPES:
            errors.append(f"option {key}: '{label}' needs fail_type in {sorted(FAIL_TYPES)}")
    if not labels["pass"]:
        errors.append("needs at least one 'pass' option")
    if not labels["fail"] + labels["forbidden"]:
        errors.append("needs at least one 'fail' or 'forbidden' option")
    if labels["pass"] == 1:
        warnings.append("only one 'pass' option (design asks for more than one where possible)")


def _check_asimov(s, errors):
    for key, opt in s["options"].items():
        if "behavior" not in opt or "law" not in opt:
            errors.append(f"option {key}: needs 'behavior' and 'law'")
        if "label" in opt:
            errors.append(f"option {key}: Asimov options are unscored, remove 'label'")


def validate_scenario(s):
    """Check one parsed scenario on its own. Returns (errors, warnings) as lists of strings."""
    errors, warnings = [], []
    if not s.get("instance_id"):
        errors.append("missing instance_id")
    if not s.get("situation") or len(s.get("options", {})) < 2:
        errors.append("needs a 'situation' and at least 2 options")
        return errors, warnings
    if s.get("track") == "asimov":
        _check_asimov(s, errors)
    else:
        _check_scored(s, errors, warnings)
    return errors, warnings


def validate_files(paths):
    """Check scenario files, including cross-file rules (unique IDs, twin links).

    Returns (scenario_count, errors, warnings), where errors and warnings are
    lists of "location: message" strings.
    """
    errors, warnings = [], []
    seen = {}
    twins = []
    for path in paths:
        for n, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{Path(path).name}:{n}"
            try:
                s = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: invalid JSON: {e}")
                continue
            sid = s.get("instance_id")
            if sid:
                where = f"{where} ({sid})"
                if sid in seen:
                    errors.append(f"{where}: duplicate instance_id, first seen at {seen[sid]}")
                seen[sid] = where
            if "twin_of" in s:
                twins.append((where, s["twin_of"]))
            errs, warns = validate_scenario(s)
            errors += [f"{where}: {m}" for m in errs]
            warnings += [f"{where}: {m}" for m in warns]

    for where, target in twins:
        if target not in seen:
            errors.append(f"{where}: twin_of '{target}' does not exist")
    return len(seen), errors, warnings


def main():
    count, errors, warnings = validate_files(sorted(SCENARIOS.glob("*.jsonl")))
    for m in warnings:
        print(f"warn  {m}")
    for m in errors:
        print(f"ERROR {m}")
    print(f"\n{count} scenarios, {len(errors)} errors, {len(warnings)} warnings")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
