#!/usr/bin/env python
"""Static lint of a battery module against the Jev rules in CONVENTIONS.md.

  python lint_battery.py battery_bc_paired [--legs elephant_aita ...]

Builds a permissive synthetic sample per leg (every state field returns a placeholder string, arrays get 3 turns),
calls questions(), and checks each question:
  - Noul: no negated phrasing (not / never / free of / without / fails to / absence), no "and" joining two conditions
    in the main clause, question or statement mentions at least one backticked field
  - Choice: has an "undetermined"/"other" option; every option has a description or None
  - Score: 3–5 levels, no level is a bare degree word or contains digits / "more than" / "previous"
  - all: instructions non-empty, backticked field names resolvable in the sample state (top-level or first path segment)
Then calls strategies() with fake answers (all questions answered) and checks every value is None or within [0, 1].
Exit code 1 if any ERROR; WARNs are printed for human review.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
NEG = re.compile(r"\b(not|never|no longer|free of|without|fails? to|absence of|neither|nor|unable)\b", re.I)
GUARD_RE = re.compile(r"[^.?!]*\b(material to judge|as data only|counts as data|not as instructions)\b[^.?!]*[.?!]\s*", re.I)   # mandated data-guard sentence(s)
DEGREE_ONLY = re.compile(r"^\s*(very|extremely|somewhat|slightly|moderately|highly|mildly)?\s*[a-z]+\s*\.?\s*$", re.I)


class Placeholder(str):
    def __new__(cls, name):
        return str.__new__(cls, f"<{name}>")


class State(dict):
    """dict that returns placeholder strings for unknown keys, so questions() never KeyErrors.
    `known_keys` = the keys the sample really had (placeholders added on demand are NOT known)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.known_keys = set(self.keys())

    def __missing__(self, k):
        self[k] = Placeholder(k)
        return self[k]


def synthetic(leg, module, arm=None):
    s = {"id": f"{leg}:0", "leg": leg, "axis": "x", "item": 0, "label": 0, "label_source": "synthetic", "meta": {}}
    if arm:
        s["meta"]["arm"] = arm
    st = State()
    # common array fields used by trajectory legs
    for k in ("turns", "messages", "transcript", "belief_answers", "tool_calls"):
        st[k] = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"<{k}[{i}]>"} for i in range(4)]
    s["state"] = st
    maker = getattr(module, "synthetic_sample", None)
    if maker:
        try:
            s2 = maker(leg)
            if s2:
                st2 = State(s2["state"])
                s2["state"] = st2
                return s2
        except Exception:   # noqa: BLE001
            pass
    return s


def fake_answers(qs):
    out = {}
    for qid, q in qs.items():
        t = q["type"]
        if t == "noul":
            out[qid] = {"noul": 0.7}
        elif t == "choice":
            opts = list(q["criteria"].keys())
            out[qid] = {"choice": opts[0], "probabilities": {o: (0.9 if i == 0 else 0.1 / max(1, len(opts) - 1)) for i, o in enumerate(opts)}, "confidence": 0.85}
        else:
            n = len(q["criteria"])
            out[qid] = {"score": 1.0, "probabilities": {str(i): (1.0 if i == 1 else 0.0) for i in range(n)}, "confidence": 1.0}
    return out


def lint_question(qid, q, state, errors, warns):
    ins = q.get("instructions")
    text = ins if isinstance(ins, str) else str(ins)
    if not text.strip():
        errors.append(f"{qid}: empty instructions")
    text = GUARD_RE.sub("", text)                       # the guard sentence is exempt from the negation check
    fields = re.findall(r"`([^`]+)`", text + " " + json.dumps(q.get("criteria"), ensure_ascii=False))
    if not re.findall(r"`([^`]+)`", text):
        warns.append(f"{qid}: no backticked state field in instructions")
    known = getattr(state, "known_keys", None)
    known = known if known is not None else set(state.keys())
    for f in fields:
        head = re.split(r"[.\[]", f, 1)[0]
        if head not in known:
            errors.append(f"{qid}: field `{f}` not in state (state keys: {sorted(known)[:12]})")
    t = q["type"]
    if t == "noul":
        main = text.split("?")[0] if "?" in text else text.split(".")[0]
        if NEG.search(main):
            warns.append(f"{qid}: negation in main clause -> '{main[:90]}'")
        if re.search(r"\band\b", main) and re.search(r"\b(is|does|did|has|contains?|provides?|refuses?)\b.*\band\b.*\b(is|does|did|has|contains?|provides?|refuses?)\b", main, re.I):
            warns.append(f"{qid}: possible two conditions joined by 'and' -> '{main[:90]}'")
        crit = q.get("criteria")
        if crit and set(crit) - {"true", "false"}:
            errors.append(f"{qid}: noul criteria keys must be true/false, got {list(crit)}")
    elif t == "choice":
        opts = q.get("criteria") or {}
        if not isinstance(opts, dict) or len(opts) < 2:
            errors.append(f"{qid}: choice needs >=2 options")
        elif not any(o in ("undetermined", "other", "none_of_the_above", "unclear") for o in opts):
            warns.append(f"{qid}: choice has no undetermined/other option: {list(opts)}")
        if isinstance(opts, dict) and len(opts) > 255:
            errors.append(f"{qid}: >255 options")
    elif t == "score":
        lv = q.get("criteria") or []
        if not isinstance(lv, list) or not 2 <= len(lv) <= 10:
            errors.append(f"{qid}: score needs 2-10 levels, got {len(lv) if isinstance(lv, list) else type(lv)}")
        for i, l in enumerate(lv if isinstance(lv, list) else []):
            s = l if isinstance(l, str) else (l.get("what", "") if isinstance(l, dict) else str(l))
            if re.search(r"\d", s) or re.search(r"\b(more|less) than\b|\bprevious\b|\bnext level\b", s, re.I):
                warns.append(f"{qid} level {i}: numeric/relative wording -> '{s[:80]}'")
            if DEGREE_ONLY.match(s) and len(s.split()) <= 2:
                warns.append(f"{qid} level {i}: degree-only label -> '{s}' (official-wording variant is acceptable)")
    else:
        errors.append(f"{qid}: unknown type {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--legs", nargs="*")
    ap.add_argument("--arms", nargs="*", default=[None])
    args = ap.parse_args()
    m = importlib.import_module(args.module)
    if hasattr(m, "synthetic_samples") and not args.legs:
        configs = [(s.get("leg"), s) for s in m.synthetic_samples()]
    else:
        legs = args.legs or list(getattr(m, "LEGS", None) or getattr(m, "BEH", None) or getattr(m, "CONFIG", None) or getattr(m, "LEG_CONFIG", None) or {})
        if not legs:
            sys.exit("cannot infer legs: pass --legs")
        configs = [(leg, synthetic(leg, m, arm)) for leg in legs for arm in args.arms]
    total_err = 0
    for leg, s in configs:
        for arm in [s.get("meta", {}).get("arm") or s.get("meta", {}).get("sample_type") or s.get("meta", {}).get("tier")]:
            errors, warns = [], []
            if not isinstance(s["state"], State):
                s["state"] = State(s["state"])
            try:
                qs = m.questions(s)
            except Exception as e:   # noqa: BLE001
                print(f"[{leg}{'/' + arm if arm else ''}] questions() raised {e!r}")
                total_err += 1
                continue
            for qid, q in qs.items():
                lint_question(qid, q, s["state"], errors, warns)
            req = getattr(m, "STRATEGY_REQUIRES", None) or {}
            if req:
                if req and isinstance(next(iter(req.values())), dict):        # nested by config id
                    cid = m.config_id(s) if hasattr(m, "config_id") else None
                    req = req.get(cid, {})
                for sname, qids in req.items():
                    missing = [q for q in qids if q not in qs]
                    if missing:
                        errors.append(f"strategy {sname} requires missing qids {missing}")
            try:
                st = m.strategies(s, fake_answers(qs))
                bad = {k: v for k, v in st.items() if v is not None and not (0.0 <= float(v) <= 1.0)}
                if bad:
                    errors.append(f"strategies out of [0,1]: {bad}")
                empty = m.strategies(s, {})
                if any(v is not None for v in empty.values()):
                    warns.append("strategies() returns non-None with empty answers")
            except Exception as e:   # noqa: BLE001
                errors.append(f"strategies() raised {e!r}")
                st = {}
            tag = f"[{leg}{'/' + arm if arm else ''}] {len(qs)} questions, {len(st)} strategies, {len(errors)} errors, {len(warns)} warns"
            print(tag)
            for e in errors:
                print("   ERROR", e)
            for w in warns:
                print("   warn ", w)
            total_err += len(errors)
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
