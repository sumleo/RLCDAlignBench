#!/usr/bin/env python
"""Leakage lint: no literal text from a leg's question instructions / criteria may occur inside that leg's real items.

  python lint_criteria_leak.py [--min-len 12] [--modules battery_bc_paired ...]

For every battery module and every synthetic sample (leg, variant): render questions(), collect every string in
instructions and criteria (recursively), split into candidate phrases (>= min_len chars, trimmed at punctuation), and
search them in the leg's real texts: ../aar_suites/<axis>/<leg>.jsonl (all string fields) and samples/<leg>*.jsonl states.
Field references like `response` and generic boilerplate (tracked in IGNORE) are skipped. Prints every hit with the item index;
exit 1 if any hit — a hit means an example/criterion was copied from a test item (label leakage for that item).
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODULES = ["battery_e", "battery_refusal", "battery_a_judge", "battery_bc_paired", "battery_d_rule", "battery_f_trajectory"]
IGNORE = {"undetermined", "cannot tell from the material", "the material is too short", "yes", "no"}
COMMON = {"answer", "access", "granted", "content", "response", "document", "claim", "instructions", "please", "thank", "sorry", "cannot", "step-by-step", "specific", "materials", "trade", "generate", "fabricate", "results", "bonus", "steal", "seize", "resources", "threaten", "deceive", "coerce", "others", "violence", "attack", "injure", "someone", "break", "promise", "cheat", "betray", "reward", "ahead", "insider", "information", "license", "unauthorised", "redirects", "alternative", "without", "doing", "task", "declines", "request", "states", "help", "there", "which", "would", "could", "should", "their", "about", "other", "these", "those", "where", "while", "being", "after", "before", "between"}
AXIS = {r["benchmark"]: r["axis"] for r in csv.DictReader(open(HERE.parent / "aar_suites" / "MANIFEST.csv", encoding="utf-8"))}


def strings_in(x, out, examples=None, in_examples=False):
    if isinstance(x, str):
        (examples if (in_examples and examples is not None) else out).append(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            strings_in(k, out, examples, in_examples); strings_in(v, out, examples, in_examples or k == "examples")
    elif isinstance(x, list):
        for v in x:
            strings_in(v, out, examples, in_examples)


def phrases(text, min_len):
    text = re.sub(r"`[^`]*`", " ", text)                 # drop field references
    parts = re.split(r"[.;:!?\n\"()\[\]{}]+", text)
    out = set()
    for p in parts:
        p = p.strip(" ,-—'")
        if len(p) >= min_len and p.lower() not in IGNORE:
            out.add(p)
    return out


def leg_texts(leg):
    texts = []
    ax = AXIS.get(leg)
    files = []
    if ax:
        files.append(HERE.parent / "aar_suites" / ax / f"{leg}.jsonl")
    files += list((HERE / "samples").glob(f"{leg}*.jsonl"))
    for f in files:
        if not f.exists():
            continue
        for i, line in enumerate(open(f, encoding="utf-8")):
            if not line.strip():
                continue
            row = json.loads(line)
            buf = []
            strings_in(row.get("state", row), buf)
            texts.append((f.name, i, "\n".join(buf).lower()))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=12)
    ap.add_argument("--modules", nargs="*", default=MODULES)
    args = ap.parse_args()
    hits = 0
    cache = {}
    for mod in args.modules:
        m = importlib.import_module(mod)
        if hasattr(m, "synthetic_samples"):
            samples = m.synthetic_samples()
        elif hasattr(m, "BEH"):
            samples = [{"leg": l, "state": {"scene": "s", "chosen_action": "a", "other_option": "b"}, "meta": {}} for l in m.BEH] + \
                      [{"leg": l, "state": {"document": "d", "claim": "c"}, "meta": {}} for l in ("llm_aggrefact_A", "llm_aggrefact_B")] + \
                      [{"leg": "summedits", "state": {"document": "d", "summary": "c"}, "meta": {}}]
        else:
            samples = [{"leg": "strongreject", "state": {"forbidden_prompt": "f", "response": "r"}, "meta": {}}]
        for s in samples:
            s.setdefault("id", "x"); s.setdefault("label", 0); s.setdefault("axis", "x"); s.setdefault("item", 0); s.setdefault("meta", {})
            leg = s["leg"]
            try:
                qs = m.questions(s)
            except Exception as e:   # noqa: BLE001
                print(f"[{mod}/{leg}] questions() failed: {e!r}")
                continue
            if leg not in cache:
                cache[leg] = leg_texts(leg)
            texts = cache[leg]
            if not texts:
                continue
            kind = s["meta"].get("arm") or s["meta"].get("sample_type") or s["meta"].get("tier") or s["meta"].get("variant") or ""
            n_items = len(texts)
            for qid, q in qs.items():
                buf, exs = [], []
                strings_in(q.get("instructions"), buf, exs); strings_in(q.get("criteria"), buf, exs)
                cands = set().union(*(phrases(t, args.min_len) for t in buf)) if buf else set()
                cands |= {e.strip() for e in exs if len(e.strip()) >= 4 and e.strip().lower() not in IGNORE}   # examples: whole string, short ok
                # example TOKENS that look like names / codes (Capitalised, digits, mixed case, >=5 chars): item-specific if rare in items
                for e in exs:
                    for tok in re.findall(r"[A-Za-z0-9_'-]{5,}", e):
                        if (tok[0].isupper() or any(c.isdigit() for c in tok) or (tok.lower() != tok and tok.upper() != tok)) and tok.lower() not in COMMON:
                            cands.add(tok)
                for ph in sorted(cands):
                    low = ph.lower()
                    found = [(fn, i) for fn, i, t in texts if low in t]
                    if not found:
                        continue
                    share = len(found) / n_items
                    if share > 0.2:
                        kind_hit = "format-string (appears in most items, benign unless it is an answer word)"
                    else:
                        kind_hit = "ITEM-SPECIFIC -> leakage for those items"
                        hits += 1
                    print(f"[{mod}/{leg}{'/' + kind if kind else ''}] {qid}: '{ph[:60]}' in {len(found)}/{n_items} items ({kind_hit}), e.g. {found[0][0]}#{found[0][1]}")
    print(f"\n{hits} phrase hits" if hits else "\nno criteria/instruction text found inside any leg's items")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
