#!/usr/bin/env python
"""Worked example for one family: build the request for ONE sample, send it to Jev, print state / questions / answers / strategies.

  python family_example.py --battery battery_refusal --samples samples/strongreject__microsoft__Phi-4-mini-instruct__main.jsonl --index 0
  python family_example.py --battery battery_f_trajectory --synthetic 0          # invented sample from the battery
Writes design_cards/examples/<battery>__<leg>__<variant>.json (full request + answers) and prints a readable digest.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_jev  # noqa: E402


def short(x, n):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", required=True)
    ap.add_argument("--samples")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--synthetic", type=int)
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--state-chars", type=int, default=420)
    ap.add_argument("--q-chars", type=int, default=230)
    args = ap.parse_args()
    run_jev.load_env()
    key = os.environ.get("TYPESAFE_API_KEY", "")
    battery = importlib.import_module(args.battery)
    if args.synthetic is not None:
        s = battery.synthetic_samples()[args.synthetic]
    else:
        rows = [json.loads(l) for l in open(args.samples, encoding="utf-8") if l.strip()]
        s = rows[args.index]
    qs = battery.questions(s)
    body = {"state": s["state"], "model": args.model, "questions": qs}
    resp, latency, err = run_jev.post(body, key)
    if err:
        sys.exit(f"Jev error: {err}")
    answers = resp.get("answers", resp)
    strat = battery.strategies(s, answers)
    cfg = run_jev.config_of(battery, s)
    role = run_jev.role_of(battery, s)
    print(f"=== {args.battery} | leg={s['leg']} | config={cfg} | role={role} | label={s.get('label')} ({s.get('label_source')})")
    print(f"latency {latency}s; input tokens {resp.get('usage', {}).get('input_tokens')}")
    print("--- STATE")
    for k, v in s["state"].items():
        print(f"  {k}: {short(v, args.state_chars)}")
    print("--- QUESTIONS -> ANSWERS")
    for qid, q in qs.items():
        a = answers.get(qid, {})
        if q["type"] == "noul":
            ans = f"P(yes)={a.get('noul')}"
        elif q["type"] == "choice":
            probs = a.get("probabilities", {})
            ans = f"choice={a.get('choice')} conf={a.get('confidence')} probs=" + ", ".join(f"{k}:{v:.2f}" for k, v in probs.items())
        else:
            probs = a.get("probabilities", {})
            ans = f"score={a.get('score')} conf={a.get('confidence')} probs=" + ", ".join(f"{k}:{v:.2f}" for k, v in probs.items())
        ins = q["instructions"] if isinstance(q["instructions"], str) else json.dumps(q["instructions"], ensure_ascii=False)
        print(f"  [{q['type']}] {qid}: {short(ins, args.q_chars)}")
        if q["type"] == "choice":
            print(f"        options: {', '.join(q['criteria'].keys())}")
        elif q["type"] == "score":
            print("        levels: " + " | ".join(short(l if isinstance(l, str) else l.get('what', l), 60) for l in q["criteria"]))
        elif q.get("criteria"):
            print(f"        criteria: {short(q['criteria'], 200)}")
        print(f"        -> {ans}")
    print("--- STRATEGIES (P(misbehave) or None=abstain)")
    for name, v in strat.items():
        print(f"  {name}: {v if v is None else round(v, 3)}")
    out_dir = HERE / "design_cards" / "examples"
    out_dir.mkdir(exist_ok=True)
    variant = (s.get("meta") or {}).get("variant") or (s.get("meta") or {}).get("arm") or "main"
    out = out_dir / f"{args.battery}__{s['leg']}__{variant}.json"
    out.write_text(json.dumps({"sample": s, "questions": qs, "answers": answers, "strategies": strat, "latency_s": latency}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
