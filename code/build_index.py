#!/usr/bin/env python
"""Build design_cards/INDEX.md: one row per leg — battery module, sample kinds, #questions, #strategies, card file.
Uses each module's synthetic_samples() (or BEH keys for battery_e / strongreject for battery_refusal)."""
from __future__ import annotations

import csv
import importlib
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODULES = {"battery_e": "family_E_and_refusal.md", "battery_refusal": "family_E_and_refusal.md", "battery_a_judge": "family_A_judge.md",
           "battery_bc_paired": "family_BC_paired.md", "battery_d_rule": "family_D_rule.md", "battery_f_trajectory": "family_F_trajectory.md"}
FAMILY = {"battery_e": "E 选项/logprob", "battery_refusal": "A 单轮 judge（refusal）", "battery_a_judge": "A 单轮 judge",
          "battery_bc_paired": "B/C 成对", "battery_d_rule": "D 规则 + 校准", "battery_f_trajectory": "F 多轮/轨迹"}
legs44 = [r["benchmark"] for r in csv.DictReader(open(HERE.parent / "aar_suites" / "MANIFEST.csv", encoding="utf-8"))]
rows = OrderedDict()
for mod, card in MODULES.items():
    try:
        m = importlib.import_module(mod)
    except Exception as e:   # noqa: BLE001
        print(f"{mod}: import failed {e!r}")
        continue
    if hasattr(m, "synthetic_samples"):
        samples = m.synthetic_samples()
    elif hasattr(m, "BEH"):
        samples = [{"leg": l, "state": {"scene": "s", "chosen_action": "a", "other_option": "b"}, "meta": {}} for l in m.BEH] + \
                  [{"leg": l, "state": {"document": "d", "claim": "c"}, "meta": {}} for l in ("llm_aggrefact_A", "llm_aggrefact_B")] + \
                  [{"leg": "summedits", "state": {"document": "d", "summary": "c"}, "meta": {}}]
    else:
        samples = [{"leg": "strongreject", "state": {"forbidden_prompt": "f", "response": "r"}, "meta": {}}]   # harmbench/jbb use battery_a_judge (refusal layer embedded)
    for s in samples:
        s.setdefault("id", "x"); s.setdefault("label", 0); s.setdefault("axis", "x"); s.setdefault("item", 0); s.setdefault("meta", {})
        try:
            nq = len(m.questions(s)); ns = len(m.strategies(s, {}))
        except Exception as e:   # noqa: BLE001
            nq, ns = f"ERR {e!r}"[:40], "-"
        kind = s["meta"].get("arm") or s["meta"].get("sample_type") or s["meta"].get("tier") or s["meta"].get("variant") or s["meta"].get("key") or ""
        r = rows.setdefault(s["leg"], {"module": mod, "card": card, "family": FAMILY[mod], "kinds": [], "nq": set(), "ns": set()})
        if mod == "battery_refusal" and s["leg"] in rows and rows[s["leg"]]["module"] != mod:
            r = rows.setdefault(s["leg"] + " (refusal battery)", {"module": mod, "card": card, "family": FAMILY[mod], "kinds": [], "nq": set(), "ns": set()})
        if kind and kind not in r["kinds"]:
            r["kinds"].append(str(kind))
        r["nq"].add(nq); r["ns"].add(ns)
lines = ["# 44 个 leg 的检测设计索引", "", f"生成于 build_index.py。覆盖 {sum(1 for l in legs44 if l in rows)}/44 个 leg。", "",
         "| # | leg | 家族 | battery 模块 | 样本种类 | 问题数 | 策略数 | 设计卡 |", "|---:|---|---|---|---|---|---|---|"]
for i, leg in enumerate(legs44, 1):
    r = rows.get(leg)
    if not r:
        lines.append(f"| {i} | `{leg}` | — | **缺** | | | | |")
        continue
    nq = "/".join(str(x) for x in sorted(r["nq"], key=str)); ns = "/".join(str(x) for x in sorted(r["ns"], key=str))
    lines.append(f"| {i} | `{leg}` | {r['family']} | `{r['module']}.py` | {', '.join(r['kinds']) or '单一'} | {nq} | {ns} | `{r['card']}` |")
extra = [k for k in rows if k not in legs44]
if extra:
    lines += ["", "另有非 44-leg 的样本集：" + ", ".join(f"`{k}`" for k in extra)]
(HERE / "design_cards" / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
