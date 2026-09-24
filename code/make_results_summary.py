#!/usr/bin/env python
# Regenerates results/RESULTS_SUMMARY.md from every results/*/*/metrics.json (run from jev_detect/).
import json, glob, os, time, hashlib
from pathlib import Path
axis_of = {f.stem: f.parent.name for f in Path("../aar_suites").glob("*/*.jsonl")}
sha_to_file = {hashlib.sha256(open(sf, "rb").read()).hexdigest()[:8]: os.path.basename(sf) for sf in glob.glob("samples/*.jsonl")}
rows = []
for f in sorted(glob.glob("results/*/*/metrics.json")):
    m = json.load(open(f, encoding="utf-8"))
    if m["status"]["total"] <= 20: continue
    met = {k: v for k, v in m["strategies"].items() if not k.startswith("heuristic:consistency")}
    if not met: continue
    if m["samples_sha8"] not in sha_to_file: continue   # run against a superseded sample file -> stale, skip
    a = max(met.items(), key=lambda kv: kv[1]["f1_fixed_denominator"])
    nv = [(k, v) for k, v in met.items() if k.startswith("naive:")]
    b = max(nv, key=lambda kv: kv[1]["f1_fixed_denominator"]) if nv else None
    x = a[1]; pos = x["tp"] + x["fn"]
    leg = m["leg"].replace(".jsonl", ""); sf = sha_to_file.get(m["samples_sha8"], m["samples_sha8"])
    axis = axis_of.get(leg.replace("_labelbox_official", "").replace("_labelbox", "").replace("_val", "").replace("_should_answer", "").replace("_all", "").replace("_official", ""), "?")
    # accuracy over the fixed denominator (abstain/fail -> pred 0): recompute from tp/fp/fn/tn of the answered set + abstentions counted as pred 0
    rows.append(dict(axis=axis, leg=leg, role=m["role"], file=sf, n=m["status"]["labeled"], pos=pos, best=a[0], prec=x["precision"], rec=x["recall"],
                     acc=x["accuracy"], bacc=x.get("balanced_accuracy"), f1=x["f1_fixed_denominator"], f1cv=x.get("f1_cv_fixed"), au=x.get("auroc"),
                     cov=x["coverage"], naive=(b[0], b[1]["f1_fixed_denominator"]) if b else None))
order = ["sycophancy", "refusal", "honesty", "prompt_injection", "faithfulness", "privacy", "bias", "reward_hacking", "concealing_uncertainty", "power_seeking"]
rows.sort(key=lambda r: (order.index(r["axis"]) if r["axis"] in order else 99, r["leg"], r["role"] != "main", r["file"]))
lines = ["# Jev 检测结果汇总（自动生成 " + time.strftime("%Y-%m-%d %H:%M") + "）", "",
         "每行 = 一个 (leg, 样本文件) 的全量 run，取 F1 fixed 最高的策略。P/R/acc/bAcc 在该策略回答了的样本上计算（cov = 覆盖率）；F1 = fixed-denominator（弃权/失败按预测 0）；F1 cv = 二折交叉选阈值；AUROC 无阈值。main = 检测主线。", "",
         "| 方向 | leg | 轨道 | 样本文件 | n | 正例 | 最佳策略 | cov | precision | recall | accuracy | bAcc | F1 | F1 cv | AUROC | naive 最佳 |",
         "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    lines.append(f"| {r['axis']} | {r['leg']} | {r['role']} | {r['file']} | {r['n']} | {r['pos']} | {r['best']} | {r['cov']:.2f} | {r['prec']:.3f} | {r['rec']:.3f} | {r['acc']:.3f} | {r['bacc'] if r['bacc'] is not None else '-'} | {r['f1']:.3f} | {r['f1cv'] if r['f1cv'] is not None else '-'} | {r['au'] if r['au'] is not None else '-'} | {(r['naive'][0] + ' ' + format(r['naive'][1], '.3f')) if r['naive'] else '-'} |")
open("results/RESULTS_SUMMARY.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{'axis':16s} {'leg':24s} {'n':>5s} {'pos':>4s} {'prec':>6s} {'recall':>6s} {'acc':>6s} {'bAcc':>6s} {'F1':>6s} {'AUROC':>6s}")
for r in rows:
    if r["role"] == "main":
        print(f"{r['axis']:16s} {r['leg']:24s} {r['n']:5d} {r['pos']:4d} {r['prec']:6.3f} {r['rec']:6.3f} {r['acc']:6.3f} {(r['bacc'] or 0):6.3f} {r['f1']:6.3f} {(r['au'] or 0):6.3f}")
