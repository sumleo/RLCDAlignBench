#!/usr/bin/env python3
"""Build the Hugging Face release of RLCDAlignBench from the raw experiment package.

    python tools/build_release.py --raw <raw_package_dir> --out <hf_dataset_dir>

The raw package is the internal layout (data/detection_samples, results/jev_responses, ...).
The output is the developer-facing layout documented in the dataset card:

    benchmarks.csv                                   44-row index (copied from data/benchmarks.csv)
    data/benchmarks/<failure_type>/<benchmark>.jsonl canonical detection instances (7,193 rows)
    data/all.jsonl                                   the same 7,193 rows in one file (state/meta as JSON strings)
    data/index.jsonl                                 benchmarks.csv as JSONL (backs the `index` config)
    data/human/<name>.jsonl                          human-labelled sets (StrongREJECT, HarmBench)
    data/variants/<benchmark>/<variant>.jsonl        every input view used in the paper (ablations, oracle, ...)
    jev/responses/<benchmark>/<variant>/<battery>.jsonl   Jev's raw per-question answers
    jev/metrics/<benchmark>/<variant>/<battery>.json      per-strategy metrics for that run
    results/                                         paper-level tables
    provenance/                                      generation outputs, judge calls, official scores, logs
    file_map.csv                                     original path -> release path, sha256 (for the legacy layout)

Every original file is copied byte-for-byte somewhere under the release. The only derived files are
data/benchmarks/*, data/human/*, data/all.jsonl, data/index.jsonl, and the CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Legs that are not among the 44 benchmarks but belong to one of them (or to a human-labelled set).
LEG_ALIAS = {
    "abstentionbench_should_answer": ("abstentionbench", "should_answer"),
    "privacylens_all": ("privacylens", "all_items"),
    "ragtruth_official": ("ragtruth", "official_scorer"),
    "harmbench_val": ("harmbench_human", ""),
    "strongreject_labelbox": ("strongreject_human", ""),
    "strongreject_labelbox_official": ("strongreject_human", "official_rubric"),
}
HUMAN = {"harmbench_human", "strongreject_human"}
TARGET_ORGS = {"Qwen", "microsoft", "google", "meta-llama", "allenai"}

TRACK_DOC = {
    "main": "detection input used for the paper's main results",
    "official_track": "the input the official scorer sees (may contain fields that encode the label)",
    "ablation": "context ablation that adds or removes one field",
    "oracle_upper_bound": "adds a field that reveals the reference (upper bound, not deployable)",
    "nominal": "reported but not used in aggregates",
    "understanding": "gold labels on source data with no target-model output (task understanding)",
    "exclusion": "label 1 iff the official scorer excludes the item",
}


def sha(p: Path, n: int | None = None) -> str:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    return h[:n] if n else h


def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_name(fname: str) -> tuple[str, str | None, str]:
    """leg__Org__Model__variant.jsonl -> (leg, 'Org/Model', variant). Files without a model -> (leg, None, variant)."""
    parts = fname[: -len(".jsonl")].split("__")
    if len(parts) >= 3 and parts[1] in TARGET_ORGS:
        return parts[0], f"{parts[1]}/{parts[2]}", "__".join(parts[3:])
    return parts[0], None, "__".join(parts[1:])


def load_tracks(raw: Path) -> dict[str, str]:
    tracks: dict[str, str] = {}
    for line in (raw / "results/RESULTS_SUMMARY.md").read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.split("|")]
        if len(c) > 5 and c[4].endswith(".jsonl"):
            tracks.setdefault(c[4], c[3])
    return tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", type=Path, help="optional dir with paper-level CSV tables to ship under results/")
    args = ap.parse_args()
    raw, out = args.raw.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    bench = pd.read_csv(REPO / "data/benchmarks.csv")
    ft_of = dict(zip(bench.benchmark, bench.failure_type))
    canon_file = dict(zip(bench.source_file, bench.benchmark))
    tracks = load_tracks(raw)
    shutil.copy2(REPO / "data/benchmarks.csv", out / "benchmarks.csv")

    file_map: list[dict] = []
    variants: list[dict] = []
    samples_by_sha8: dict[str, tuple[str, str]] = {}

    def place(src: Path, dst: Path, derived: bool = False) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not derived:
            shutil.copy2(src, dst)
        file_map.append({"original": str(src.relative_to(raw)), "release": str(dst.relative_to(out)),
                         "sha256": sha(src)})

    # ---- detection samples -------------------------------------------------------------------
    all_rows: list[dict] = []
    for src in sorted((raw / "data/detection_samples").glob("*.jsonl")):
        leg, model, variant = split_name(src.name)
        benchmark, prefix = LEG_ALIAS.get(leg, (leg, ""))
        variant = "__".join(x for x in (prefix, variant) if x) or "source_gold"
        track = tracks.get(src.name, "main")
        if variant.endswith("excluded"):
            track = "exclusion"
        rows = read_jsonl(src)
        dst = out / "data/variants" / benchmark / f"{variant}.jsonl"
        place(src, dst)
        samples_by_sha8[sha(src, 8)] = (benchmark, variant)
        variants.append({"benchmark": benchmark, "variant": variant, "track": track,
                         "is_canonical": src.name in canon_file, "rows": len(rows),
                         "labelled": sum(r["label"] in (0, 1) for r in rows),
                         "state_fields": "|".join(rows[0]["state"].keys()) if rows else "",
                         "path": str(dst.relative_to(out)), "original_file": src.name})

        if src.name in canon_file or benchmark in HUMAN:
            kept = []
            for r in rows:
                status = r["meta"].get("label_status") or ""
                if r["label"] not in (0, 1) or status.startswith("excluded"):
                    continue
                kept.append({
                    "id": r["id"], "benchmark": benchmark,
                    "failure_type": ft_of.get(benchmark, "jailbreak"),
                    "variant": variant, "target_model": model or r["meta"].get("model"),
                    "item": r["item"], "state": r["state"], "label": int(r["label"]),
                    "label_source": r.get("label_source"), "meta": r["meta"],
                })
            if src.name in canon_file:
                write_jsonl(out / "data/benchmarks" / ft_of[benchmark] / f"{benchmark}.jsonl", kept)
                all_rows += kept
            else:
                write_jsonl(out / "data/human" / f"{benchmark}{'__' + variant if variant != 'source_gold' else ''}.jsonl", kept)

    pd.DataFrame(variants).sort_values(["benchmark", "variant"]).to_csv(out / "variants.csv", index=False)
    # one file for all benchmarks: state keys differ per benchmark, so state and meta are JSON strings here
    write_jsonl(out / "data/all.jsonl", [{**{k: v for k, v in r.items() if k not in ("state", "meta")},
                                          "state": json.dumps(r["state"], ensure_ascii=False),
                                          "meta": json.dumps(r["meta"], ensure_ascii=False)} for r in all_rows])
    write_jsonl(out / "data/index.jsonl", json.loads(bench.to_json(orient="records")))
    assert len(all_rows) == int(bench.n.sum()), (len(all_rows), int(bench.n.sum()))

    # ---- Jev responses and metrics -------------------------------------------------------------
    seen: dict[tuple, tuple[str, str]] = {}
    for run in sorted((raw / "results/jev_responses").glob("*/*/")):
        name = run.name                        # <battery>__<format>__<model>__<samples_sha8>__<battery_sha8>
        battery, fmt, _, s8, _ = name.split("__")
        benchmark, variant = samples_by_sha8[s8]
        resp = run / "responses.jsonl"
        key = (benchmark, variant, battery, fmt)
        digest = sha(resp)
        if key in seen and seen[key][0] == digest:     # same run filed under two leg directories
            file_map.append({"original": str(resp.relative_to(raw)), "release": seen[key][1], "sha256": digest})
            continue
        stem = battery if fmt == "object" else f"{battery}__{fmt}"
        if key in seen:
            stem += f"__{s8}"
        dst = out / "jev/responses" / benchmark / variant / f"{stem}.jsonl"
        seen[key] = (digest, str(dst.relative_to(out)))
        place(resp, dst)
        mdir = raw / "results/metrics" / run.parent.name / name
        for ext in ("json", "md"):
            m = mdir / f"metrics.{ext}"
            if m.exists():
                place(m, out / "jev/metrics" / benchmark / variant / f"{stem}.{ext}")

    # ---- results summary and paper tables ------------------------------------------------------
    place(raw / "results/RESULTS_SUMMARY.md", out / "results/RESULTS_SUMMARY.md")
    if args.tables:
        for name in ("main_results.csv", "tab2_main.csv", "baselines.csv", "human_agreement.csv",
                     "context_ablation_v2_summary.csv", "corrected_labels.csv", "cost_totals.csv"):
            p = args.tables / name
            if p.exists():
                shutil.copy2(p, out / "results" / name)

    # ---- provenance ----------------------------------------------------------------------------
    for sub in ("model_outputs", "judge_records", "official_scores"):
        for src in sorted((raw / "data" / sub).rglob("*")):
            if src.is_file():
                place(src, out / "provenance" / sub / src.relative_to(raw / "data" / sub))
    suites = raw / "data/benchmark_suites"
    capability = {"mmlu.jsonl", "gsm8k.jsonl", "ifeval.jsonl"}
    cap_done: dict[str, str] = {}
    for src in sorted(suites.rglob("*")):
        rel = src.relative_to(suites)
        if not src.is_file() or rel.parts[0] == "_ds_latest" or rel.name in ("README.md", "BENCHMARK_TABLE.md"):
            continue
        if src.name in capability:          # byte-identical copy in every axis directory
            digest = sha(src)
            if src.name in cap_done:
                assert cap_done[src.name] == digest, src
                file_map.append({"original": str(src.relative_to(raw)),
                                 "release": f"provenance/benchmark_suites/_capability/{src.name}", "sha256": digest})
                continue
            cap_done[src.name] = digest
            place(src, out / "provenance/benchmark_suites/_capability" / src.name)
            continue
        place(src, out / "provenance/benchmark_suites" / rel)
    for src in sorted((raw / "logs").rglob("*")):
        if src.is_file():
            place(src, out / "provenance/logs" / src.relative_to(raw / "logs"))

    with (out / "file_map.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["original", "release", "sha256"])
        w.writeheader()
        w.writerows(file_map)

    counts = defaultdict(int)
    for v in variants:
        counts[v["track"]] += 1
    print(f"benchmarks: {len(bench)} files, {len(all_rows)} rows")
    print(f"variants: {len(variants)} files  {dict(counts)}")
    print(f"file_map: {len(file_map)} entries")


if __name__ == "__main__":
    main()
