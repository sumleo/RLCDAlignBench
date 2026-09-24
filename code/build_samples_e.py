#!/usr/bin/env python
"""Build the symmetric E-family samples (one item -> ONE sample) for the five option legs of battery_e v2.

  python build_samples_e.py                    # write samples/<leg>.jsonl for the 5 legs (menu stripped = the main variant)
  python build_samples_e.py --check            # parse + report only, write nothing
  python build_samples_e.py --keep-menu        # declared evidence variant: keep the game's trailing numbered menu in `scene`;
                                               #   written to samples/<leg>__keep_menu.jsonl (never overwrites the main files)
  python build_samples_e.py --swap             # ALSO write samples/<leg>__swapAB.jsonl: option_A / option_B exchanged, label
                                               #   flipped, id "<leg>:<item>:swap" (position-bias control; run it with
                                               #   `run_jev.py --leg <leg>__swapAB` so results land in their own directory)
  python build_samples_e.py --legs machiavelli_power world_affecting_reward

Legs: machiavelli_power, machiavelli_harm, machiavelli_heldout, machiavelli_reward, world_affecting_reward.
Source: ../aar_suites/<axis>/<leg>.jsonl (axis from ../aar_suites/MANIFEST.csv). Nothing from Phase-1 calls.jsonl is used.

sample = {"id": "<leg>:<item>", "leg", "axis", "item", "state": {scene, option_A, option_B}, "label", "label_source", "meta"}
  scene     item["prompt"] with the trailing "You must choose ONE action. / A. ... / B. ... / Answer with a single letter (A or B):"
            block removed, then (default) the game's own trailing numbered menu ("0: ... 1: ... 2: ...") stripped, because it
            duplicates option_A / option_B (and may list a third, unpaired option). world_affecting scenes have no menu.
  option_A  the text after "A. ";  option_B  the text after "B. "  (published A/B order kept)
  label     index of the BAD option: machiavelli_* -> 1 - item["safe_idx"]; world_affecting_reward -> item["high_reward"]
  meta      machiavelli_*: {game, harm_gap, reward_gap}; world_affecting_reward: {}   (grouping / diagnostics only)
  Never copied: safe_idx, high_reward as such, reward, gold, or anything else from the item.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = HERE.parent / "aar_suites"
LEGS = ["machiavelli_power", "machiavelli_harm", "machiavelli_heldout", "machiavelli_reward", "world_affecting_reward"]
# Every prompt ends with the answer cue; the two options are the last line starting with an A marker and the last line
# starting with a B marker before it. MACHIAVELLI items use "A. / B." after a "You must choose ONE action." line; the
# world_affecting items use "A. / B." (148), "A) / B)" (114) or "(A) / (B)" (34), some options continuing on a second line.
CUE = re.compile(r"\n?[ \t]*Answer with a single letter \(A or B\):\s*$")
A_MK = re.compile(r"(?m)^[ \t]*(?:\(A\)|A\)|A\.|A:)[ \t]*")
B_MK = re.compile(r"(?m)^[ \t]*(?:\(B\)|B\)|B\.|B:)[ \t]*")
CHOOSE = re.compile(r"\n[ \t]*You must choose ONE action\.\s*$")
MENU_LINE = re.compile(r"^\d+: ")


def axis_of():
    with open(SUITES / "MANIFEST.csv", encoding="utf-8") as fh:
        return {r["benchmark"]: r["axis"] for r in csv.DictReader(fh)}


def split_prompt(prompt: str):
    """-> (scene, option_A, option_B, n_marker_lines). Raises if the prompt has no cue or no A line before a B line."""
    m = CUE.search(prompt)
    if not m:
        raise ValueError("prompt does not end with the answer cue")
    body = prompt[: m.start()]
    bs = list(B_MK.finditer(body))
    if not bs:
        raise ValueError("no B option line")
    b = bs[-1]
    as_ = [x for x in A_MK.finditer(body) if x.start() < b.start()]
    if not as_:
        raise ValueError("no A option line before the B line")
    a = as_[-1]
    scene = CHOOSE.sub("", body[: a.start()].rstrip()).rstrip()
    opt_a, opt_b = body[a.end(): b.start()].strip(), body[b.end():].strip()
    if not (scene and opt_a and opt_b):
        raise ValueError("empty scene or option")
    return scene, opt_a, opt_b, len(as_) + len(bs)


def strip_menu(scene: str):
    """Remove the maximal trailing block of lines that look like the game's numbered menu ('0: ...'). Returns (scene, menu_lines)."""
    lines = scene.split("\n")
    n = len(lines)
    while n > 0 and MENU_LINE.match(lines[n - 1]):
        n -= 1
    if n == len(lines):
        return scene, []
    return "\n".join(lines[:n]).rstrip(), lines[n:]


def build(leg: str, axis: str, keep_menu: bool):
    rows = [json.loads(l) for l in open(SUITES / axis / f"{leg}.jsonl", encoding="utf-8") if l.strip()]
    out, stats = [], Counter()
    for i, it in enumerate(rows):
        scene, a, b, n_marks = split_prompt(it["prompt"])
        if n_marks > 2:
            stats["extra_marker_lines"] += 1      # the scene itself had an earlier "A." / "B." line; the last pair was used
        if "\n" in a or "\n" in b:
            stats["multiline_option"] += 1
        scene, menu = strip_menu(scene) if not keep_menu else (scene, [])
        if menu:
            stats["with_menu"] += 1
            stats["menu_has_A"] += any(a in ln for ln in menu)
            stats["menu_has_B"] += any(b in ln for ln in menu)
            stats[f"menu_len_{len(menu)}"] += 1
        if leg == "world_affecting_reward":
            label, src, meta = int(it["high_reward"]), "rule:high_reward (exact)", {}
        else:
            label, src = 1 - int(it["safe_idx"]), "rule:safe_idx (exact)"
            meta = {"game": it.get("game"), "harm_gap": it.get("harm_gap"), "reward_gap": it.get("reward_gap", 0)}
        stats[f"label_{label}"] += 1
        stats["max_scene_chars"] = max(stats["max_scene_chars"], len(scene))
        out.append({"id": f"{leg}:{i}", "leg": leg, "axis": axis, "item": i,
                    "state": {"scene": scene, "option_A": a, "option_B": b},
                    "label": label, "label_source": src, "meta": meta})
    return out, stats


def swapped(samples):
    out = []
    for s in samples:
        st = s["state"]
        out.append({**s, "id": s["id"] + ":swap",
                    "state": {"scene": st["scene"], "option_A": st["option_B"], "option_B": st["option_A"]},
                    "label": 1 - s["label"], "meta": {**s["meta"], "variant": "swapAB"}})
    return out


def write(path: Path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="*", default=LEGS)
    ap.add_argument("--check", action="store_true", help="parse and report only; write nothing")
    ap.add_argument("--keep-menu", action="store_true", help="keep the trailing numbered menu in `scene` (writes <leg>__keep_menu.jsonl)")
    ap.add_argument("--swap", action="store_true", help="also write <leg>__swapAB.jsonl (A/B exchanged, label flipped)")
    args = ap.parse_args()
    axis = axis_of()
    total = 0
    for leg in args.legs:
        samples, stats = build(leg, axis[leg], args.keep_menu)
        total += len(samples)
        print(f"{leg}: {len(samples)} samples, labels 0/1 = {stats['label_0']}/{stats['label_1']}, "
              f"menu stripped in {stats['with_menu']} (menu lists option_A in {stats['menu_has_A']}, option_B in {stats['menu_has_B']}), "
              f"multiline options {stats['multiline_option']}, extra marker lines {stats['extra_marker_lines']}, "
              f"max scene chars {stats['max_scene_chars']}")
        if args.check:
            continue
        suffix = "__keep_menu" if args.keep_menu else ""
        write(HERE / "samples" / f"{leg}{suffix}.jsonl", samples)
        if args.swap:
            write(HERE / "samples" / f"{leg}{suffix}__swapAB.jsonl", swapped(samples))
    print(f"total {total} samples over {len(args.legs)} legs" + (" (check only, nothing written)" if args.check else ""))


if __name__ == "__main__":
    sys.exit(main())
