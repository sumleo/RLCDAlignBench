#!/usr/bin/env python
"""Run a question battery against Jev for one leg: ONE request per sample carrying every question, cache raw answers by
request identity, then score every strategy on the same answers with a fixed, fully-enumerated denominator.

  python run_jev.py --leg machiavelli_power --battery battery_e --limit 20        # pilot
  python run_jev.py --leg machiavelli_power --battery battery_e                   # full leg
  python run_jev.py --leg summedits --battery battery_e --state-format text       # state variant (separate pass)
  python run_jev.py --leg summedits --battery battery_e --dry-run                 # print one request, send nothing
  python run_jev.py --leg strongreject --battery battery_refusal --rescore        # metrics only, from cache

Outputs: results/<leg>/<run_name>/{responses.jsonl, metrics.json, metrics.md}
  run_name = <battery>_<version>__<state_format>__<model>__<samples_sha8>__<battery_sha8>

Design rules implemented (REVIEW.md §4 T2–T6, DECISIONS.md):
  - cache identity = sha256(request body) + battery file hash + samples file hash + resolved model; a cached answer is
    reused only when the exact same request was sent before (T3). --rescore never sends.
  - every sample ends in exactly one status: answered | no_label | api_failed | not_applicable(strategy) | abstained(strategy)
    and the metrics report all counts (T2). Rows without a label never enter a metric. API failures are counted as
    misses (pred=0) in the fixed-denominator F1 (declared policy) and excluded from the judged-only metrics.
  - strategies may return None (abstain) or a float risk; a battery may expose STRATEGY_REQUIRES {name: [qids]} — a
    strategy whose required answers are missing is marked not_applicable, not abstained (T4).
  - metrics per strategy: n, coverage, precision, recall, F1, accuracy, balanced accuracy, AUROC (rank-based), F1 on the
    fixed denominator, and a grouped bootstrap 95% CI for F1 (groups = sample.meta.group_id or the item index; U12).
  - CONFIG_ROLE (battery, optional) labels the run: main | official_track | oracle_upper_bound | understanding (three tracks).
Speed: default 64 concurrent requests, 1100 requests/min (official limit 1200 rpm), exponential backoff on 429/5xx.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINT = os.environ.get("TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone")
PRICE_PER_MTOK = 0.042


# --- env / io ---------------------------------------------------------------------------------------
def load_env():
    p = HERE.parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_t = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)


def render_text(state) -> str:
    """Canonical text rendering of an object state (declared state variant): '### field' headers, nested values as JSON."""
    if isinstance(state, str):
        return state
    parts = []
    for k, v in state.items():
        body = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)
        parts.append(f"### {k}\n{body}")
    return "\n\n".join(parts)


def post(body: dict, key: str, timeout: float = 90.0, retries: int = 6):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(ENDPOINT, data=data, headers=hdr)
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8")), round(time.monotonic() - t0, 3), None
        except urllib.error.HTTPError as e:
            txt = e.read().decode("utf-8", errors="replace")[:500]
            last = f"HTTP {e.code}: {txt}"
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < retries:
                ra = e.headers.get("retry-after")
                time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(2 ** attempt, 30))
                continue
            return None, round(time.monotonic() - t0, 3), last
        except Exception as e:   # noqa: BLE001 — network/timeouts
            last = repr(e)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
                continue
            return None, round(time.monotonic() - t0, 3), last
    return None, 0.0, last


# --- metrics --------------------------------------------------------------------------------------------
def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    ranked = sorted(zip(scores, labels), key=lambda x: x[0])
    ranks, i = {}, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    sum_pos = sum(ranks[k] for k, (s, l) in enumerate(ranked) if l == 1)
    n1, n0 = len(pos), len(neg)
    return round((sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0), 4)


def binary_metrics(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / n if n else 0.0
    bacc = (rec + spec) / 2 if (tp + fn) and (tn + fp) else None
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, precision=round(prec, 4), recall=round(rec, 4), specificity=round(spec, 4),
                f1=round(f1, 4), accuracy=round(acc, 4), balanced_accuracy=(round(bacc, 4) if bacc is not None else None))


def grouped_bootstrap_f1(rows, threshold, n_boot=500, seed=0):
    """rows: list of (group_id, risk, label) with risk not None. Resample groups with replacement."""
    groups = {}
    for g, r, l in rows:
        groups.setdefault(g, []).append((r, l))
    keys = list(groups)
    if len(keys) < 5:
        return None
    rng = random.Random(seed)
    f1s = []
    for _ in range(n_boot):
        sample = [groups[rng.choice(keys)] for _ in keys]
        preds = [1 if r >= threshold else 0 for grp in sample for r, _ in grp]
        labels = [l for grp in sample for _, l in grp]
        f1s.append(binary_metrics(preds, labels)["f1"])
    f1s.sort()
    return [round(f1s[int(0.025 * n_boot)], 4), round(f1s[int(0.975 * n_boot)], 4)]


def config_of(battery, s):
    """Configuration key of a sample: battery.config_id() (bc/d/e/f) or battery.config_key() (refusal/a_judge)."""
    for fn in ("config_id", "config_key"):
        if hasattr(battery, fn):
            return getattr(battery, fn)(s)
    return None


def role_of(battery, s):
    """Track role of a sample's configuration: config_role() > CONFIG_ROLE[cfg] > CONFIG_ROLE[(leg, variant)] > CONFIG_ROLE[leg] > meta.track > main."""
    if hasattr(battery, "config_role"):
        return battery.config_role(s)
    role_map = getattr(battery, "CONFIG_ROLE", {}) or {}
    meta = s.get("meta") or {}
    cfg = config_of(battery, s)
    for key in (cfg, (s["leg"], meta.get("sample_type") or meta.get("variant") or meta.get("arm") or ""), s["leg"]):
        if key in role_map:
            return role_map[key]
    return meta.get("track") or "main"

THR_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def cv_threshold_f1(rows, fixed_rows=None):
    """Two-fold cross-fitted threshold (Jev docs: evaluate thresholds on your own data, never on the test rows themselves).
    rows: (group, risk, label) with numeric risk. Groups are split deterministically by sha1 parity; the threshold that maximises
    F1 on one fold (grid 0.05..0.95, ties -> closest to 0.5) is applied to the other fold; predictions are pooled.
    Returns (f1_cv, f1_cv_fixed, (thr_fold0, thr_fold1)) or (None, None, None) when a fold lacks both classes."""
    if not rows:
        return None, None, None
    fold = {}
    for g, _, _ in rows:
        if g not in fold:
            fold[g] = int(hashlib.sha1(str(g).encode("utf-8")).hexdigest(), 16) % 2
    parts = [[r for r in rows if fold[r[0]] == k] for k in (0, 1)]
    if any(len({l for _, _, l in part}) < 2 for part in parts):
        return None, None, None
    thr, pooled = [], []
    for k in (0, 1):
        train, test = parts[1 - k], parts[k]
        best = max(THR_GRID, key=lambda th: (binary_metrics([1 if r >= th else 0 for _, r, _ in train], [l for _, _, l in train])["f1"], -abs(th - 0.5)))
        thr.append(best)
        pooled += [(g, 1 if r >= best else 0, l) for g, r, l in test]
    f1 = binary_metrics([p for _, p, _ in pooled], [l for _, _, l in pooled])["f1"]
    f1_fixed = None
    if fixed_rows is not None:
        pred = {(g, i): p for i, (g, p, _) in enumerate(pooled)}
        thr_of = {g: thr[fold[g]] for g in fold}
        full = [(1 if (isinstance(r, (int, float)) and r >= thr_of.get(g, 0.5)) else 0, l) for g, r, l in fixed_rows]
        f1_fixed = binary_metrics([p for p, _ in full], [l for _, l in full])["f1"]
    return f1, f1_fixed, tuple(thr)

def evaluate(samples, responses, battery, threshold=0.5):
    """responses: {id: record}. Returns (per-strategy metrics, status counts)."""
    requires = getattr(battery, "STRATEGY_REQUIRES", {}) or {}
    labeled = [s for s in samples if s.get("label") in (0, 1)]
    status = {"total": len(samples), "labeled": len(labeled), "no_label": len(samples) - len(labeled), "answered": 0, "api_failed": 0}
    strat_rows = {}       # name -> list of (group, risk|None|"n/a", label)
    for s in labeled:
        r = responses.get(s["id"])
        group = (s.get("meta") or {}).get("group_id", s.get("item", s["id"]))
        if not r or not r.get("answers"):
            status["api_failed"] += 1
            names = requires if not (requires and isinstance(next(iter(requires.values())), dict)) else requires.get(config_of(battery, s), {})
            target = (s.get("meta") or {}).get("label_target")
            for name in names or []:
                if target and name.startswith("target:") and not name.startswith(f"target:{target}/"):
                    continue
                strat_rows.setdefault(name, []).append((group, "api_failed", s["label"]))
            continue
        status["answered"] += 1
        ans = r["answers"]
        out = battery.strategies(s, ans)
        req_map = requires
        if requires and isinstance(next(iter(requires.values())), dict):      # nested by config id
            req_map = requires.get(config_of(battery, s), {})
        target = (s.get("meta") or {}).get("label_target")          # battery_refusal: target:<name>/... strategies only score their own label
        for name, risk in out.items():
            if target and name.startswith("target:") and not name.startswith(f"target:{target}/"):
                continue                                             # other label target: excluded from this file's metrics entirely
            req = req_map.get(name)
            if req and any(q not in ans for q in req):
                strat_rows.setdefault(name, []).append((group, "n/a", s["label"]))
            else:
                strat_rows.setdefault(name, []).append((group, risk, s["label"]))
    metrics = {}
    for name, rows in strat_rows.items():
        judged = [(g, p, l) for g, p, l in rows if isinstance(p, (int, float))]
        n_abstain = sum(1 for _, p, _ in rows if p is None)
        n_na = sum(1 for _, p, _ in rows if p == "n/a")
        n_fail = sum(1 for _, p, _ in rows if p == "api_failed")
        preds = [1 if p >= threshold else 0 for _, p, _ in judged]
        labels = [l for _, _, l in judged]
        m = binary_metrics(preds, labels)
        m["coverage"] = round(len(judged) / len(labeled), 4) if labeled else 0.0
        m["abstained"] = n_abstain
        m["not_applicable"] = n_na
        m["api_failed"] = n_fail
        m["abstained_pos"] = sum(1 for _, p, l in rows if p is None and l == 1)
        m["abstained_neg"] = sum(1 for _, p, l in rows if p is None and l == 0)
        m["auroc"] = auroc([p for _, p, _ in judged], labels)
        # fixed denominator over ALL labeled rows: abstain / n/a / api_failed count as pred=0 (declared policy)
        full_preds = [1 if (isinstance(p, (int, float)) and p >= threshold) else 0 for _, p, _ in rows]
        full_labels = [l for _, _, l in rows]
        fm = binary_metrics(full_preds, full_labels)
        m["f1_fixed_denominator"] = fm["f1"]
        m["recall_fixed_denominator"] = fm["recall"]
        m["f1_ci95_grouped"] = grouped_bootstrap_f1(judged, threshold)
        m["f1_cv"], m["f1_cv_fixed"], m["thr_cv"] = cv_threshold_f1(judged, rows)
        metrics[name] = m
    return metrics, status


def metrics_md(leg, run_name, role, m, status, usage):
    lines = [f"# {leg} — {run_name}", "", f"track/role: **{role}**", "",
             f"samples: total {status['total']}, labeled {status['labeled']}, no_label {status['no_label']}, answered {status['answered']}, "
             f"api_failed {status['api_failed']}; input tokens {usage['input_tokens']:,}; est. cost ${usage['input_tokens'] / 1e6 * PRICE_PER_MTOK:.3f}; "
             f"median latency {usage['median_latency']}s; wall {usage['wall_s']}s", "",
             "| strategy | n | cov | prec | recall | F1 | acc | bAcc | AUROC | F1 fixed | F1 cv | thr cv | F1 CI95 | abst (+/−) | n/a |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|"]
    for name, x in sorted(m.items(), key=lambda kv: -(kv[1]["f1"])):
        ci = x["f1_ci95_grouped"]
        lines.append(f"| {name} | {x['n']} | {x['coverage']:.2f} | {x['precision']:.3f} | {x['recall']:.3f} | **{x['f1']:.3f}** | {x['accuracy']:.3f} | "
                     f"{x['balanced_accuracy'] if x['balanced_accuracy'] is not None else '-'} | {x['auroc'] if x['auroc'] is not None else '-'} | "
                     f"{x['f1_fixed_denominator']:.3f} | {x['f1_cv_fixed'] if x.get('f1_cv_fixed') is not None else '-'} | "
                     f"{'/'.join(str(v) for v in x['thr_cv']) if x.get('thr_cv') else '-'} | {ci if ci else '-'} | {x['abstained_pos']}/{x['abstained_neg']} | {x['not_applicable']} |")
    lines += ["", "cov = coverage of labeled samples; bAcc = balanced accuracy; F1 fixed = abstentions / n/a / API failures counted as pred=0; "
              "F1 cv = fixed-denominator F1 with a threshold chosen on the other half (2-fold by item, grid 0.05-0.95; thr cv = the two thresholds); "
              "CI95 = grouped bootstrap by item; abst = abstained positives/negatives."]
    return "\n".join(lines) + "\n"


# --- main ---------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True)
    ap.add_argument("--battery", required=True, help="python module in this dir, e.g. battery_e")
    ap.add_argument("--samples", default=None, help="samples jsonl (default samples/<leg>.jsonl)")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--state-format", default="object", choices=["object", "text"])
    ap.add_argument("--limit", type=int, default=None, help="pilot: first N samples")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--rpm", type=int, default=1100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="only recompute metrics from cached responses; never sends")
    args = ap.parse_args()

    load_env()
    sys.path.insert(0, str(HERE))
    battery = importlib.import_module(args.battery)
    battery_file = Path(battery.__file__)
    samples_path = Path(args.samples) if args.samples else HERE / "samples" / f"{args.leg}.jsonl"
    samples = [json.loads(l) for l in open(samples_path, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[: args.limit]
    b_sha, s_sha = sha256_file(battery_file)[:8], sha256_file(samples_path)[:8]
    run_name = f"{battery.NAME}_{battery.VERSION}__{args.state_format}__{args.model}__{s_sha}__{b_sha}"
    out_dir = HERE / "results" / args.leg / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / "responses.jsonl"
    role_map = getattr(battery, "CONFIG_ROLE", {}) or {}
    first_meta = (samples[0].get("meta") or {}) if samples else {}
    role = role_of(battery, samples[0]) if samples else "main"

    def build(s):
        state = s["state"] if args.state_format == "object" else render_text(s["state"])
        return {"state": state, "model": args.model, "questions": battery.questions(s)}

    if args.dry_run:
        body = build(samples[0])
        print(json.dumps(body, ensure_ascii=False, indent=1)[:6000])
        print(f"\n{len(body['questions'])} questions; {len(samples)} samples; run_name={run_name}; role={role}")
        return

    # cache: reuse only identical requests (request sha256 recorded per line)
    cached = {}
    if resp_path.exists():
        for l in open(resp_path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("answers") and r.get("request_sha256"):
                    cached[r["request_sha256"]] = r
    responses, todo, bodies = {}, [], {}
    for s in samples:
        body = build(s)
        h = sha256_json(body)
        bodies[s["id"]] = (body, h)
        if h in cached:
            responses[s["id"]] = cached[h]
        else:
            todo.append(s)
    usage = {"input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in responses.values()), "latencies": [], "wall_s": 0}
    if todo and not args.rescore:
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            sys.exit("TYPESAFE_API_KEY missing (put it in ../.env)")
        limiter = RateLimiter(args.rpm)
        lock = threading.Lock()
        fh = open(resp_path, "a", encoding="utf-8")
        t_wall = time.monotonic()
        nq = len(bodies[todo[0]["id"]][0]["questions"])
        print(f"{args.leg} [{role}]: {len(todo)} requests to send ({len(responses)} cached), ~{nq} questions each, "
              f"concurrency={args.concurrency} rpm={args.rpm} run={run_name}", flush=True)
        done_n = [0]

        def work(s):
            body, h = bodies[s["id"]]
            limiter.wait()
            resp, lat, err = post(body, key)
            rec = {"id": s["id"], "leg": s["leg"], "label": s.get("label"), "state_format": args.state_format, "model": args.model,
                   "battery": f"{battery.NAME}_{battery.VERSION}", "battery_sha8": b_sha, "samples_sha8": s_sha, "request_sha256": h,
                   "latency_s": lat, "answers": resp.get("answers") if resp else None, "usage": resp.get("usage") if resp else None,
                   "model_reported": resp.get("model") if resp else None, "error": err, "t": time.time()}
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if rec["answers"]:
                    responses[s["id"]] = rec
                    usage["input_tokens"] += (rec["usage"] or {}).get("input_tokens", 0)
                    usage["latencies"].append(lat)
                done_n[0] += 1
                if done_n[0] % 200 == 0 or done_n[0] == len(todo):
                    fails = sum(1 for x in todo[: done_n[0]] if x["id"] not in responses)
                    print(f"  {done_n[0]}/{len(todo)} sent, failures so far ~{fails}, {time.monotonic() - t_wall:.0f}s", flush=True)
            return rec

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(work, todo))
        fh.close()
        usage["wall_s"] = round(time.monotonic() - t_wall, 1)
    elif todo and args.rescore:
        print(f"rescore: {len(todo)} samples have no cached answer for the current request and are counted as api_failed", flush=True)
    lat = sorted(usage["latencies"])
    usage["median_latency"] = lat[len(lat) // 2] if lat else None
    usage.pop("latencies", None)
    m, status = evaluate(samples, responses, battery, args.threshold)
    model_reported = next((r.get("model_reported") for r in responses.values() if r.get("model_reported")), None)
    (out_dir / "metrics.json").write_text(json.dumps({"leg": args.leg, "run": run_name, "role": role, "threshold": args.threshold,
                                                       "model_reported": model_reported, "battery_sha8": b_sha, "samples_sha8": s_sha,
                                                       "status": status, "usage": usage, "strategies": m}, indent=1, ensure_ascii=False), encoding="utf-8")
    md = metrics_md(args.leg, run_name, role, m, status, usage)
    (out_dir / "metrics.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
