#!/usr/bin/env python
"""Phase 1 / Phase 2 driver around the AAR harness (aar_repo/aar/eval_pod/run_eval.py).

Phase 1  (GPU, no API keys)   python run_generate.py --axis refusal --model microsoft/Phi-4-mini-instruct
    Runs every safety + held-out benchmark of the axis with the official AAR loaders and the
    official ("golden") decoding, but with all LLM judges STUBBED. Every call the benchmark makes
    to the target model is recorded to   outputs/<axis>/<model_tag>/calls.jsonl
    (one JSON line per prompt: benchmark, call kind, input, output / logits).
    Rule- and logprob-scored benchmarks get their real scores; judge-scored ones get placeholder
    scores (ignore them) — their judging happens in Phase 2.

Phase 2  (any machine, API keys set)   python run_generate.py --axis refusal --model <same> --replay
    Re-runs the same suite with a REPLAY model that answers from calls.jsonl (no GPU needed) and
    the REAL judges (JUDGE_BACKEND / OAI_API / ANTHROPIC_API_KEY as in aar_repo/REPRODUCE.md).
    Judge inputs and verdicts are recorded to judge_calls.jsonl next to calls.jsonl.

Capability benchmarks (mmlu / gsm8k / ifeval) are skipped by default (not needed for detection);
pass --with-capability to include them.
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_MODELS = {   # paper Table 1
    "sycophancy": "Qwen/Qwen3.5-2B",
    "refusal": "microsoft/Phi-4-mini-instruct",
    "prompt_injection": "Qwen/Qwen3.5-2B",
    "power_seeking": "meta-llama/Llama-3.2-3B-Instruct",
    "honesty": "google/gemma-2-2b-it",
    "faithfulness": "meta-llama/Llama-3.2-3B-Instruct",
    "bias": "allenai/Olmo-3-7B-Instruct",
    "privacy": "microsoft/Phi-4-mini-instruct",
    "reward_hacking": "Qwen/Qwen3.5-2B",
    "concealing_uncertainty": "allenai/Olmo-3-7B-Instruct",
}


def _tag(model_ref: str) -> str:
    return model_ref.replace("/", "__").replace(":", "_")


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except TypeError:
        return json.loads(json.dumps(x, default=str))


class Recorder:
    """Append-only JSONL writer for model calls; `bench` is set by the scoring loop."""

    def __init__(self, path: Path):
        self.path = path
        self.bench = None
        self.call_no = 0
        self.depth = 0
        self.n = 0
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, rec: dict) -> None:
        rec = {"bench": self.bench, "call": self.call_no, "t": round(time.time(), 3), **rec}
        self._fh.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
        self.n += 1

    def flush(self) -> None:
        self._fh.flush()


def wrap_model(model, rec: Recorder):
    """Wrap every Model-protocol method on the instance so each prompt/output pair is recorded."""
    def _wrap(name, kind):
        orig = getattr(model, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def w(*a, **k):
            rec.depth += 1
            try:
                out = orig(*a, **k)
            finally:
                rec.depth -= 1
            if rec.depth > 0:          # nested call (e.g. StubModel.generate_batch -> generate): record once, at the outer level
                return out
            rec.call_no += 1
            try:
                if kind == "generate":
                    rec.write({"kind": kind, "i": 0, "input": a[0], "output": out})
                elif kind == "generate_batch":
                    for i, (p, o) in enumerate(zip(a[0], out)):
                        rec.write({"kind": kind, "i": i, "input": p, "output": o})
                elif kind == "candidate_logits":
                    rec.write({"kind": kind, "i": 0, "input": a[0], "candidates": a[1],
                               "use_chat_template": k.get("use_chat_template", a[2] if len(a) > 2 else True), "logits": out})
                elif kind == "candidate_logits_batch":
                    uct = k.get("use_chat_template", a[2] if len(a) > 2 else True)
                    for i, (p, o) in enumerate(zip(a[0], out)):
                        rec.write({"kind": kind, "i": i, "input": p, "candidates": a[1], "use_chat_template": uct, "logits": o})
                elif kind == "completion_logprob_batch":
                    uct = k.get("use_chat_template", a[2] if len(a) > 2 else False)
                    for i, (p, c, o) in enumerate(zip(a[0], a[1], out)):
                        rec.write({"kind": kind, "i": i, "input": p, "completion": c, "use_chat_template": uct, "logprob": o})
            finally:
                rec.flush()
            return out
        setattr(model, name, w)

    for name in ("generate", "generate_batch", "candidate_logits", "candidate_logits_batch", "completion_logprob_batch"):
        _wrap(name, name)
    return model


class ReplayModel:
    """Answers model calls from a Phase-1 calls.jsonl (FIFO per (bench, kind, input))."""

    def __init__(self, calls_path: Path):
        self.q: dict[tuple, collections.deque] = collections.defaultdict(collections.deque)
        self.bench = None
        with open(calls_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["bench"], r["kind"], json.dumps(r["input"], sort_keys=True, ensure_ascii=False),
                       json.dumps(r.get("candidates"), sort_keys=True), json.dumps(r.get("completion")))
                self.q[key].append(r)
        # attributes _apply_golden prints after apply_decoding(); decoding is meaningless on replay
        self.max_new_tokens = None
        self.batch_size = 1
        self._auto_ceiling = None
        self._no_repeat_ngram = 0
        self._temperature = 1.0
        self._top_p = 1.0
        self._seed = 1234

    def apply_decoding(self, **_):   # golden decoding is a no-op on replay
        pass

    def _pop(self, kind, inp, candidates=None, completion=None):
        key = (self.bench, kind, json.dumps(inp, sort_keys=True, ensure_ascii=False),
               json.dumps(candidates, sort_keys=True), json.dumps(completion))
        dq = self.q.get(key)
        if not dq:
            # tolerate a different call shape (generate vs generate_batch) for the same prompt
            alt = "generate" if kind == "generate_batch" else "generate_batch" if kind == "generate" else None
            if alt:
                dq = self.q.get((self.bench, alt) + key[2:])
        if not dq:
            raise KeyError(f"replay: no recorded output for bench={self.bench} kind={kind} input={str(inp)[:120]!r}")
        return dq.popleft()

    def generate(self, prompt, **_):
        return self._pop("generate", prompt)["output"]

    def generate_batch(self, prompts, **_):
        return [self._pop("generate_batch", p)["output"] for p in prompts]

    def candidate_logits(self, prompt, candidates, use_chat_template=True):
        return self._pop("candidate_logits", prompt, candidates)["logits"]

    def candidate_logits_batch(self, prompts, candidates, use_chat_template=True):
        return [self._pop("candidate_logits_batch", p, candidates)["logits"] for p in prompts]

    def completion_logprob_batch(self, prompts, completions, use_chat_template=False):
        return [self._pop("completion_logprob_batch", p, None, c)["logprob"] for p, c in zip(prompts, completions)]


def install_judge_patches(stub: bool, jrec: Recorder | None):
    """Must run BEFORE registry.discover(): benchmark modules bind these names at import time.
    stub=True  -> every judge returns None / "" (no API, no GPU judge) — Phase 1.
    stub=False -> real judges, wrapped to record (judge input, verdict) — Phase 2."""
    import aar.eval_pod.judges as J
    import aar.benchmarks._judge_http as H
    import aar.eval_pod.refusal_judges as R
    import aar.eval_pod.run_eval as E

    single = ("make_local_judge", "make_openai_judge", "make_anthropic_judge")
    batch_bool = ("make_local_judge_batch",)
    batch_text = ("make_local_judge_text_batch", "make_anthropic_judge_text_batch")

    def rec(kind, inp, out):
        if jrec is not None:
            jrec.write({"kind": kind, "input": inp, "output": out})
            jrec.flush()

    if stub:
        for env in ("REFUSAL_STUB", "MASK_STUB", "DECEPTIONBENCH_STUB", "BEHONEST_STUB"):   # per-leg keyword stubs
            os.environ[env] = "1"
        os.environ["RAGTRUTH_ALLOW_PROMPT_JUDGE"] = "1"   # ragtruth's finetuned detector is not shipped; judge is stubbed here anyway
        for n in single:
            setattr(J, n, lambda *a, **k: (lambda p, *x, **y: None))
        for n in batch_bool:
            setattr(J, n, lambda *a, **k: (lambda ps, *x, **y: [None] * len(ps)))
        for n in batch_text:
            setattr(J, n, lambda *a, **k: (lambda ps, *x, **y: [""] * len(ps)))
        H.openai_chat = lambda *a, **k: ""
        H.anthropic_chat = lambda *a, **k: ""
        H._anthropic_key = lambda *a, **k: "stub"
        E._resolve_judge_fn = lambda *a, **k: (lambda p, *x, **y: None)
        return

    def _wrap_factory(mod, name, batch):
        orig = getattr(mod, name)

        def factory(*a, **k):
            fn = orig(*a, **k)
            if fn is None:
                return None
            if batch:
                def w(ps, *x, **y):
                    outs = fn(ps, *x, **y)
                    for p, o in zip(ps, outs):
                        rec(name, p, o)
                    return outs
            else:
                def w(p, *x, **y):
                    o = fn(p, *x, **y)
                    rec(name, p, o)
                    return o
            return w
        setattr(mod, name, factory)

    for n in single + batch_bool:
        _wrap_factory(J, n, batch=n in batch_bool)
    for n in batch_text:
        _wrap_factory(J, n, batch=True)
    for n in ("openai_chat", "anthropic_chat"):
        orig = getattr(H, n)

        def chat(messages, *a, _orig=orig, _n=n, **k):
            o = _orig(messages, *a, **k)
            rec(_n, messages, o)
            return o
        setattr(H, n, chat)
    for n in ("get_strongreject_judge", "get_harmbench_judge", "get_llamaguard_judge", "get_refusal_judge"):
        if hasattr(R, n):
            _wrap_factory(R, n, batch=False)   # these judges take (forbidden, response); recorded as tuple via *args
    # JUDGE_BACKEND=anthropic: legs that hard-code the LOCAL Qwen2.5-7B text judge (instrumental_eval) get the Anthropic
    # text-batch judge instead (same raw-CoT contract; recorded like every other judge). Documented deviation from the
    # harness's local judge: a 7B judge does not fit next to the target model on this machine.
    if os.getenv("JUDGE_BACKEND", "").lower() == "anthropic":
        try:
            import aar.benchmarks.instrumental_eval.benchmark as IE
            IE.make_local_judge_text_batch = J.make_anthropic_judge_text_batch
        except Exception as e:   # noqa: BLE001
            print(f"[judge-patch] instrumental_eval anthropic redirect failed: {e!r}", flush=True)
    if os.getenv("JUDGE_BACKEND", "").lower() == "anthropic":
        # abstentionbench hard-codes the LOCAL Qwen2.5-7B YES/NO batch judge; redirect to the (recorded) Anthropic single judge,
        # mapped over the batch with a small thread pool. Documented deviation: judge model differs from the harness default.
        from concurrent.futures import ThreadPoolExecutor
        def _anthropic_bool_batch(model=None):
            j = J.make_anthropic_judge()
            def judge_batch(prompts):
                if not prompts:
                    return []
                with ThreadPoolExecutor(max_workers=int(os.getenv("JUDGE_CONCURRENCY", "8"))) as ex:
                    return list(ex.map(j, prompts))
            return judge_batch
        J.make_local_judge_batch = _anthropic_bool_batch
    if os.getenv("JUDGE_BACKEND", "").lower() == "openai":
        # JUDGE_BACKEND=openai: the bias legs (bias_refletter/refbio/race_content/scene_heldout) hard-code the Anthropic
        # text-batch judge (claude-haiku-4-5) and abstentionbench the LOCAL Qwen2.5-7B YES/NO batch judge. When the
        # Anthropic account has no credit (2026-09-23: HTTP 400 "credit balance is too low" on every call) both are
        # redirected to gpt-4o over the harness's own openai_chat / make_openai_judge callers, temperature 0. Recorded
        # under their own kinds ("make_openai_judge_text_batch", "make_openai_judge") so provenance stays visible.
        # Documented deviation: judge model differs from the harness default.
        import sys
        from concurrent.futures import ThreadPoolExecutor
        _oai_model = os.getenv("JUDGE_MODEL_OPENAI", "gpt-4o")

        def _openai_text_batch(model=None, max_tokens: int = 1000):
            conc = int(os.getenv("JUDGE_CONCURRENCY", "8"))

            def judge_batch(prompts):
                if not prompts:
                    return []
                out = [""] * len(prompts)

                def _one(i):
                    try:
                        out[i] = H.openai_chat([{"role": "user", "content": prompts[i]}],
                                               model=_oai_model, max_tokens=max_tokens, temperature=0.0)
                    except Exception as e:   # noqa: BLE001  fail-closed like the Anthropic original
                        print(f"[judge-patch] openai text-judge error (-> '' for item {i}): {e!r}", flush=True)
                    rec("make_openai_judge_text_batch", prompts[i], out[i])

                with ThreadPoolExecutor(max_workers=conc) as ex:
                    list(ex.map(_one, range(len(prompts))))
                return out
            return judge_batch

        J.make_anthropic_judge_text_batch = _openai_text_batch
        for _m in list(sys.modules.values()):          # modules that already did `from ... import make_anthropic_judge_text_batch`
            if _m is not None and _m is not J and getattr(_m, "make_anthropic_judge_text_batch", None) is not None:
                setattr(_m, "make_anthropic_judge_text_batch", _openai_text_batch)

        def _openai_bool_batch(model=None):
            j = J.make_openai_judge(model=_oai_model)   # wrapped above -> records kind "make_openai_judge"

            def judge_batch(prompts):
                if not prompts:
                    return []
                with ThreadPoolExecutor(max_workers=int(os.getenv("JUDGE_CONCURRENCY", "8"))) as ex:
                    return list(ex.map(j, prompts))
            return judge_batch
        J.make_local_judge_batch = _openai_bool_batch

        _orig_resolve_oai = E._resolve_judge_fn

        def _resolve_oai(model=None, *a, **k):     # a benchmark declaring a claude judge_model would otherwise be sent to OpenAI verbatim
            if isinstance(model, str) and model.startswith("claude"):
                print(f"[judge-patch] judge_model {model!r} -> {_oai_model} (JUDGE_BACKEND=openai)", flush=True)
                model = _oai_model
            return _orig_resolve_oai(model, *a, **k)
        E._resolve_judge_fn = _resolve_oai
    orig_resolve = E._resolve_judge_fn

    def resolve(*a, **k):                      # build_benchmark() passes the benchmark's declared judge model
        fn = orig_resolve(*a, **k)
        if fn is None:
            return None

        def w(p):
            o = fn(p)
            rec("suite_judge", p, o)
            return o
        return w
    E._resolve_judge_fn = resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=sorted(TARGET_MODELS))
    ap.add_argument("--model", default=None, help="HF id / local path / stub:perfect (default: paper target model)")
    ap.add_argument("--repo", default=str(HERE / "aar_repo"))
    ap.add_argument("--suites", default=str(HERE / "aar_suites"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--replay", action="store_true", help="Phase 2: replay calls.jsonl with real judges")
    ap.add_argument("--with-capability", action="store_true", help="also run mmlu/gsm8k/ifeval")
    ap.add_argument("--only", nargs="*", help="only these benchmark ids")
    ap.add_argument("--stub-judges", action="store_true", help="force stubbed judges even in --replay (mechanics test)")
    args = ap.parse_args()

    repo, suites = Path(args.repo).resolve(), Path(args.suites).resolve()
    model_ref = args.model or TARGET_MODELS[args.axis]
    out_dir = Path(args.out).resolve() / args.axis / _tag(model_ref)
    out_dir.mkdir(parents=True, exist_ok=True)
    calls_path = out_dir / "calls.jsonl"
    phase = "replay" if args.replay else "generate"
    if not args.replay and calls_path.exists() and calls_path.stat().st_size > 0 and not args.only:
        sys.exit(f"{calls_path} already exists — delete it, choose another --out, or pass --only <benchmarks> to append missing legs.")
    if not args.replay and args.only and calls_path.exists():
        done_benches = {json.loads(l)["bench"] for l in open(calls_path, encoding="utf-8") if l.strip()}
        clash = [b for b in args.only if b in done_benches]
        if clash:
            sys.exit(f"{clash} already recorded in {calls_path}; remove those lines first to regenerate them.")
    if args.replay and not calls_path.exists():
        sys.exit(f"--replay needs {calls_path} from Phase 1.")

    sys.path.insert(0, str(repo))
    os.environ["BENCHMARK_DOCS_DIR"] = str(repo / "benchmark_docs")
    os.environ["AAR_BENCHMARK_DOCS"] = str(repo / "benchmark_docs")
    os.environ["EVAL_GPUS"] = "1"
    os.environ.setdefault("PYTHONUTF8", "1")
    os.chdir(repo)   # some loaders use relative paths

    rec = Recorder(calls_path) if not args.replay else None
    jrec = Recorder(out_dir / "judge_calls.jsonl") if args.replay else None
    install_judge_patches(stub=(not args.replay) or args.stub_judges, jrec=jrec)

    import aar.eval_pod.run_eval as E
    import aar.eval_pod.models as M
    from aar.benchmarks import registry as REG
    from aar.benchmarks.base import BenchmarkScore

    # Repo quirk: the bias_* benchmarks subclass JudgeBenchmark but override score() without defining the
    # abstract judge_prompt(), so Python refuses to instantiate them. Give every registered class a no-op
    # for any still-abstract method (it is never called because score() is overridden).
    for _name, _cls in REG.discover().items():
        _abs = set(getattr(_cls, "__abstractmethods__", ()) or ())
        if _abs:
            for _m in _abs:
                setattr(_cls, _m, lambda self, *a, **k: "")
            _cls.__abstractmethods__ = frozenset()

    log = open(out_dir / f"{phase}.log", "a", encoding="utf-8")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    errors: dict[str, str] = {}
    timings: dict[str, float] = {}

    def score_sequential(specs, model_ref_, secret_dir, suite_name=""):
        model = ReplayModel(calls_path) if args.replay else wrap_model(M.load_model(model_ref_), rec)
        judge_fn = E._resolve_judge_fn()
        scores = {}
        for spec in specs:
            skip = (spec.role == "capability_filter" and not args.with_capability) or (args.only and spec.id not in args.only)
            if skip:
                scores[spec.id] = BenchmarkScore(0.0, 0.0, 0.0, 0)
                say(f"  {spec.id:24s} skipped")
                continue
            if rec is not None:
                rec.bench = spec.id
            if isinstance(model, ReplayModel):
                model.bench = spec.id
            if jrec is not None:
                jrec.bench = spec.id
            t0 = time.time()
            try:
                E._apply_golden(model, spec, suite_name)
                sc = E.build_benchmark(spec, secret_dir, real_judge_fn=judge_fn).score(model)
                scores[spec.id] = sc
                say(f"  {spec.id:24s} [{spec.category:10s} {spec.role:18s}] mean={sc.mean:.4f} n={sc.n}  "
                    f"({time.time() - t0:.0f}s, {rec.n if rec else 0} calls recorded so far)")
            except Exception as e:   # noqa: BLE001 — one benchmark must not sink the axis
                errors[spec.id] = repr(e)
                scores[spec.id] = BenchmarkScore(0.0, 0.0, 0.0, 0)
                say(f"  {spec.id:24s} FAILED: {e!r}")
                log.write(traceback.format_exc() + "\n")
            timings[spec.id] = round(time.time() - t0, 1)
        return scores

    E._score_sequential = score_sequential
    if args.replay:   # run() also calls load_model() outside _score_sequential (decoding print / capability path): never load weights on replay
        E.load_model = lambda ref, *a, **k: ReplayModel(calls_path)
    suite_yaml = suites / args.axis / f"{args.axis}.yaml"
    say(f"== {phase} axis={args.axis} model={model_ref} suite={suite_yaml} out={out_dir}  {time.strftime('%F %T')}")
    t0 = time.time()
    try:
        E.run(str(suite_yaml), model_ref, str(suites / args.axis), str(out_dir / f"scores_{phase}.json"), str(out_dir))
    except Exception as e:   # noqa: BLE001
        say(f"run() ended with error (per-benchmark outputs are still on disk): {e!r}")
        log.write(traceback.format_exc() + "\n")
    meta = {"phase": phase, "axis": args.axis, "model": model_ref, "started": t0, "seconds": round(time.time() - t0, 1),
            "errors": errors, "timings": timings, "calls_recorded": rec.n if rec else None,
            "judge_calls_recorded": jrec.n if jrec else None, "repo_commit": (repo / "COMMIT.txt").read_text().strip()
            if (repo / "COMMIT.txt").exists() else None}
    (out_dir / f"{phase}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    say(f"== done in {meta['seconds']}s; errors={len(errors)}; calls={meta['calls_recorded']} judge_calls={meta['judge_calls_recorded']}")


if __name__ == "__main__":
    main()
