#!/usr/bin/env python
"""把本数据包的目录布局还原成脚本期望的工作布局，之后即可离线重算全部指标：

    python code/restore_layout.py            # 在包根目录下运行
    python code/run_jev.py --leg <leg> --battery <battery> \
           --samples code/samples/<file>.jsonl --rescore    # 不发任何请求

还原后新增两个目录（软链接不可用时为复制）：
    code/samples/                  <- data/detection_samples/
    code/results/<leg>/<run>/...   <- results/jev_responses/ + results/metrics/
原目录保持不变。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = Path(__file__).resolve().parent   # run_jev.py 以自身所在目录为基准

def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)

n = 0
for p in (ROOT / "data/detection_samples").glob("*.jsonl"):
    link_or_copy(p, CODE / "samples" / p.name)
    n += 1
m = 0
for group in ("jev_responses", "metrics"):
    for p in (ROOT / "results" / group).glob("*/*/*"):
        leg, run = p.parent.parent.name, p.parent.name
        link_or_copy(p, CODE / "results" / leg / run / p.name)
        m += 1
print(f"samples/ {n} 个文件；results/<leg>/<run>/ {m} 个文件。现在可以用 --rescore 离线重算。")
