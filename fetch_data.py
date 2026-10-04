#!/usr/bin/env python3
"""Fetch the FDABench parquet shards this benchmark builds on.

Data is not committed; it is re-fetched from the HuggingFace parquet export.
Base benchmark: FDABench, arXiv:2509.02473 (KDD'26), MIT licence,
dataset `FDAbench2026/FDAbench-Full`.

Run:  ./.venv/bin/python fetch_data.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from common import CONFIGS, DATA

BASE = (
    "https://huggingface.co/datasets/FDAbench2026/FDAbench-Full/"
    "resolve/refs%2Fconvert%2Fparquet/{cfg}/train/0000.parquet"
)
# Recorded 2026-10-03. A mismatch means the upstream export changed; that is a
# reproducibility event worth knowing about, so it warns rather than passing.
KNOWN_SHA256 = {
    "report": "caedfb5dd6d15420dd97197e126f543e2b940e5fe9c9948747158eff20f28592",
    "multiple": "e4dac0c161e9d663127e0ece7c2ee66daec0a53de3e5cac4d32566339b0d681f",
    "single": "b80b864106c1e18dba3ab9b0141fe2c8c926af5de7f24545a398e8a907cc946d",
}


def main() -> int:
    DATA.mkdir(exist_ok=True)
    for cfg in CONFIGS:
        dest = DATA / f"{cfg}.parquet"
        if dest.exists():
            print(f"  have   {dest.name}")
        else:
            url = BASE.format(cfg=cfg)
            print(f"  fetch  {dest.name} <- {url}")
            urllib.request.urlretrieve(url, dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        known = KNOWN_SHA256.get(cfg)
        if known is None:
            print(f"         sha256 {digest}")
        elif digest != known:
            print(f"  WARN   {dest.name} differs from the recorded export", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
