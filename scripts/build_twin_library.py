"""Build the digital-twin signature library for the web demo.

For every (fault, severity) in ASHRAE RP-1043 we keep one contiguous block of
raw sensor rows on which the CA-TCC CNN agrees with the ground-truth label, and
record the median of each fault's root variable so the UI sliders are anchored
to real measured values.

Output: webapp/data/twin_library.parquet + webapp/data/anchors.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fit_cnn_demo_preprocess import _attach_extras

from chiller_fault_detection.config import DATASET_PARQUET
from chiller_fault_detection.data.public.preprocess import (
    filter_kw_range,
    sort_within_files,
)
from chiller_fault_detection.labels import ID_TO_LABEL
from demo.cnn_schema import CNN_FEATURE_COLUMNS
from demo.pipeline import (
    WINDOW,
    cut_windows,
    load_bundle,
    predict_windows,
    prepare_model_frame,
)

OUT_DIR = REPO / "webapp" / "data"
BLOCK_ROWS = 1920  # 5h20' at 10 s/sample
ROOT_VARS = ["CDW_DT", "CHW_DT", "TCA", "TRC_sub", "Tsh_suc", "PRC", "PO_net", "TRE", "kW", "TCI", "TEO"]


def best_block(frame: pd.DataFrame, fault: str, bundle) -> tuple[pd.DataFrame, float]:
    """Return the longest run of rows whose windows the CNN labels as ``fault``."""
    best, best_score = None, -1.0
    for source, group in frame.groupby("source_file", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        if len(group) < WINDOW * 2:
            continue
        g = group.copy()
        g["source_file"] = "x"
        prepared, _ = prepare_model_frame(g[["time", *CNN_FEATURE_COLUMNS]], bundle["residualizer"], bundle["scaler"])
        if len(prepared) != len(group):
            continue
        windows, meta = cut_windows(prepared)
        pred, _proba = predict_windows(bundle["model"], windows, bundle["device"])
        ok = np.array([ID_TO_LABEL[int(p)].value == fault for p in pred])
        # longest run of correct windows
        run_start, run_len, cur_s, cur_l = 0, 0, 0, 0
        for i, flag in enumerate(ok):
            if flag:
                if cur_l == 0:
                    cur_s = i
                cur_l += 1
                if cur_l > run_len:
                    run_start, run_len = cur_s, cur_l
            else:
                cur_l = 0
        if run_len == 0:
            continue
        r0 = meta[run_start]["start_idx"]
        r1 = meta[run_start + run_len - 1]["end_idx"]
        score = run_len + ok.mean()
        if score > best_score:
            best_score = score
            best = group.iloc[r0:r1].copy()
            best.attrs["acc"] = float(ok.mean())
            best.attrs["src"] = source
    return best, best_score


def tile(block: pd.DataFrame, n: int) -> pd.DataFrame:
    """Repeat a block forward/backward (ping-pong) to reach n rows without jumps."""
    parts, fwd = [], True
    while sum(len(p) for p in parts) < n:
        parts.append(block if fwd else block.iloc[::-1])
        fwd = not fwd
    return pd.concat(parts, ignore_index=True).iloc[:n]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    dataset = pd.read_parquet(DATASET_PARQUET)
    frame = _attach_extras(dataset)
    frame, _ = filter_kw_range(frame)
    frame = sort_within_files(frame)

    blocks, anchors = [], {}
    for (fault, sev), part in frame.groupby(["fault", "severity_level"]):
        sev = int(sev)
        block, _score = best_block(part, fault, bundle)
        if block is None:
            print(f"!! {fault}/{sev}: no correctly classified window, skipped")
            continue
        print(f"{fault}/{sev}: {len(block)} rows from {block.attrs['src']} (file acc {block.attrs['acc']:.2f})")
        out = tile(block[CNN_FEATURE_COLUMNS].reset_index(drop=True), BLOCK_ROWS)
        out["fault"], out["severity"], out["step"] = fault, sev, np.arange(len(out))
        blocks.append(out)
        anchors.setdefault(fault, {})[str(sev)] = {v: float(part[v].median()) for v in ROOT_VARS}

    lib = pd.concat(blocks, ignore_index=True)
    for c in CNN_FEATURE_COLUMNS:
        lib[c] = lib[c].astype("float32")
    lib.to_parquet(OUT_DIR / "twin_library.parquet", index=False)
    (OUT_DIR / "anchors.json").write_text(json.dumps(anchors, indent=1))
    print(f"Wrote {len(lib)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
