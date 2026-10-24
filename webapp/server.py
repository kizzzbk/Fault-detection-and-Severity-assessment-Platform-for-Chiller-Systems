"""FastAPI backend for the Chiller Digital Twin demo.

* ``GET  /api/config``   slider definitions anchored on RP-1043 medians
* ``POST /api/simulate`` run the twin for N hours with the injected faults and
                         return the generated sensor stream + CNN diagnosis
* ``GET  /healthz``      liveness probe for uptime monitors
"""

from __future__ import annotations

import json
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from demo.cnn_schema import CNN_FEATURE_COLUMNS  # noqa: E402
from demo.pipeline import WINDOW, diagnose_frame, load_bundle  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STATIC = HERE / "static"
DT_S = 10.0
FAULTS = ["fwc", "fwe", "cf", "ro", "rl", "nc", "eo"]

# fault -> (slider variable, label shown on the machine, unit, component)
SLIDERS = {
    "fwc": ("CDW_DT", "Condenser water ΔT", "°F", "condenser"),
    "fwe": ("CHW_DT", "Chilled water ΔT", "°F", "evaporator"),
    "cf": ("TCA", "Condenser approach", "°F", "condenser"),
    "ro": ("TRC_sub", "Liquid subcooling", "°F", "expansion"),
    "rl": ("TRC_sub", "Liquid subcooling", "°F", "expansion"),
    "nc": ("PRC", "Condensing pressure", "psig", "condenser"),
    "eo": ("PO_net", "Net oil pressure", "psi", "compressor"),
}

app = FastAPI(title="Chiller Digital Twin", docs_url="/api/docs")
app.add_middleware(GZipMiddleware, minimum_size=2048)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def library() -> pd.DataFrame:
    return pd.read_parquet(DATA / "twin_library.parquet")


@lru_cache(maxsize=1)
def anchors() -> dict:
    return json.loads((DATA / "anchors.json").read_text())


@lru_cache(maxsize=1)
def bundle() -> dict:
    return load_bundle()


def slider_config() -> list[dict]:
    a = anchors()
    base = a["normal"]["0"]
    out = []
    for fault, (var, label, unit, comp) in SLIDERS.items():
        levels = {int(s): v[var] for s, v in a.get(fault, {}).items()}
        nominal = base[var]
        vals = [nominal, *levels.values()]
        direction = 1 if np.mean(list(levels.values())) >= nominal else -1
        span = max(abs(v - nominal) for v in vals) or 1.0
        lo, hi = (nominal - 0.25 * span, nominal + 1.2 * span) if direction > 0 else (nominal - 1.2 * span, nominal + 0.25 * span)
        out.append(
            dict(
                fault=fault, var=var, label=label, unit=unit, component=comp,
                nominal=round(nominal, 2), min=round(lo, 2), max=round(hi, 2),
                direction=direction,
                levels={str(k): round(v, 2) for k, v in sorted(levels.items())},
            )
        )
    return out


class SimRequest(BaseModel):
    hours: float = Field(2.0, ge=1.0, le=5.0)
    faults: dict[str, int] = Field(default_factory=dict)  # fault -> severity 1..4
    seed: int | None = None


def _segment(fault: str, sev: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    lib = library()
    block = lib[(lib.fault == fault) & (lib.severity == sev)]
    if block.empty:  # nearest available severity
        avail = lib.loc[lib.fault == fault, "severity"].unique()
        if len(avail) == 0:
            raise HTTPException(400, f"No signature for {fault}")
        sev = int(avail[np.argmin(np.abs(avail - sev))])
        block = lib[(lib.fault == fault) & (lib.severity == sev)]
    start = int(rng.integers(0, max(1, len(block) - n)))
    seg = block.iloc[start : start + n][CNN_FEATURE_COLUMNS].reset_index(drop=True).astype("float64")
    seg["scenario"] = fault if fault == "normal" else f"{fault}"
    seg["scenario_sev"] = sev
    return seg


def build_stream(req: SimRequest) -> pd.DataFrame:
    rng = np.random.default_rng(req.seed)
    n = int(round(req.hours * 3600 / DT_S))
    active = [(f, int(np.clip(s, 1, 4))) for f, s in req.faults.items() if f in FAULTS and s > 0]
    plan: list[tuple[str, int, int]] = []
    if not active:
        plan.append(("normal", 0, n))
    else:
        lead = max(WINDOW, int(0.2 * n))
        plan.append(("normal", 0, lead))
        rest = n - lead
        share = rest // len(active)
        for i, (f, s) in enumerate(active):
            plan.append((f, s, share if i < len(active) - 1 else rest - share * (len(active) - 1)))
    frame = pd.concat([_segment(f, s, k, rng) for f, s, k in plan if k > 0], ignore_index=True)
    frame.insert(0, "time", np.arange(len(frame)) * DT_S)
    return frame


@app.get("/healthz")
@app.head("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict:
    return {"sliders": slider_config(), "baseline": anchors()["normal"]["0"], "dt_s": DT_S, "window": WINDOW}


@app.post("/api/simulate")
def simulate(req: SimRequest) -> dict:
    stream = build_stream(req)
    with _lock:
        diag = diagnose_frame(stream[["time", *CNN_FEATURE_COLUMNS]], bundle())
    cols = ["time", *CNN_FEATURE_COLUMNS]
    return {
        "columns": cols,
        "rows": np.round(stream[cols].to_numpy(dtype=float), 3).tolist(),
        "scenario": stream["scenario"].tolist(),
        "diagnosis": {
            "n_windows": diag.n_windows,
            "fault_seconds": diag.fault_seconds,
            "health_ratio": diag.health_ratio,
            "by_fault_seconds": diag.by_fault_seconds,
            "t_span_s": diag.t_span_s,
            "windows": [w.__dict__ for w in diag.windows],
            "segments": [s.__dict__ for s in diag.segments],
        },
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def _warm() -> None:
    library(), anchors(), bundle()
