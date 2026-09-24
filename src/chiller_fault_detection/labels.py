"""Hierarchical labels for ASHRAE RP-1043.

Level 1 — fault type (8-way).
Level 2 — severity_level 0–4 (0 = normal, 1–4 = SL1–SL4).

``severity_pct`` is inferred from the filename only long enough to map to
``severity_level``; it is not stored in the parquet dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict


class FaultType(str, Enum):
    NORMAL = "normal"
    CF = "cf"
    FWE = "fwe"
    FWC = "fwc"
    RO = "ro"
    RL = "rl"
    NC = "nc"
    EO = "eo"


LABEL_TO_ID: Dict[FaultType, int] = {
    FaultType.NORMAL: 0,
    FaultType.CF: 1,
    FaultType.FWE: 2,
    FaultType.FWC: 3,
    FaultType.RO: 4,
    FaultType.RL: 5,
    FaultType.NC: 6,
    FaultType.EO: 7,
}

ID_TO_LABEL: Dict[int, FaultType] = {v: k for k, v in LABEL_TO_ID.items()}

VALID_FAULT_IDS = frozenset(LABEL_TO_ID.values())
VALID_SEVERITY_LEVELS = frozenset({0, 1, 2, 3, 4})

# Percent in filename → SL1–SL4 (RP-1043 canonical levels).
SEVERITY_PCT_TO_LEVEL: Dict[FaultType, Dict[int, int]] = {
    FaultType.CF: {6: 1, 12: 1, 20: 2, 30: 3, 45: 4},
    FaultType.FWE: {10: 1, 20: 2, 30: 3, 40: 4},
    FaultType.FWC: {10: 1, 20: 2, 30: 3, 40: 4},
    FaultType.RL: {10: 1, 20: 2, 30: 3, 40: 4},
    FaultType.RO: {10: 1, 20: 2, 30: 3, 40: 4},
    FaultType.EO: {14: 1, 32: 2, 50: 3, 68: 4, 73: 4},
    FaultType.NC: {1: 1, 2: 2, 3: 3, 5: 4},
}

_FAULT_FROM_STEM = re.compile(
    r"(?P<fault>fwe|fwc|cf|rl|ro|eo|nc)(?P<pct>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceLabel:
    fault: FaultType
    severity_level: int

    @property
    def fault_id(self) -> int:
        return LABEL_TO_ID[self.fault]

    @property
    def path_label(self) -> str:
        return f"{self.fault.value}/{self.severity_level}"


def severity_level_for(fault: FaultType, severity_pct: int) -> int:
    if fault is FaultType.NORMAL:
        return 0
    table = SEVERITY_PCT_TO_LEVEL.get(fault)
    if table is None or severity_pct not in table:
        raise ValueError(f"No SL mapping for {fault.value} at {severity_pct}%")
    return table[severity_pct]


def parse_source_label(path: Path) -> SourceLabel:
    """Infer hierarchical labels from an RP-1043 test filename.

    Exclusion of unwanted files is handled by ``load.EXCLUDED_FILES``, not here.
    Unrecognized names raise ``ValueError``.
    """
    stem = path.stem.lower().replace(" ", "")

    if stem.startswith("normal"):
        return SourceLabel(FaultType.NORMAL, 0)

    match = _FAULT_FROM_STEM.search(path.stem.lower())
    if match is None:
        raise ValueError(f"Unrecognized RP-1043 filename for labeling: {path.name}")

    fault = FaultType(match.group("fault").lower())
    pct = int(match.group("pct"))
    return SourceLabel(fault, severity_level_for(fault, pct))
