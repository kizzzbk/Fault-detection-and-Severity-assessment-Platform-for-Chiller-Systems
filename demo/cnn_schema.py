"""Column contract the CA-TCC CNN checkpoint was trained with (8 Oct 2026).

``ckp_ft.pt`` expects 19 channels in this exact order. The public schema later
dropped ``T_suc``, ``Tsh_dis``, ``TEA`` and ``PO_feed``.
"""

from __future__ import annotations

CNN_OPERATING_COLUMNS: list[str] = ["TEO", "TCI", "kW"]

CNN_RESIDUAL_COLUMNS: list[str] = [
    "TEI",
    "TCO",
    "TRE",
    "T_suc",
    "Tsh_suc",
    "Tsh_dis",
    "PRE",
    "TRC",
    "TRC_sub",
    "TR_dis",
    "TEA",
    "TCA",
    "PRC",
    "P_lift",
    "TO_sump",
    "CDW_DT",
    "CHW_DT",
]

CNN_KEEP_RAW_COLUMNS: list[str] = ["PO_net", "PO_feed"]

CNN_MODEL_FEATURE_COLUMNS: list[str] = [*CNN_RESIDUAL_COLUMNS, *CNN_KEEP_RAW_COLUMNS]

CNN_FEATURE_COLUMNS: list[str] = [
    "TEI",
    "TEO",
    "TCI",
    "TCO",
    "TRE",
    "TRC",
    "TRC_sub",
    "T_suc",
    "Tsh_suc",
    "TR_dis",
    "Tsh_dis",
    "TEA",
    "TCA",
    "kW",
    "TO_sump",
    "PO_net",
    "PRE",
    "PRC",
    "P_lift",
    "PO_feed",
    "CDW_DT",
    "CHW_DT",
]

CNN_SOURCE_FEATURE_COLUMNS: list[str] = [
    col for col in CNN_FEATURE_COLUMNS if col not in ("CDW_DT", "CHW_DT")
]

CNN_INPUT_CHANNELS = len(CNN_MODEL_FEATURE_COLUMNS)
CKPT_NAME = "ckp_ft.pt"
