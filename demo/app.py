"""Streamlit FDD dashboard: landing upload, then a one-screen diagnosis view."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from demo.pipeline import (
    DEMO_DIR,
    LANE_COLORS,
    LANE_KEYS,
    SHORT_LABELS,
    Diagnosis,
    diagnose_frame,
    format_duration,
    load_bundle,
)

st.set_page_config(
    page_title="Chiller Fault Detection and Diagnosis (FDD)",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DISPLAY_LANES: dict[str, str] = {
    "ro": "Overcharge (RO)",
    "rl": "Leak / Undercharge (RL)",
    "nc": "Non-condensable (NC)",
    "fwe": "Evaporator flow (FWE)",
    "fwc": "Condenser flow (FWC)",
    "eo": "Excess oil (EO)",
    "cf": "Condenser fouling (CF)",
    "normal": "Normal",
}

INK = "#0A0A0A"
AXIS_FONT = dict(color=INK, size=15, family="Arial, Helvetica, sans-serif")
TITLE_FONT = dict(color=INK, size=16, family="Arial, Helvetica, sans-serif")

PAGE_CSS = """
<style>
    html, body, .stApp {
        background: linear-gradient(180deg, #DFF6EC 0%, #E7F7F1 42%, #F2FBF6 100%) !important;
        overflow: hidden;
        height: 100vh;
        color: #0A0A0A;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu { display: none; }
    .block-container { padding: 0.55rem 1.1rem 0.3rem !important; max-width: 1400px; }
    [data-testid="stHeading"] { text-align: center; }
    [data-testid="stCaption"] { text-align: center; }

    .hero { max-width: 760px; margin: 0.4vh auto 0; text-align: center; }
    .hero h1 {
        margin: 0.2rem 0 0.4rem;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0A0A0A;
    }
    .hero .lead {
        margin: 0 auto 0.7rem;
        max-width: 38rem;
        color: #134E4A;
        font-size: 1.05rem;
        line-height: 1.45;
        font-weight: 600;
    }
    .hero .bits { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; margin-bottom: 0.4rem; }
    .hero .bit {
        background: #fff;
        border: 1.5px solid #0A0A0A;
        color: #0A0A0A;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .chiller {
        width: 250px;
        height: 250px;
        margin: 0 auto;
        filter: drop-shadow(0 14px 22px rgba(16, 185, 129, 0.22));
    }
    .fan { transform-origin: 84px 62px; animation: spin 1.8s linear infinite; }
    .glow { animation: pulse 2.2s ease-in-out infinite; }
    .flow { stroke-dasharray: 6 8; animation: dash 1.1s linear infinite; }
    .flake { animation: flake 2.8s ease-in-out infinite; }
    .flake.f2 { animation-delay: -1.1s; }
    .flake.f3 { animation-delay: -1.9s; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes pulse { 50% { opacity: 0.55; } }
    @keyframes dash { to { stroke-dashoffset: -28; } }
    @keyframes flake {
        0%, 100% { transform: translateY(0); opacity: 0.9; }
        50% { transform: translateY(8px); opacity: 0.35; }
    }

    [data-testid="stFileUploader"] { padding-top: 0; }
    [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 2px dashed #0F766E !important;
        border-radius: 20px !important;
        min-height: 132px;
        padding: 18px 16px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        gap: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] > * {
        margin-left: auto !important;
        margin-right: auto !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0A0A0A !important;
        font-weight: 700 !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { stroke: #0F766E; }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderInstructions"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 1rem !important;
        display: block !important;
        text-align: center !important;
        line-height: 1.35 !important;
    }

    .progress-wrap {
        background: #fff;
        border: 1.5px solid #0A0A0A;
        border-radius: 18px;
        padding: 18px 20px 14px;
        margin-top: 0.6rem;
    }
    .progress-wrap .k {
        font-weight: 800;
        color: #0A0A0A;
        font-size: 1.05rem;
    }

    .topbar {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 0.45rem;
    }
    .topbar h1 {
        margin: 0;
        font-size: 1.28rem;
        font-weight: 800;
        color: #0A0A0A;
        letter-spacing: -0.02em;
    }
    .chart-title {
        font-size: 1.02rem;
        font-weight: 800;
        color: #0A0A0A;
        margin: 0 0 6px;
        letter-spacing: -0.01em;
    }
    .fdd-card {
        background: #ffffff;
        border: 1.5px solid #0A0A0A;
        border-radius: 12px;
        padding: 10px 12px 8px;
        margin-bottom: 8px;
    }
    .fdd-k { color: #0A0A0A; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.01em; }
    .fdd-v { color: #0A0A0A; font-size: 1.35rem; font-weight: 800; line-height: 1.15; }
    .fdd-sub { color: #0A0A0A; font-size: 0.86rem; font-weight: 700; margin: 3px 0 0; }
    .fdd-report {
        margin: 0 0 2px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: #0A0A0A;
    }
    .legend-item {
        color: #0A0A0A;
        font-size: 1.05rem;
        font-weight: 800;
        margin: 5px 0 0;
        line-height: 1.35;
    }
    .fdd-dot {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%;
        margin-right: 8px; border: 1.5px solid #0A0A0A; vertical-align: -1px;
    }
    .stats-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }
    [data-testid="stBaseButton-primary"] {
        background: #0F766E !important;
        color: #FFFFFF !important;
        border: 2.5px solid #0A0A0A !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        box-shadow: 3px 3px 0 #0A0A0A !important;
        height: 2.7rem !important;
    }
</style>
"""

@st.cache_resource
def _artifacts(version: str = "ca_tcc_cnn_ckp_ft"):
    return load_bundle()


def _pie_slices(diag: Diagnosis) -> list[tuple[str, str, float]]:
    slices = []
    for key in ("normal", *[k for k in LANE_KEYS if k != "normal"]):
        sec = diag.by_fault_seconds.get(key, 0.0)
        if sec > 0:
            slices.append((SHORT_LABELS[key], LANE_COLORS[key], sec))
    return slices or [("Normal", LANE_COLORS["normal"], 1.0)]


def _pie(diag: Diagnosis) -> go.Figure:
    slices = _pie_slices(diag)
    labels = [name for name, _, _ in slices]
    colors = [color for _, color, _ in slices]
    values = [sec for _, _, sec in slices]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.56,
            sort=False,
            direction="clockwise",
            rotation=0,
            textinfo="percent",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont=dict(size=16, color=INK, family="Arial, Helvetica, sans-serif"),
            marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
            showlegend=False,
            texttemplate="%{percent:.0%}",
        )
    )
    fig.update_layout(
        margin=dict(t=8, b=4, l=4, r=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        font=AXIS_FONT,
        annotations=[
            dict(
                text=f"{diag.health_ratio * 100:.0f}%",
                x=0.5,
                y=0.5,
                font=dict(size=28, color=INK, family="Arial, Helvetica, sans-serif"),
                showarrow=False,
            )
        ],
    )
    return fig


def _pie_legend(diag: Diagnosis) -> str:
    rows = []
    for name, color, _sec in _pie_slices(diag):
        rows.append(
            f'<div class="legend-item"><span class="fdd-dot" style="background:{color}"></span>{name}</div>'
        )
    return "".join(rows)


def _lanes(diag: Diagnosis) -> go.Figure:
    origin = min((w.t0_s for w in diag.windows), default=0.0)
    y_bottom_first = list(reversed(LANE_KEYS))
    fig = go.Figure()
    for key in y_bottom_first:
        fig.add_trace(
            go.Bar(
                y=[DISPLAY_LANES[key]],
                x=[0],
                base=[0],
                orientation="h",
                marker=dict(color="rgba(0,0,0,0)"),
                width=0.58,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for seg in diag.segments:
        x0 = (seg.t0_s - origin) / 3600.0
        x1 = (seg.t1_s - origin) / 3600.0
        if x1 <= x0:
            x1 = x0 + 1e-4
        fig.add_trace(
            go.Bar(
                y=[DISPLAY_LANES[seg.fault]],
                x=[x1 - x0],
                base=[x0],
                orientation="h",
                marker=dict(color=LANE_COLORS[seg.fault], line=dict(color=INK, width=2.4)),
                width=0.58,
                showlegend=False,
                hovertemplate=(
                    f"{SHORT_LABELS[seg.fault]}<br>"
                    f"{x0:.2f}–{x1:.2f} h<br>"
                    f"{format_duration(seg.duration_s)}<extra></extra>"
                ),
            )
        )
    span_h = max(diag.t_span_s / 3600.0, 0.25)
    x_end = span_h * 1.06
    fig.update_layout(
        height=620,
        barmode="overlay",
        bargap=0.22,
        margin=dict(t=8, b=56, l=10, r=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=AXIS_FONT,
        xaxis=dict(
            title=dict(text="Time (h)", font=TITLE_FONT),
            range=[-0.02 * span_h, x_end],
            showgrid=False,
            zeroline=False,
            linecolor=INK,
            linewidth=2.5,
            ticks="outside",
            tickcolor=INK,
            tickfont=AXIS_FONT,
            title_font=TITLE_FONT,
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=[DISPLAY_LANES[k] for k in y_bottom_first],
            showgrid=False,
            zeroline=False,
            linecolor=INK,
            linewidth=2.5,
            ticks="outside",
            tickcolor=INK,
            tickfont=AXIS_FONT,
        ),
    )
    fig.update_xaxes(showline=True, mirror=False, linecolor=INK, linewidth=2.5)
    fig.update_yaxes(showline=True, mirror=False, linecolor=INK, linewidth=2.5)
    return fig


def _read_frame(uploaded) -> pd.DataFrame:
    sample_path = DEMO_DIR / "sample.csv"
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(uploaded)
        return pd.read_parquet(uploaded)
    if sample_path.is_file():
        return pd.read_csv(sample_path)
    raise FileNotFoundError("Chưa tải file và demo/sample.csv cũng không có.")


def _run_classification(uploaded) -> Diagnosis:
    box = st.empty()
    bar = st.empty()
    steps = [
        (12, "Reading sensor file"),
        (38, "Cutting windows 128 / 64"),
        (72, "Predicting fault labels"),
        (88, "Building the timeline"),
    ]
    with box.container():
        st.markdown(
            '<div class="progress-wrap"><div class="k">Classification Progress</div></div>',
            unsafe_allow_html=True,
        )
    for value, label in steps:
        bar.progress(value, text=label)
        time.sleep(0.16)
    diag = diagnose_frame(_read_frame(uploaded), _artifacts())
    bar.progress(100, text="Done")
    time.sleep(0.12)
    box.empty()
    bar.empty()
    return diag


def _page_landing() -> None:
    components.html(
        """
        <div style="text-align:center;font-family:Arial,Helvetica,sans-serif;color:#0A0A0A">
          <div style="width:250px;height:250px;margin:0 auto 8px;filter:drop-shadow(0 14px 22px rgba(16,185,129,.22))">
            <svg viewBox="0 0 168 168" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="cab" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#7DD3FC"/>
                  <stop offset="55%" stop-color="#2563EB"/>
                  <stop offset="100%" stop-color="#1E3A8A"/>
                </linearGradient>
                <linearGradient id="coil" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stop-color="#FDE68A"/>
                  <stop offset="100%" stop-color="#F59E0B"/>
                </linearGradient>
                <style>
                  .fan { transform-origin: 84px 62px; animation: spin 1.8s linear infinite; }
                  .glow { animation: pulse 2.2s ease-in-out infinite; }
                  .flow { stroke-dasharray: 6 8; animation: dash 1.1s linear infinite; }
                  .flake { animation: flake 2.8s ease-in-out infinite; }
                  .f2 { animation-delay: -1.1s; }
                  .f3 { animation-delay: -1.9s; }
                  @keyframes spin { to { transform: rotate(360deg); } }
                  @keyframes pulse { 50% { opacity: .55; } }
                  @keyframes dash { to { stroke-dashoffset: -28; } }
                  @keyframes flake { 50% { transform: translateY(8px); opacity: .35; } }
                </style>
              </defs>
              <circle class="flake" cx="28" cy="28" r="5" fill="#38BDF8"/>
              <circle class="flake f2" cx="142" cy="24" r="4" fill="#A78BFA"/>
              <circle class="flake f3" cx="150" cy="52" r="3.5" fill="#34D399"/>
              <rect x="38" y="34" width="92" height="102" rx="16" fill="url(#cab)"/>
              <rect class="glow" x="48" y="86" width="72" height="36" rx="8" fill="#FDE68A"/>
              <path d="M54 96h60M54 104h60M54 112h60" stroke="url(#coil)" stroke-width="3" stroke-linecap="round"/>
              <circle cx="84" cy="62" r="22" fill="#DBEAFE"/>
              <g class="fan">
                <path d="M84 62 l0 -18 a8 8 0 0 1 10 16 z" fill="#F97316"/>
                <path d="M84 62 l16 9 a8 8 0 0 1 -18 6 z" fill="#34D399"/>
                <path d="M84 62 l-16 9 a8 8 0 0 1 8 -22 z" fill="#A78BFA"/>
                <circle cx="84" cy="62" r="5" fill="#0A0A0A"/>
              </g>
              <path class="flow" d="M38 122 H18 V78" fill="none" stroke="#22D3EE" stroke-width="6" stroke-linecap="round"/>
              <path class="flow" d="M130 122 H150 V78" fill="none" stroke="#FB7185" stroke-width="6" stroke-linecap="round"/>
              <rect x="56" y="138" width="56" height="10" rx="4" fill="#0A0A0A"/>
            </svg>
          </div>
          <h1 style="margin:4px 0 8px;font-size:32px;font-weight:800;letter-spacing:-0.03em">Chiller Fault Detection &amp; Diagnosis</h1>
          <p style="margin:0 auto 12px;max-width:560px;font-size:17px;font-weight:600;line-height:1.4;color:#134E4A">Tải lên 1 file thông số cảm biến và nhận chẩn đoán lỗi chi tiết theo thời gian từ chúng tôi</p>
          <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
            <span style="background:#fff;border:1.5px solid #0A0A0A;border-radius:999px;padding:4px 12px;font-size:13px;font-weight:700">8 nhãn RP-1043</span>
            <span style="background:#fff;border:1.5px solid #0A0A0A;border-radius:999px;padding:4px 12px;font-size:13px;font-weight:700">Cửa sổ 128 / bước 64</span>
            <span style="background:#fff;border:1.5px solid #0A0A0A;border-radius:999px;padding:4px 12px;font-size:13px;font-weight:700">CSV / Parquet</span>
          </div>
        </div>
        """,
        height=430,
    )
    left, mid, right = st.columns([0.16, 0.68, 0.16])
    with mid:
        st.markdown("**Kéo file .csv hoặc .parquet vào khung bên dưới**")
        uploaded = st.file_uploader(
            "Kéo file vào đây hoặc bấm để chọn",
            type=["csv", "parquet"],
            label_visibility="collapsed",
        )
        sample = st.button("Chạy file mẫu", use_container_width=True)
        if uploaded is not None or sample:
            try:
                st.session_state.diag = _run_classification(uploaded if uploaded is not None else None)
                st.session_state.source_name = (
                    uploaded.name if uploaded is not None else "sample.csv"
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _page_results(diag: Diagnosis) -> None:
    name = st.session_state.get("source_name", "uploaded file")
    st.markdown("**DIAGNOSTIC REPORT BASED ON CHILLER SENSORS**")
    st.markdown(f"### Source file: {name}")

    left, right = st.columns([0.32, 0.68], gap="medium")
    with left:
        st.markdown('<div class="chart-title">Operating health</div>', unsafe_allow_html=True)
        lines = [
            f'<div class="fdd-sub">- {SHORT_LABELS[key]}: {format_duration(sec)}</div>'
            for key, sec in diag.by_fault_seconds.items()
            if key != "normal" and sec > 0
        ]
        st.markdown(
            f"""
            <div class="stats-row">
              <div class="fdd-card"><div class="fdd-k">Faults</div><div class="fdd-v">{diag.n_fault_types}</div></div>
              <div class="fdd-card"><div class="fdd-k">Time with faults</div><div class="fdd-v">{format_duration(diag.fault_seconds)}</div></div>
              <div class="fdd-card"><div class="fdd-k">Health ratio</div><div class="fdd-v">{diag.health_ratio * 100:.0f}%</div></div>
            </div>
            <div class="fdd-card">{''.join(lines)}</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(_pie(diag), use_container_width=True, config={"displayModeBar": False})
        st.markdown(_pie_legend(diag), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="chart-title">Fault lanes over time</div>', unsafe_allow_html=True)
        st.plotly_chart(_lanes(diag), use_container_width=True, config={"displayModeBar": False})

    _, mid, _ = st.columns([0.32, 0.36, 0.32])
    with mid:
        if st.button("Phân tích file khác", type="primary", use_container_width=True):
            st.session_state.pop("diag", None)
            st.session_state.pop("source_name", None)
            st.rerun()


def main() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    if "diag" in st.session_state:
        _page_results(st.session_state.diag)
        return
    _page_landing()


if __name__ == "__main__":
    main()
