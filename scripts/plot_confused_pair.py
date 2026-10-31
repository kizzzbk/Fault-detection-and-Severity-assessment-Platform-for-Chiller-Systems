"""Plot confused A→B windows as mean±std, 2-channel scatter, or PCA.

A window is 19 channels × 128 steps, so a scatter needs two numbers per
window: either two named channels (time-averaged) or the first two principal
components of all 19.

Example:
  python scripts/plot_confused_pair.py --true normal --pred rl --split test
  python scripts/plot_confused_pair.py --true normal --pred rl --scatter TRC TCA
  python scripts/plot_confused_pair.py --true normal --pred rl --pca
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from chiller_fault_detection.config import PROCESSED_DIR, SCALER_PATH
from chiller_fault_detection.data.load import MODEL_FEATURE_COLUMNS
from chiller_fault_detection.labels import LABEL_TO_ID, FaultType
from chiller_fault_detection.models.trainer import TrainConfig, load_window_splits


def _class_id(name: str) -> int:
    try:
        return LABEL_TO_ID[FaultType(name)]
    except ValueError as exc:
        valid = ", ".join(ft.value for ft in FaultType)
        raise SystemExit(f"Unknown class {name!r}. Use one of: {valid}") from exc


def _unscale(windows: np.ndarray, scaler) -> np.ndarray:
    n, channels, steps = windows.shape
    flat = windows.transpose(0, 2, 1).reshape(-1, channels)
    restored = scaler.inverse_transform(flat).astype(np.float32)
    return restored.reshape(n, steps, channels).transpose(0, 2, 1)


def channel_mean_std(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std per channel over every timestep in the windows."""
    stacked = windows.transpose(0, 2, 1).reshape(-1, windows.shape[1])
    return stacked.mean(axis=0), stacked.std(axis=0)


def window_vectors(windows: np.ndarray) -> np.ndarray:
    """One 19-d point per window: mean over the 128 timesteps."""
    return windows.mean(axis=2)


def confused_windows(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    true_id: int,
    pred_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_a = x[y_true == true_id]
    true_b = x[y_true == pred_id]
    a_to_b = x[(y_true == true_id) & (y_pred == pred_id)]
    return true_a, a_to_b, true_b


def plot_mean_std(
    groups: list[tuple[str, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    n_ch = len(MODEL_FEATURE_COLUMNS)
    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), sharex=False)
    colors = ("#4c72b0", "#c44e52", "#55a868")
    x_pos = (0.0, 1.0, 2.0)
    for i, name in enumerate(MODEL_FEATURE_COLUMNS):
        ax = axes.flat[i]
        for (label, windows), color, xpos in zip(groups, colors, x_pos):
            if len(windows) == 0:
                continue
            mean, std = channel_mean_std(windows)
            ax.errorbar(
                xpos,
                mean[i],
                yerr=std[i],
                fmt="o",
                color=color,
                capsize=4,
                label=label if i == 0 else None,
            )
        ax.set_title(name, fontsize=9)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(["A", "A→B", "B"], fontsize=8)
        ax.axhline(0.0, color="0.7", linewidth=0.8, linestyle="--")
    for ax in axes.flat[n_ch:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _channel_index(name: str) -> int:
    try:
        return MODEL_FEATURE_COLUMNS.index(name)
    except ValueError as exc:
        valid = ", ".join(MODEL_FEATURE_COLUMNS)
        raise SystemExit(f"Unknown channel {name!r}. Use one of: {valid}") from exc


def plot_scatter(
    groups: list[tuple[str, np.ndarray]],
    x_name: str,
    y_name: str,
    out_path: Path,
    title: str,
) -> None:
    xi, yi = _channel_index(x_name), _channel_index(y_name)
    colors = ("#4c72b0", "#c44e52", "#55a868")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (label, windows), color in zip(groups, colors):
        if len(windows) == 0:
            continue
        vec = window_vectors(windows)
        ax.scatter(vec[:, xi], vec[:, yi], s=28, alpha=0.75, c=color, label=label, edgecolors="none")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca(
    groups: list[tuple[str, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    chunks = [window_vectors(windows) for _, windows in groups if len(windows)]
    fitted = PCA(n_components=2, random_state=0).fit(np.concatenate(chunks, axis=0))
    colors = ("#4c72b0", "#c44e52", "#55a868")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (label, windows), color in zip(groups, colors):
        if len(windows) == 0:
            continue
        xy = fitted.transform(window_vectors(windows))
        ax.scatter(xy[:, 0], xy[:, 1], s=28, alpha=0.75, c=color, label=label, edgecolors="none")
    ax.set_xlabel(f"PC1 ({fitted.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({fitted.explained_variance_ratio_[1]:.0%})")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--true", required=True, help="True class A, e.g. normal")
    parser.add_argument("--pred", required=True, help="Predicted class B, e.g. rl")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=PROCESSED_DIR / "ca_tcc_residual",
    )
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="Keep StandardScaler units (what the model sees). Default is inverse-transform.",
    )
    parser.add_argument(
        "--scatter",
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Scatter two channels, e.g. --scatter TRC TCA",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Scatter the first two principal components of all 19 channels.",
    )
    args = parser.parse_args()

    true_id = _class_id(args.true)
    pred_id = _class_id(args.pred)
    y_true = np.load(args.exp_dir / f"{args.split}_y_true.npy")
    y_pred = np.load(args.exp_dir / f"{args.split}_y_pred.npy")

    config = TrainConfig(window=args.window, stride_windows=args.stride)
    x, y_from_frame = load_window_splits(config)[args.split][:2]
    if len(y_from_frame) != len(y_true) or not np.array_equal(y_from_frame, y_true):
        raise SystemExit(
            "Window order does not match saved y_true. "
            "Use the same --window/--stride as training."
        )

    if not args.scaled:
        x = _unscale(x, joblib.load(SCALER_PATH))

    true_a, a_to_b, true_b = confused_windows(x, y_true, y_pred, true_id, pred_id)
    print(
        f"{args.split}: {len(a_to_b)} windows true={args.true} predicted={args.pred} "
        f"(true {args.true}={len(true_a)}, true {args.pred}={len(true_b)})"
    )
    if len(a_to_b) == 0:
        raise SystemExit("No windows in that confusion cell.")

    groups = [
        (f"true {args.true} (n={len(true_a)})", true_a),
        (f"{args.true}→{args.pred} (n={len(a_to_b)})", a_to_b),
        (f"true {args.pred} (n={len(true_b)})", true_b),
    ]
    stem = f"{args.split}_confused_{args.true}_to_{args.pred}"
    if args.scatter is None and not args.pca:
        out_path = args.exp_dir / f"{stem}.png"
        plot_mean_std(groups, out_path, f"{args.split}: mean±std of {args.true} → {args.pred} windows")
        print(f"Wrote {out_path}")
        return
    if args.scatter is not None:
        x_name, y_name = args.scatter
        out_path = args.exp_dir / f"{stem}_scatter_{x_name}_{y_name}.png"
        plot_scatter(
            groups,
            x_name,
            y_name,
            out_path,
            f"{args.split}: {args.true} → {args.pred}  ({x_name} vs {y_name})",
        )
        print(f"Wrote {out_path}")
    if args.pca:
        out_path = args.exp_dir / f"{stem}_pca.png"
        plot_pca(groups, out_path, f"{args.split}: {args.true} → {args.pred}  (PCA of 19 channels)")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
