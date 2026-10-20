"""Merge a run's CSV logs across resumes and plot its training curves.

    python tools/training_curves.py models/diep_fold1/training

A resumed run writes a new ``logs/version_N/metrics.csv`` and restarts from the best epoch
so far, so consecutive versions overlap. The later version wins for any epoch both cover,
which is what the model actually went on to use. Writes ``metrics_merged.csv`` (one row per
epoch, train and val side by side) and ``curves.png`` into the same directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Categorical slots 1-2 of the reference palette (validated as an adjacent pair), on its
# light chart surface with its text inks.
TRAIN, VAL = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [  # (column suffix, title, scale to display units)
    ("Total_Loss", "Total loss (L1, E + F + 0.1 S)", 1.0),
    ("Energy_MAE", "Energy MAE (meV/atom)", 1000.0),
    ("Force_MAE", "Force MAE (eV/Å)", 1.0),
    ("Stress_MAE", "Stress MAE (GPa)", 1.0),
]


def merge_logs(training_dir: Path) -> pd.DataFrame:
    frames = []
    for version_dir in sorted((training_dir / "logs").glob("version_*"),
                              key=lambda p: int(p.name.split("_")[1])):
        df = pd.read_csv(version_dir / "metrics.csv")
        df["version"] = int(version_dir.name.split("_")[1])
        frames.append(df)
    if not frames:
        raise SystemExit(f"no logs/version_*/metrics.csv under {training_dir}")
    raw = pd.concat(frames, ignore_index=True)
    cols = [c for c in raw.columns if c.startswith(("train_", "val_"))]
    rows = raw[raw[cols].notna().any(axis=1)]  # drops the test row
    # Train and val land on separate rows; collapse each (version, epoch) to one row, then
    # keep the latest version of each epoch.
    per = rows.groupby(["version", "epoch"])[cols].first().reset_index()
    merged = per.sort_values(["epoch", "version"]).groupby("epoch").last().reset_index()
    return merged[["epoch", "version"] + cols]


def plot(merged: pd.DataFrame, out: Path, best_epoch: int | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, facecolor=SURFACE)
    for ax, (key, title, scale) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.plot(merged["epoch"], merged[f"train_{key}"] * scale, color=TRAIN, lw=1.5, label="train")
        ax.plot(merged["epoch"], merged[f"val_{key}"] * scale, color=VAL, lw=1.5, label="validation")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0, 7.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=INK_2, lw=0.8, ls=(0, (3, 3)))
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    if best_epoch is not None:
        axes[0, 0].annotate(f"best epoch {best_epoch}", xy=(best_epoch, 1), xycoords=("data", "axes fraction"),
                            xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                            fontsize=8, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle("diep_fold1 training curves (log scale)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("training_dir", type=Path, help="directory holding logs/version_*/metrics.csv")
    p.add_argument("--best-epoch", type=int, default=None,
                   help="marked on the plot; read from ../PROVENANCE.json when omitted")
    args = p.parse_args(argv)

    best = args.best_epoch
    prov = args.training_dir.parent / "PROVENANCE.json"
    if best is None and prov.exists():
        import json

        best = json.loads(prov.read_text()).get("epoch")
    merged = merge_logs(args.training_dir)
    merged.to_csv(args.training_dir / "metrics_merged.csv", index=False)
    plot(merged, args.training_dir / "curves.png", best)
    print(f"{len(merged)} epochs ({merged.epoch.min()}-{merged.epoch.max()}) from versions "
          f"{sorted(merged.version.unique().tolist())} -> {args.training_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
