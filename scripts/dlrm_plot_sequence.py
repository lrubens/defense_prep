#!/usr/bin/env python3
"""Generate five-frame DLRM cost-throughput builds for talk slides.

The performance and cost model comes from samml's dlrm_pareto.py. This
script loads only that file's model definitions, without running its plotting
or file-writing code, and emits matched 16 GB and 100 GB slide sequences.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(__file__).resolve().with_name("dlrm_pareto.py")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots" / "dlrm"

FRAME_NAMES = {
    1: "axes",
    2: "anchor",
    3: "fixed_family",
    4: "stitch_space",
    5: "pareto",
}

CATEGORY_STYLE = {
    "GPU-only": {"color": "#7B2CBF", "marker": "*", "label": "GPU only"},
    "Multi-GPU": {"color": "#D55E00", "marker": "P", "label": "Multi-GPU"},
    "CPU+GPU": {"color": "#3A923A", "marker": "s", "label": "CPU + GPU"},
    "CPU+GPU+CGRA": {
        "color": "#2878B5",
        "marker": "D",
        "label": "CPU + GPU + accelerator",
    },
}

ANCHOR_COLOR = "#F28E2B"
FAMILY_COLOR = "#2878B5"
FRONTIER_COLOR = "#202124"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _is_target_gbs_assignment(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign):
        return False
    return any(
        isinstance(target, ast.Name) and target.id == "TARGET_GBS"
        for target in node.targets
    )


def _is_makedirs_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "os"
        and function.attr == "makedirs"
    )


def load_model(source_path: Path) -> dict:
    """Load model definitions while skipping the source script's main sweep."""
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    definitions = []
    for node in tree.body:
        if _is_target_gbs_assignment(node):
            break
        if _is_makedirs_call(node):
            continue
        definitions.append(node)

    module = ast.Module(body=definitions, type_ignores=[])
    namespace = {"__file__": str(source_path)}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def select_anchor(configs):
    """Select the same modeled hardware configuration at each embedding size."""
    matches = [
        config
        for config in configs
        if config.cpu == "Xeon-6ch"
        and config.gpu == "RTX PRO 4500"
        and config.cgra == "U280"
        and config.interconnect == "PCIe4"
        and config.num_nodes == 1
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one fixed-mapping anchor, found {len(matches)}")
    return matches[0]


def configure_axes(ax, embedding_gb: int) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(700, 120_000)
    ax.set_ylim(50, 15_000)
    ax.set_xlabel("System cost (USD)", fontsize=15, labelpad=7)
    ax.set_ylabel("Throughput (K inferences/s)", fontsize=15, labelpad=8)
    ax.set_title(f"{embedding_gb} GB embeddings", fontsize=18, pad=12)
    ax.tick_params(axis="both", which="major", labelsize=12, length=5, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.5)
    ax.grid(True, which="major", color="#B8BDC5", alpha=0.35, linewidth=0.7)
    ax.grid(True, which="minor", color="#D9DCE1", alpha=0.18, linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_color("#4A4A4A")
        spine.set_linewidth(0.8)


def plot_anchor(ax, anchor, embedding_gb: int, annotate: bool = True) -> None:
    ax.scatter(
        [anchor.cost],
        [anchor.throughput_kinfs],
        color=ANCHOR_COLOR,
        marker="o",
        s=155,
        edgecolor="white",
        linewidth=1.8,
        zorder=12,
    )
    ax.scatter(
        [anchor.cost],
        [anchor.throughput_kinfs],
        facecolor="none",
        marker="o",
        s=185,
        edgecolor="#8A4A00",
        linewidth=1.2,
        zorder=13,
    )
    if not annotate:
        return

    if embedding_gb == 16:
        text_position = (0.43, 0.24)
    else:
        text_position = (0.45, 0.17)
    label = (
        "FleetRec-style proxy\n"
        "fixed CPU + GPU + U280 mapping\n"
        f"${anchor.cost / 1000:.2f}K, {anchor.throughput_kinfs:.0f} K/s"
    )
    ax.annotate(
        label,
        xy=(anchor.cost, anchor.throughput_kinfs),
        xycoords="data",
        xytext=text_position,
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#5A3100",
        arrowprops={
            "arrowstyle": "->",
            "color":"#8A4A00",
            "lw": 1.3,
            "shrinkA": 4,
            "shrinkB": 7,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#FFF4E6",
            "edgecolor": "#D8872B",
            "linewidth": 0.9,
        },
        zorder=20,
    )


def plot_fixed_family(ax, fixed_pareto) -> None:
    costs = [config.cost for config in fixed_pareto]
    throughputs = [config.throughput_kinfs for config in fixed_pareto]
    ax.plot(
        costs,
        throughputs,
        color=FAMILY_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=2.0,
        alpha=0.9,
        zorder=7,
    )
    ax.scatter(
        costs,
        throughputs,
        color=FAMILY_COLOR,
        marker="D",
        s=54,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )


def plot_search_space(ax, configs) -> None:
    for category, style in CATEGORY_STYLE.items():
        selected = [config for config in configs if config.category == category]
        if not selected:
            continue
        size = 30 if category == "GPU-only" else 15
        ax.scatter(
            [config.cost for config in selected],
            [config.throughput_kinfs for config in selected],
            color=style["color"],
            marker=style["marker"],
            s=size,
            alpha=0.13,
            edgecolor="none",
            zorder=2,
        )


def plot_overall_frontier(ax, overall_pareto) -> None:
    costs = [config.cost for config in overall_pareto]
    throughputs = [config.throughput_kinfs for config in overall_pareto]
    ax.plot(
        costs,
        throughputs,
        color=FRONTIER_COLOR,
        linewidth=2.4,
        zorder=9,
    )
    for config in overall_pareto:
        style = CATEGORY_STYLE[config.category]
        size = 115 if config.category == "GPU-only" else 72
        ax.scatter(
            [config.cost],
            [config.throughput_kinfs],
            color=style["color"],
            marker=style["marker"],
            s=size,
            edgecolor="white",
            linewidth=1.0,
            zorder=10,
        )


def add_oom_badge(ax) -> None:
    ax.text(
        0.025,
        0.965,
        "Single GPU: OOM",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color="#B42318",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "#FEE4E2",
            "edgecolor": "#D92D20",
            "linewidth": 0.9,
        },
        zorder=30,
    )


def add_legend(ax, include_frontier: bool) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markersize=8,
            label=style["label"],
        )
        for style in CATEGORY_STYLE.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=FAMILY_COLOR,
            linestyle=(0, (4, 3)),
            marker="D",
            markerfacecolor=FAMILY_COLOR,
            markeredgecolor="white",
            linewidth=1.7,
            markersize=6,
            label="Fixed-decomposition envelope",
        )
    )
    if include_frontier:
        handles.append(
            Line2D(
                [0],
                [0],
                color=FRONTIER_COLOR,
                linewidth=2.4,
                label="Overall Pareto frontier",
            )
        )
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#CFD3D8",
        borderpad=0.7,
        labelspacing=0.45,
        handlelength=2.2,
    )


def make_frame(
    embedding_gb: int,
    stage: int,
    configs,
    anchor,
    fixed_pareto,
    overall_pareto,
):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.975, bottom=0.18, top=0.86)
    configure_axes(ax, embedding_gb)

    if stage >= 4:
        plot_search_space(ax, configs)
    if stage >= 3:
        plot_fixed_family(ax, fixed_pareto)
    if stage >= 5:
        plot_overall_frontier(ax, overall_pareto)
    if stage >= 2:
        plot_anchor(ax, anchor, embedding_gb, annotate=stage <= 3)
    if embedding_gb == 100 and stage >= 2:
        add_oom_badge(ax)
    if stage == 4:
        add_legend(ax, include_frontier=False)
    elif stage == 5:
        add_legend(ax, include_frontier=True)

    return fig


def save_configs(path: Path, configs) -> None:
    rows = [asdict(config) for config in configs]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def generate_sequence(model: dict, output_dir: Path, dpi: int) -> dict:
    evaluate_all = model["evaluate_all"]
    pareto_frontier = model["pareto_frontier"]
    metadata = {
        "source": str(DEFAULT_SOURCE),
        "anchor_note": (
            "Modeled FleetRec-style proxy, not the original paper hardware. "
            "The same CPU, GPU, FPGA, and interconnect are selected at both sizes."
        ),
        "sequences": {},
    }

    for embedding_gb in (16, 100):
        configs = evaluate_all(embedding_gb)
        anchor = select_anchor(configs)
        fixed_family = [
            config for config in configs if config.category == "CPU+GPU+CGRA"
        ]
        fixed_pareto = pareto_frontier(fixed_family)
        overall_pareto = pareto_frontier(configs)

        size_dir = output_dir / f"{embedding_gb}gb"
        size_dir.mkdir(parents=True, exist_ok=True)
        save_configs(size_dir / "all_configurations.csv", configs)

        for stage, name in FRAME_NAMES.items():
            figure = make_frame(
                embedding_gb,
                stage,
                configs,
                anchor,
                fixed_pareto,
                overall_pareto,
            )
            stem = size_dir / f"{stage:02d}_{name}"
            figure.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
            figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
            plt.close(figure)

        metadata["sequences"][str(embedding_gb)] = {
            "configuration_count": len(configs),
            "fixed_family_count": len(fixed_family),
            "fixed_family_pareto_count": len(fixed_pareto),
            "overall_pareto_count": len(overall_pareto),
            "anchor": asdict(anchor),
            "fixed_family_pareto": [asdict(config) for config in fixed_pareto],
            "overall_pareto": [asdict(config) for config in overall_pareto],
        }

    return metadata


def main() -> None:
    args = parse_args()
    model = load_model(args.source)
    metadata = generate_sequence(model, args.output_dir, args.dpi)
    resolved_source = args.source.resolve()
    try:
        metadata["source"] = resolved_source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        metadata["source"] = str(resolved_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sequence_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Generated 10 plot frames under {args.output_dir}")


if __name__ == "__main__":
    main()
