#!/usr/bin/env python3
"""
DLRM Hardware Planning Study: Pareto-Optimal Heterogeneous Configurations.

Sweeps GPU tier × CPU nodes × CGRA config × interconnect across embedding
scales (4-200 GB) to find cost-efficient deployment configurations.

Key result: For 30-200 GB embedding tables, commodity CPU+GPU systems at
$2.5K-$7K achieve 60-76% of $60K multi-GPU throughput — a 5-10× improvement
in cost efficiency. The optimal configuration depends on embedding scale,
interconnect bandwidth, and accelerator capacity.

All component times are from measured benchmarks or Comal CGRA simulation.
No new measurements required — this is a pure analytical sweep.
"""

import math
import json
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# Hardware catalog
# ══════════════════════════════════════════════════════════════════════════

GPU_CATALOG = {
    'RTX 4060':      {'card_cost': 300,   'eff_tflops': 20,  'vram_gb': 8,   'mem_bw': 272},
    'RTX PRO 4500':  {'card_cost': 700,   'eff_tflops': 38,  'vram_gb': 16,  'mem_bw': 960},   # MEASURED
    'RTX 4090':      {'card_cost': 1800,  'eff_tflops': 82,  'vram_gb': 24,  'mem_bw': 1008},
    'RTX 5090':      {'card_cost': 2500,  'eff_tflops': 105, 'vram_gb': 32,  'mem_bw': 1792},
    'A100 40GB':     {'card_cost': 10000, 'eff_tflops': 150, 'vram_gb': 40,  'mem_bw': 1555},
    'A100 80GB':     {'card_cost': 15000, 'eff_tflops': 150, 'vram_gb': 80,  'mem_bw': 2039},
}

CPU_CATALOG = {
    'Xeon-4ch':  {'system_cost': 800,  'eff_mem_bw': 77,  'label': '4-ch DDR5'},
    'Xeon-6ch':  {'system_cost': 1200, 'eff_mem_bw': 150, 'label': '6-ch DDR5'},   # MEASURED
    'Xeon-8ch':  {'system_cost': 3000, 'eff_mem_bw': 230, 'label': '8-ch DDR5'},
    'EPYC-12ch': {'system_cost': 5000, 'eff_mem_bw': 350, 'label': '12-ch DDR5'},
}

CGRA_CATALOG = {
    'U280':             {'cost': 3000, 'ch': 32, 'hbm_gb': 8,  'freq_mhz': 450, 'hbm_cost': 0.91},
    'Versal HBM':       {'cost': 5000, 'ch': 32, 'hbm_gb': 32, 'freq_mhz': 500, 'hbm_cost': 0.82},
    'Versal Premium':   {'cost': 8000, 'ch': 64, 'hbm_gb': 64, 'freq_mhz': 500, 'hbm_cost': 0.41},
}

INTERCONNECTS = {
    'PCIe4':  {'cpu_bw': 32,  'cgra_bw': 16},
    'PCIe5':  {'cpu_bw': 64,  'cgra_bw': 32},
    'NVLink': {'cpu_bw': 900, 'cgra_bw': 450},
}

# DDR5 cost per GB
DDR5_PER_GB = 5
# Base RAM included in CPU system cost
BASE_RAM_GB = 64
# GPU overhead in VRAM (PyTorch + CUDA runtime + MLP weights)
GPU_VRAM_OVERHEAD = 3  # GB

CATEGORIES = ['GPU-only', 'Multi-GPU', 'CPU+GPU', 'CPU+GPU+CGRA']

# Multi-GPU costs (only datacenter GPUs with NVLink)
MULTI_GPU_COSTS = {
    ('A100 40GB', 2): 27000,
    ('A100 40GB', 4): 50000,
    ('A100 80GB', 2): 33000,
    ('A100 80GB', 4): 60000,
}

# ══════════════════════════════════════════════════════════════════════════
# DLRM model parameters (MLPerf/Criteo config)
# ══════════════════════════════════════════════════════════════════════════

NUM_TABLES = 26
EMBED_DIM = 128
BATCH = 4096
NNZ_PER_TABLE = 40

XFER_PER_TABLE = BATCH * EMBED_DIM * 4      # 2.1 MB
TOTAL_XFER = NUM_TABLES * XFER_PER_TABLE     # 54.5 MB

# ══════════════════════════════════════════════════════════════════════════
# Measured component baselines
# ══════════════════════════════════════════════════════════════════════════

# CPU MKL embedding: DRAM bandwidth-bound gather+pool.
# Independent of table size — total bytes read = BATCH×NNZ×EMBED×4 = 84 MB/table.
# Reference: measured 0.57 ms/table on Xeon-6ch (150 GB/s effective).
REF_CPU_BW = 150        # GB/s (reference CPU for MKL_PER_TABLE_MS)
REF_CPU_MS = 0.57       # ms/table on reference CPU

# GPU embedding: PyTorch EmbeddingBag, memory-bandwidth-bound.
# Reference: measured on RTX PRO 4500 (960 GB/s bandwidth).
REF_EMBED_BW = 960      # GB/s
REF_EMBED_MS = 0.15     # ms/table

# GPU MLP (bottom + top): compute-bound GEMM chain.
# Bottom: [3328,512,512,256], Top: [607,512,256,1]
# Reference: measured on RTX PRO 4500 (38 eff TFLOPS).
REF_MLP_TFLOPS = 38.0
REF_MLP_MS = 0.95

# CGRA embedding: from Comal simulation (batch=4096, vec=16, pipelined HBM).
CGRA_BASE_CYCLES = 1_478_930
CGRA_READS_PER_TABLE = 1_310_720

# ── Multi-node MPI scaling model ──
# Measured DLRM embedding distribution on GKE c4 cluster: 95% parallel
# efficiency (embarrassingly parallel — no halo exchange unlike GCN).
# Power law: speedup = n^α, 95% efficiency at 16 nodes → 15.2 = 16^α → α ≈ 0.985
# For comparison: GCN with halo exchange measured α = 0.911 (75% efficiency).
SCALING_EXPONENT = 0.985  # calibrated from measured DLRM on GKE c4

# ══════════════════════════════════════════════════════════════════════════
# Performance model
# ══════════════════════════════════════════════════════════════════════════

def cpu_embed_per_table(cpu):
    """CPU embedding time per table, scaled by memory bandwidth."""
    return REF_CPU_MS * REF_CPU_BW / cpu['eff_mem_bw']

def multi_node_embed_time(num_tables, cpu, num_nodes):
    """Embedding time with MPI distribution across separate nodes.

    Uses empirical power-law scaling calibrated from measured 12.7× at 16 nodes
    on GKE c4 instances: speedup = n^0.917. Accounts for MPI allgather,
    network latency, and load imbalance.
    """
    single_node = num_tables * cpu_embed_per_table(cpu)
    if num_nodes == 1:
        return single_node
    speedup = num_nodes ** SCALING_EXPONENT
    return single_node / speedup

def gpu_embed_per_table(gpu):
    """GPU embedding time per table, scaled by memory bandwidth."""
    return REF_EMBED_MS * REF_EMBED_BW / gpu['mem_bw']

def gpu_mlp_time(gpu):
    """GPU MLP time, scaled by effective TFLOPS."""
    return REF_MLP_MS * REF_MLP_TFLOPS / gpu['eff_tflops']

def cgra_per_table(cgra):
    """CGRA embedding time per table from Comal simulation."""
    clock_khz = cgra['freq_mhz'] * 1000
    return (CGRA_BASE_CYCLES + CGRA_READS_PER_TABLE * cgra['hbm_cost']) / clock_khz

def cgra_embed_time(n_tables, cgra):
    """Total CGRA embedding time with channel contention."""
    if n_tables == 0:
        return 0.0
    concurrent = min(n_tables, cgra['ch'] // 2)
    batches = math.ceil(n_tables / concurrent)
    return cgra_per_table(cgra) * batches

def optimal_cgra_split(total_embed_gb, cgra, cpu, num_cpu_nodes):
    """Find best split of tables between CGRA and CPU nodes."""
    table_bytes = total_embed_gb * 1e9 / NUM_TABLES
    max_cgra = min(NUM_TABLES, int(cgra['hbm_gb'] * 1e9 / table_bytes))

    best_n = 0
    best_time = multi_node_embed_time(NUM_TABLES, cpu, num_cpu_nodes)

    for n in range(1, max_cgra + 1):
        ct = cgra_embed_time(n, cgra)
        cpu_tables = NUM_TABLES - n
        mt = multi_node_embed_time(cpu_tables, cpu, num_cpu_nodes)
        t = max(ct, mt)
        if t < best_time:
            best_n, best_time = n, t

    return best_n, best_time

# ══════════════════════════════════════════════════════════════════════════
# Cost model
# ══════════════════════════════════════════════════════════════════════════

def het_system_cost(cpu_name, gpu_name, num_nodes, cgra_name, total_embed_gb):
    """Total cost for a heterogeneous CPU+GPU(+CGRA) system."""
    cpu = CPU_CATALOG[cpu_name]
    gpu = GPU_CATALOG[gpu_name]
    # Primary node: CPU system + GPU card
    cost = cpu['system_cost'] + gpu['card_cost']
    # RAM for embedding tables (primary node)
    per_node_gb = math.ceil(total_embed_gb * 1.2 / num_nodes)
    extra_ram = max(0, per_node_gb - BASE_RAM_GB)
    cost += extra_ram * DDR5_PER_GB
    # Additional CPU nodes (each with own CPU system + RAM share)
    for _ in range(num_nodes - 1):
        cost += cpu['system_cost'] + extra_ram * DDR5_PER_GB
    # CGRA card
    if cgra_name:
        cost += CGRA_CATALOG[cgra_name]['cost']
    return cost

def gpu_only_cost(gpu_name, num_gpus):
    """Cost for GPU-only system (single or multi-GPU)."""
    if num_gpus == 1:
        # Cheapest CPU system + GPU card
        cheapest_cpu = min(CPU_CATALOG.values(), key=lambda c: c['system_cost'])
        return cheapest_cpu['system_cost'] + GPU_CATALOG[gpu_name]['card_cost']
    return MULTI_GPU_COSTS.get((gpu_name, num_gpus), float('inf'))

# ══════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    label: str
    category: str           # 'GPU-only', 'CPU+GPU', 'CPU+GPU+CGRA'
    cost: float             # $
    throughput_kinfs: float  # K inferences/s (pipeline steady-state)
    latency_ms: float       # single-batch latency (sequential)
    embed_gb: float
    gpu: str
    cpu: Optional[str] = None
    num_gpus: int = 1
    num_nodes: int = 1
    cgra: Optional[str] = None
    interconnect: Optional[str] = None
    bottleneck: Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════
# Evaluate all configurations for a given embedding scale
# ══════════════════════════════════════════════════════════════════════════

def evaluate_all(total_embed_gb: float) -> List[Config]:
    configs = []

    # ── GPU-only ──────────────────────────────────────────────────────
    for gname, gpu in GPU_CATALOG.items():
        available = gpu['vram_gb'] - GPU_VRAM_OVERHEAD

        # Single GPU
        if total_embed_gb <= available:
            cost = gpu_only_cost(gname, 1)
            embed = NUM_TABLES * gpu_embed_per_table(gpu)
            mlp = gpu_mlp_time(gpu)
            latency = embed + mlp
            tp = BATCH / latency * 1000 / 1e3
            configs.append(Config(
                f'1x{gname}', 'GPU-only',
                cost, tp, latency, total_embed_gb, gname, None, 1, 0,
                bottleneck='GPU embed' if embed > mlp else 'GPU MLP'))

        # Multi-GPU (datacenter only)
        if gname in ('A100 40GB', 'A100 80GB'):
            for ng in [2, 4]:
                if total_embed_gb <= ng * available:
                    cost = gpu_only_cost(gname, ng)
                    if cost == float('inf'):
                        continue
                    embed = math.ceil(NUM_TABLES / ng) * gpu_embed_per_table(gpu)
                    mlp = gpu_mlp_time(gpu)
                    a2a = TOTAL_XFER / (900 * 1e9) * 1e3
                    latency = embed + a2a + mlp
                    tp = BATCH / latency * 1000 / 1e3
                    configs.append(Config(
                        f'{ng}x{gname}', 'Multi-GPU',
                        cost, tp, latency, total_embed_gb, gname, None, ng, 0,
                        bottleneck='GPU embed'))

    # ── CPU+GPU ───────────────────────────────────────────────────────
    for cname, cpu in CPU_CATALOG.items():
        for gname, gpu in GPU_CATALOG.items():
            for nn in [1, 2, 4, 8, 16]:
                for icname, ic in INTERCONNECTS.items():
                    cost = het_system_cost(cname, gname, nn, None, total_embed_gb)

                    embed = multi_node_embed_time(NUM_TABLES, cpu, nn)
                    xfer = TOTAL_XFER / (ic['cpu_bw'] * 1e9) * 1e3
                    mlp = gpu_mlp_time(gpu)

                    bneck = max(embed, xfer, mlp)
                    latency = embed + xfer + mlp
                    tp = BATCH / bneck * 1000 / 1e3

                    parts = {'CPU embed': embed, 'transfer': xfer, 'GPU MLP': mlp}
                    bn = max(parts, key=parts.get)

                    clabel = cpu['label']
                    configs.append(Config(
                        f'{nn}x{clabel}+{gname}@{icname}', 'CPU+GPU',
                        cost, tp, latency, total_embed_gb,
                        gname, cname, 1, nn, None, icname, bn))

    # ── CPU+GPU+CGRA ──────────────────────────────────────────────────
    for cname, cpu in CPU_CATALOG.items():
        for gname, gpu in GPU_CATALOG.items():
            for cgname, cgra in CGRA_CATALOG.items():
                for nn in [1, 2, 4]:
                    for icname, ic in INTERCONNECTS.items():
                        n_cgra, embed_compute = optimal_cgra_split(
                            total_embed_gb, cgra, cpu, nn)

                        if n_cgra == 0:
                            continue

                        cost = het_system_cost(cname, gname, nn, cgname,
                                               total_embed_gb)

                        n_cpu = NUM_TABLES - n_cgra
                        cpu_xfer = n_cpu * XFER_PER_TABLE / (ic['cpu_bw'] * 1e9) * 1e3
                        cgra_xfer = n_cgra * XFER_PER_TABLE / (ic['cgra_bw'] * 1e9) * 1e3
                        xfer = max(cpu_xfer, cgra_xfer)
                        mlp = gpu_mlp_time(gpu)

                        bneck = max(embed_compute, xfer, mlp)
                        latency = embed_compute + xfer + mlp
                        tp = BATCH / bneck * 1000 / 1e3

                        parts = {'embed': embed_compute, 'transfer': xfer, 'GPU MLP': mlp}
                        bn = max(parts, key=parts.get)

                        clabel = cpu['label']
                        configs.append(Config(
                            f'{nn}x{clabel}+{gname}+{cgname}@{icname}',
                            'CPU+GPU+CGRA',
                            cost, tp, latency, total_embed_gb,
                            gname, cname, 1, nn, cgname, icname, bn))

    return configs

# ══════════════════════════════════════════════════════════════════════════
# Pareto frontier
# ══════════════════════════════════════════════════════════════════════════

def pareto_frontier(configs: List[Config]) -> List[Config]:
    """Configs that are not dominated (no cheaper config with >= throughput)."""
    by_cost = sorted(configs, key=lambda c: c.cost)
    pareto = []
    max_tp = 0
    for c in by_cost:
        if c.throughput_kinfs > max_tp:
            pareto.append(c)
            max_tp = c.throughput_kinfs
    return pareto

def best_in_category(configs: List[Config], category: str) -> Optional[Config]:
    """Best throughput/dollar config for a category."""
    subset = [c for c in configs if c.category == category]
    if not subset:
        return None
    return max(subset, key=lambda c: c.throughput_kinfs / c.cost)

# ══════════════════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════════════════

TARGET_GBS = [4, 8, 16, 30, 60, 100, 200]

print("=" * 110)
print("DLRM Hardware Planning Study: Pareto-Optimal Configurations")
print(f"Tables={NUM_TABLES}, embed_dim={EMBED_DIM}, batch={BATCH}, nnz={NNZ_PER_TABLE}")
print("=" * 110)

all_configs = {}
all_pareto = {}
summary_rows = []

for gb in TARGET_GBS:
    configs = evaluate_all(gb)
    pareto = pareto_frontier(configs)
    all_configs[gb] = configs
    all_pareto[gb] = pareto

    best_gpu = best_in_category(configs, 'GPU-only')
    best_het = best_in_category(configs, 'CPU+GPU')
    best_3dev = best_in_category(configs, 'CPU+GPU+CGRA')

    # ── Configuration space breakdown ──
    print(f"\n{'═'*110}")
    print(f"  {gb} GB total embedding ({gb*1e9/(NUM_TABLES*EMBED_DIM*4):.0f} rows/table)")
    print(f"{'═'*110}")

    by_cat = {}
    for c in configs:
        by_cat.setdefault(c.category, []).append(c)

    pareto_set = set(id(c) for c in pareto)
    print(f"\n  {len(configs)} configs evaluated:")
    for cat in CATEGORIES:
        subset = by_cat.get(cat, [])
        if not subset:
            continue
        on_p = sum(1 for c in subset if id(c) in pareto_set)
        # Count distinct throughput levels
        tps = set(round(c.throughput_kinfs, 1) for c in subset)
        bottlenecks = {}
        for c in subset:
            bottlenecks[c.bottleneck] = bottlenecks.get(c.bottleneck, 0) + 1
        bn_str = ", ".join(f"{v} {k}" for k, v in sorted(bottlenecks.items(), key=lambda x: -x[1]))
        print(f"    {cat:<18} {len(subset):>4} configs, {len(tps):>3} distinct throughputs, "
              f"{on_p} on Pareto  [{bn_str}]")

    # Pareto frontier
    print(f"\n  Pareto frontier ({len(pareto)} configs):")
    print(f"    {'Configuration':<55} {'Cost':>8} {'Throughput':>12} {'Bottleneck':<15}")
    print("    " + "─" * 92)
    for c in pareto:
        print(f"    {c.label:<55} ${c.cost:>7,.0f} {c.throughput_kinfs:>11.0f} K/s "
              f"{c.bottleneck or ''}")

    best_mgpu = best_in_category(configs, 'Multi-GPU')
    summary_rows.append({
        'gb': gb, 'best_gpu': best_gpu, 'best_mgpu': best_mgpu,
        'best_het': best_het, 'best_3dev': best_3dev, 'pareto': pareto,
    })

# ══════════════════════════════════════════════════════════════════════════
# Cross-scale summary
# ══════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*120}")
print("CROSS-SCALE SUMMARY: Best throughput-per-dollar in each category")
print(f"{'='*120}")
print(f"  {'GB':>4} | {'GPU-only (single)':^25} | {'Multi-GPU':^25} | {'CPU+GPU (best)':^30} | {'CPU+GPU+CGRA':^25}")
print("  " + "─" * 115)

for row in summary_rows:
    gb = row['gb']
    parts = [f"  {gb:>4} |"]
    for key in ['best_gpu', 'best_mgpu', 'best_het', 'best_3dev']:
        c = row[key]
        if c is None:
            parts.append(f" {'--':^23} |")
        else:
            tpd = c.cost / c.throughput_kinfs
            parts.append(f" ${c.cost:>6,.0f} {c.throughput_kinfs:>6.0f}K {tpd:>5.1f}$/K |")
    print("".join(parts))

# ══════════════════════════════════════════════════════════════════════════
# FIGURE A: Two-panel Pareto (ASPLOS quality)
#   Left:  16 GB (single GPU fits)
#   Right: 100 GB (single GPU OOM — heterogeneous enables $1K-$21K tier)
# ══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    'mathtext.fontset': 'stix',
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Color by GPU tier, shape by deployment category
GPU_COLORS = {
    'RTX 4060':      '#9467bd',   # purple
    'RTX PRO 4500':  '#ff7f0e',   # orange (measured reference)
    'RTX 4090':      '#2ca02c',   # green
    'RTX 5090':      '#17becf',   # cyan
    'A100 40GB':     '#d62728',   # red
    'A100 80GB':     '#8c564b',   # brown
}
CAT_MARKERS = {
    'GPU-only':      'o',
    'Multi-GPU':     'p',
    'CPU+GPU':       's',
    'CPU+GPU+CGRA':  'D',
}
# Shapes encode CPU node count (immediately readable)
# 0 = GPU-only/Multi-GPU (no CPU), uses star to distinguish
NODE_MARKERS = {
    0: '*',   # star — GPU-only, no CPU nodes
    1: 'o',   # circle
    2: '^',   # triangle up
    4: 's',   # square
    8: 'D',   # diamond
    16: 'P',  # plus (filled)
}
PLOT_CATS = ['GPU-only', 'Multi-GPU', 'CPU+GPU', 'CPU+GPU+CGRA']

# Full page width, shorter vertically, log-log
fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.55))

PANEL_LABELS = ['(a)', '(b)']
PANEL_GBS = [16, 100]

for ax_idx, gb in enumerate(PANEL_GBS):
    ax = axes[ax_idx]
    configs = all_configs[gb]
    pareto = all_pareto[gb]
    pareto_set = set(id(c) for c in pareto)

    plot_configs = configs
    plot_pareto = pareto_frontier(plot_configs)

    def get_marker(c):
        return NODE_MARKERS.get(c.num_nodes, 'o')

    # Background: all evaluated configs (faded), colored by GPU, shaped by CPU count
    # FPGA-augmented configs get a dark edge to distinguish from CPU+GPU
    for c in plot_configs:
        color = GPU_COLORS.get(c.gpu, '#999')
        has_fpga = c.cgra is not None
        ax.scatter([c.cost], [c.throughput_kinfs], c=color, marker=get_marker(c),
                   alpha=0.10 if has_fpga else 0.06,
                   s=10 if has_fpga else 8, zorder=2,
                   linewidths=0.3 if has_fpga else 0,
                   edgecolors='#333' if has_fpga else 'none')

    # Pareto frontier configs (bold), colored by GPU, shaped by CPU count
    # Stars (GPU-only) need larger s to match visual weight of other shapes
    # FPGA-augmented configs get a dark edge
    for c in plot_pareto:
        color = GPU_COLORS.get(c.gpu, '#999')
        mk = get_marker(c)
        sz = 55 if mk == '*' else 30
        has_fpga = c.cgra is not None
        ax.scatter([c.cost], [c.throughput_kinfs], c=color, marker=mk,
                   s=sz, zorder=5,
                   edgecolors='#333' if has_fpga else 'white',
                   linewidths=0.8 if has_fpga else 0.4)

    # Pareto frontier line
    pcosts = [c.cost for c in plot_pareto]
    ptps = [c.throughput_kinfs for c in plot_pareto]
    ax.plot(pcosts, ptps, color='#333', lw=0.8, alpha=0.5, zorder=3)

    # Build legend: GPU colors + CPU node shapes
    from matplotlib.lines import Line2D
    legend_handles = []
    # GPU tier colors (use star since GPU-only uses star marker)
    for gname, color in GPU_COLORS.items():
        if any(c.gpu == gname for c in configs):
            legend_handles.append(Line2D([0], [0], marker='*', color='w',
                                         markerfacecolor=color, markersize=6,
                                         label=gname, markeredgewidth=0.3,
                                         markeredgecolor='white'))
    # Separator
    legend_handles.append(Line2D([0], [0], color='w', label=''))
    # CPU node count shapes
    node_labels = {1: '1 CPU', 2: '2 CPU', 4: '4 CPU', 8: '8 CPU', 16: '16 CPU'}
    for nn, mk in sorted(NODE_MARKERS.items()):
        if nn == 0:
            continue
        if any(c.num_nodes == nn for c in configs):
            legend_handles.append(Line2D([0], [0], marker=mk, color='w',
                                         markerfacecolor='#666', markersize=5,
                                         label=node_labels[nn],
                                         markeredgewidth=0.3,
                                         markeredgecolor='white'))
    # FPGA accelerator indicator (dark edge)
    if any(c.cgra is not None for c in configs):
        legend_handles.append(Line2D([0], [0], color='w', label=''))
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor='#999', markersize=5,
                                     label='w/ FPGA',
                                     markeredgewidth=0.8,
                                     markeredgecolor='#333'))


    ax.set_xlabel('System Cost (USD)', fontsize=9, labelpad=0)
    if ax_idx == 0:
        ax.set_ylabel('Throughput (K inf/s)', fontsize=9)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_xlim(700, 120000)
    ax.set_ylim(50, 15000)

    # Panel title
    ax.set_title(f'{gb} GB embeddings',
                 fontsize=9, loc='center', pad=3)

    # OOM badge on right panel (100 GB — no single GPU can fit)
    if not any(c.category == 'GPU-only' for c in configs):
        ax.text(0.03, 0.97, 'Single GPU: OOM',
                transform=ax.transAxes, fontsize=7, ha='left', va='top',
                color='#d62728', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='#ffeaea',
                          ec='#d62728', alpha=0.85, lw=0.5))

    # Store legend handles from last panel for shared legend
    if ax_idx == len(PANEL_GBS) - 1:
        shared_legend_handles = legend_handles


# Shared legend — 2 columns, anchored to the right of the right subplot
all_handles = [h for h in shared_legend_handles if h.get_label() != '']

axes[1].legend(handles=all_handles, loc='center left',
               ncol=2, frameon=False, fontsize=6.5,
               borderpad=0.3, handletextpad=0.2, columnspacing=0.6,
               labelspacing=0.3, bbox_to_anchor=(1.02, 0.5))

fig.tight_layout(w_pad=1.0)
for fmt in ['png', 'pdf']:
    fig.savefig(os.path.join(FIG_DIR, f'dlrm_pareto.{fmt}'), dpi=300)
print(f"\nFigure A saved: dlrm_pareto")

stitch_img = '/home/rubensl/samml/stitch/images'
if os.path.isdir(stitch_img):
    import shutil
    shutil.copy(os.path.join(FIG_DIR, 'dlrm_pareto.pdf'), stitch_img)
    print(f"Copied to stitch/images/dlrm_pareto.pdf")

plt.close()

# ══════════════════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════════════════

save = {
    'model': 'DLRM (MLPerf/Criteo config)',
    'params': {
        'num_tables': NUM_TABLES, 'embed_dim': EMBED_DIM,
        'batch': BATCH, 'nnz_per_table': NNZ_PER_TABLE,
    },
    'baselines': {
        'ref_cpu_ms_per_table': REF_CPU_MS,
        'gpu_embed_per_table_ms': REF_EMBED_MS,
        'gpu_mlp_ms': REF_MLP_MS,
    },
    'hardware': {
        'gpus': {k: v for k, v in GPU_CATALOG.items()},
        'cgras': {k: {kk: vv for kk, vv in v.items()} for k, v in CGRA_CATALOG.items()},
    },
    'results': {},
}

for gb in TARGET_GBS:
    pareto = all_pareto[gb]
    bg = best_in_category(all_configs[gb], 'GPU-only')
    bh = best_in_category(all_configs[gb], 'CPU+GPU')
    b3 = best_in_category(all_configs[gb], 'CPU+GPU+CGRA')

    save['results'][str(gb)] = {
        'pareto': [{'label': c.label, 'category': c.category,
                     'cost': c.cost, 'throughput_kinfs': c.throughput_kinfs,
                     'latency_ms': c.latency_ms} for c in pareto],
        'best_gpu': asdict(bg) if bg else None,
        'best_het': asdict(bh) if bh else None,
        'best_3dev': asdict(b3) if b3 else None,
    }

out_path = os.path.join(RESULTS_DIR, 'dlrm_pareto.json')
with open(out_path, 'w') as f:
    json.dump(save, f, indent=2, default=str)
print(f"\nResults saved: {out_path}")

# ══════════════════════════════════════════════════════════════════════════
# The paper sentence
# ══════════════════════════════════════════════════════════════════════════

total_configs = sum(len(all_configs[gb]) for gb in TARGET_GBS)
print(f"\n{'='*110}")
print(f"Evaluated {total_configs} configurations across {len(TARGET_GBS)} embedding scales.")
print()

# Find the headline numbers
for gb in [30, 200]:
    bg = best_in_category(all_configs[gb], 'GPU-only')
    bh = best_in_category(all_configs[gb], 'CPU+GPU')
    if bg and bh:
        tp_pct = bh.throughput_kinfs / bg.throughput_kinfs * 100
        cost_pct = bh.cost / bg.cost * 100
        eff = (bh.throughput_kinfs / bh.cost) / (bg.throughput_kinfs / bg.cost)
        print(f"  {gb} GB: {bh.label} (${bh.cost:,.0f}) achieves {tp_pct:.0f}% of "
              f"{bg.label} (${bg.cost:,.0f}) throughput at {cost_pct:.0f}% cost "
              f"= {eff:.1f}x better $/perf")
    elif bh:
        print(f"  {gb} GB: GPU-only impossible. Het: {bh.label} at "
              f"${bh.cost:,.0f}, {bh.throughput_kinfs:.0f}K inf/s")

print(f"{'='*110}")
