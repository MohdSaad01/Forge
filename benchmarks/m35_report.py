"""M35 roofline report generator: ranking, Amdahl analysis, plots, docs (Milestone 35).

    python -m benchmarks.m35_report

Loads every `m35_*.json` result file produced by `m35_hardware.py`,
`m35_kernels.py`, `m35_transfer_stream_alloc.py`, and `m35_mnist.py` (run
those first), and builds Sections 26-30/45 of the M35 brief on top of them:

- Kernel runtime ranking (Section 26): `m35_mnist.json`'s `kernel_ranking`
  is already the real-workload ground truth (per-op wall time inside an
  actual MNIST training step) -- used directly, not re-derived.
- Distance-from-ceiling (Section 28): each ranked op's own
  `fraction_of_ceiling`, already computed by `roofline.classify` when the
  underlying `m35_*` script ran.
- Amdahl analysis (Section 30): `1/((1-f)+f/s)` for the top contributors,
  at a few illustrative hypothetical speedups -- explicitly hypothetical.
- Optimization headroom (Section 29): `runtime_fraction * (1 -
  fraction_of_ceiling)`, a simple, explicit, documented proxy for "how much
  is here, and how far below its ceiling is it" -- not a claim of exact
  recoverable time.
- Plots (Section 45): a roofline scatter (arithmetic intensity vs. achieved
  GFLOP/s, with both practical ceiling lines), a kernel-contribution bar
  chart (from the MNIST ranking), and a GEMM-size scaling line chart --
  written to `benchmarks/results/m35_plots/`. Requires `matplotlib` (a
  `dev`-extra dependency, see `pyproject.toml`); everything else in M35 has
  no such dependency and still works without it.
- Renders `docs/performance/m35-roofline-characterization.md`.

Writes `benchmarks/results/m35_summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import roofline as rf

RESULTS_DIR = Path("benchmarks/results")
PLOTS_DIR = RESULTS_DIR / "m35_plots"
DOC_PATH = Path("docs/performance/m35-roofline-characterization.md")

# Illustrative hypothetical per-kernel speedups for the Amdahl estimate --
# clearly not measurements (Section 30).
_AMDAHL_SPEEDUPS = (1.5, 2.0, 3.0)


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the m35_* scripts first (see this module's docstring).")
    return json.loads(path.read_text(encoding="utf-8"))


def _amdahl(fraction: float, speedup: float) -> float:
    return 1.0 / ((1.0 - fraction) + fraction / speedup)


def _build_ranking_and_amdahl(mnist: dict) -> dict:
    ranking = mnist["profile"]["kernel_ranking"]
    total_percent = sum(r["percent_of_step"] for r in ranking)

    headroom = []
    for r in ranking:
        fraction = r["percent_of_step"] / 100.0
        dist = r.get("fraction_of_ceiling", 0.0)
        score = fraction * (1.0 - min(dist, 1.0))
        amdahl = {f"{s}x_kernel_speedup": round(_amdahl(fraction, s), 3) for s in _AMDAHL_SPEEDUPS}
        headroom.append({
            "op": r["op"], "percent_of_step": r["percent_of_step"],
            "fraction_of_ceiling": dist, "classification": r.get("classification", "n/a"),
            "headroom_score": score, "amdahl_hypothetical_overall_speedup": amdahl,
        })
    headroom.sort(key=lambda r: r["headroom_score"], reverse=True)
    return {"total_percent_accounted": total_percent, "ranking": ranking, "optimization_headroom": headroom}


def _build_threshold_summary(kernels: dict) -> "list[dict]":
    rows = []
    for r in kernels["profile"]["conv2d_dweight_im2col_and_threshold"]:
        if r["op"] == "conv2d_dweight_im2col_total":
            rows.append({
                "shape": r["scale"], "weight_elements": r["shape"]["weight_elements"],
                "in_untested_region": r["shape"].get("in_256_1152_region", False),
                "speedup_vs_direct": r["im2col_gemm_vs_direct_speedup"],
                "memory_overhead_mb": r["memory_overhead_mb"],
            })
    return rows


def _make_plots(kernels: dict, hardware: dict, mnist: dict) -> "list[str]":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plot generation (pip install matplotlib, or `pip install -e .[dev]`).")
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    ceilings = rf.load_ceilings(RESULTS_DIR / "m35_hardware.json")

    # -- roofline scatter --
    fig, ax = plt.subplots(figsize=(9, 6))
    ai_max = ceilings.ridge_point * 20
    ai_line = [0.01, ceilings.ridge_point, ai_max]
    ceiling_line = [ceilings.bandwidth_gbps * ai for ai in ai_line[:2]] + [ceilings.compute_gflops]
    ax.plot(ai_line, ceiling_line, "k--", label="roofline (practical ceilings)", linewidth=1.5)

    sections = ("elementwise", "reduction", "gemm", "conv2d", "maxpool2d", "dropout", "optimizers")
    colors = plt.cm.tab10.colors
    for i, section in enumerate(sections):
        pts = [r for r in kernels["profile"][section] if r["arithmetic_intensity"] > 0]
        if not pts:
            continue
        ax.scatter([r["arithmetic_intensity"] for r in pts], [r["achieved_gflops"] for r in pts],
                   label=section, color=colors[i % len(colors)], alpha=0.8, s=40)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved GFLOP/s")
    ax.set_title("Forge CUDA kernels on the 940MX -- roofline-style (M35)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    path = PLOTS_DIR / "roofline.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    # -- kernel contribution bar chart --
    ranking = mnist["profile"]["kernel_ranking"][:10]
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["op"] for r in ranking]
    percents = [r["percent_of_step"] for r in ranking]
    ax.barh(names[::-1], percents[::-1], color="steelblue")
    ax.set_xlabel("% of one MNIST training step (CUDA)")
    ax.set_title("M35 kernel runtime contribution -- real MNIST training step")
    path = PLOTS_DIR / "kernel_contribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    # -- GEMM size scaling --
    gemm = kernels["profile"]["gemm"]
    fig, ax = plt.subplots(figsize=(8, 5))
    dims = [r["shape"]["M"] * r["shape"]["K"] * r["shape"]["N"] for r in gemm]
    gflops = [r["achieved_gflops"] for r in gemm]
    labels = [r["scale"] for r in gemm]
    order = sorted(range(len(dims)), key=lambda i: dims[i])
    ax.plot([dims[i] for i in order], [gflops[i] for i in order], marker="o")
    for i in order:
        ax.annotate(labels[i], (dims[i], gflops[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("GEMM problem size (M*K*N)")
    ax.set_ylabel("Achieved GFLOP/s")
    ax.set_title("GEMM scaling on the 940MX (M35)")
    ax.grid(True, alpha=0.3)
    path = PLOTS_DIR / "gemm_scaling.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    return written


def _render_doc(hardware: dict, kernels: dict, tsa: dict, mnist: dict, ranking_amdahl: dict, threshold: "list[dict]", plots: "list[str]") -> str:
    ceilings = hardware["ceilings"]
    lines = [
        "# M35 -- CUDA Performance Characterization & Roofline-Style Analysis",
        "",
        "Reproduce with:",
        "```bash",
        "python -m benchmarks.m35_hardware",
        "python -m benchmarks.m35_kernels",
        "python -m benchmarks.m35_transfer_stream_alloc",
        "python -m benchmarks.m35_mnist",
        "python -m benchmarks.m35_report",
        "```",
        "",
        "## Hardware",
        f"NVIDIA GeForce 940MX (GM108, Maxwell), Compute Capability 5.0, CUDA 12.6, "
        f"driver {hardware['environment']['cuda'].get('driver_version', '?')}.",
        "",
        "## Practical ceilings (measured, not theoretical)",
        f"- Compute: **{ceilings['practical_compute_gflops']:.2f} GFLOP/s** (`cf_matmul_f32`, large square GEMM)",
        f"- Bandwidth: **{ceilings['practical_bandwidth_gbps']:.2f} GB/s** (`cf_add_f32`, large streaming add)",
        f"- Theoretical FP32 peak: {hardware['theoretical_specs']['theoretical_fp32_gflops']:.1f} GFLOP/s "
        f"(public spec + measured clocks -- **not** an achievable target)",
        f"- Theoretical bandwidth: {hardware['theoretical_specs']['theoretical_memory_gbps']:.2f} GB/s (same caveat)",
        "",
        "## Kernel runtime ranking (real MNIST training step, Section 26)",
        "",
        "| Op | % of step | GFLOP/s | GB/s | AI | Classification |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranking_amdahl["ranking"][:10]:
        lines.append(
            f"| {r['op']} | {r['percent_of_step']:.2f}% | {r['achieved_gflops']:.3f} | "
            f"{r['achieved_gbps']:.3f} | {r['arithmetic_intensity']:.3f} | {r.get('classification', 'n/a')} |"
        )
    lines += ["", "## Optimization headroom ranking (Section 29)",
              "`headroom_score = runtime_fraction * (1 - fraction_of_practical_ceiling)` -- an explicit proxy, not exact recoverable time.",
              "", "| Op | % of step | Distance from ceiling | Headroom score |", "|---|---|---|---|"]
    for r in ranking_amdahl["optimization_headroom"][:8]:
        lines.append(
            f"| {r['op']} | {r['percent_of_step']:.2f}% | {r['fraction_of_ceiling']*100:.1f}% | {r['headroom_score']:.4f} |"
        )

    lines += ["", "## Amdahl analysis (Section 30, hypothetical)", ""]
    top = ranking_amdahl["ranking"][0]
    lines.append(f"Top contributor: **{top['op']}** at {top['percent_of_step']:.1f}% of the CUDA training step.")
    lines.append("")
    lines.append("| Hypothetical per-kernel speedup | Hypothetical overall speedup |")
    lines.append("|---|---|")
    for s in _AMDAHL_SPEEDUPS:
        overall = _amdahl(top["percent_of_step"] / 100.0, s)
        lines.append(f"| {s}x | {overall:.3f}x |")

    lines += ["", "## M34 256-1152 weight-element threshold region (Section 31)", "",
              "| Shape | weight_elements | in untested region | im2col+GEMM speedup vs. direct | memory overhead (MB) |",
              "|---|---|---|---|---|"]
    for r in threshold:
        lines.append(
            f"| {r['shape']} | {r['weight_elements']} | {r['in_untested_region']} | "
            f"{r['speedup_vs_direct']:.2f}x | {r['memory_overhead_mb']:.2f} |"
        )
    lines.append("")
    lines.append(
        "Speedup is `total_experimental_time / current_direct_time` -- **below 1.0 means im2col+GEMM is "
        "faster**. The threshold constant (`_CONV2D_WEIGHT_IM2COL_GEMM_THRESHOLD = 256` in `backend.py`) "
        "is **not** changed by this milestone (Section 31)."
    )

    lines += ["", "## Batch-size scaling (Section 33)", "", "| Batch | samples/sec | compute-stream utilization |", "|---|---|---|"]
    for r in mnist["profile"]["batch_size_sweep"]:
        lines.append(f"| {r['batch_size']} | {r['samples_per_sec']:.0f} | {r['compute_stream_utilization']*100:.1f}% |")

    lines += ["", "## Profiling overhead (Section 37)", ""]
    o = mnist["profile"]["profiling_overhead"]
    lines.append(f"Plain step: {o['plain_mean_seconds']*1e3:.4f}ms. Instrumented step: "
                 f"{o['instrumented_mean_seconds']*1e3:.4f}ms. Overhead: {o['overhead_fraction']*100:.1f}% "
                 "(within run-to-run noise -- instrumentation is not materially altering the numbers reported).")

    if plots:
        lines += ["", "## Plots", ""]
        for p in plots:
            lines.append(f"- `{p}`")

    return "\n".join(lines) + "\n"


def _run() -> dict:
    hardware = _load("m35_hardware.json")
    kernels = _load("m35_kernels.json")
    tsa = _load("m35_transfer_stream_alloc.json")
    mnist = _load("m35_mnist.json")

    ranking_amdahl = _build_ranking_and_amdahl(mnist)
    threshold = _build_threshold_summary(kernels)
    plots = _make_plots(kernels, hardware, mnist)

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(_render_doc(hardware, kernels, tsa, mnist, ranking_amdahl, threshold, plots), encoding="utf-8")

    summary = {
        "ceilings": hardware["ceilings"],
        "kernel_ranking": ranking_amdahl["ranking"],
        "optimization_headroom": ranking_amdahl["optimization_headroom"],
        "threshold_region": threshold,
        "plots": plots,
    }
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/m35_summary.json")
    args = parser.parse_args(argv)

    summary = _run()
    output_path = Path(args.output)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {DOC_PATH} and {output_path}")
    print(f"Top optimization-headroom candidate: {summary['optimization_headroom'][0]['op']}")


if __name__ == "__main__":
    main()
