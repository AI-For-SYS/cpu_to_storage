"""
Multi-backend comparison bar chart: cpp grid-best vs one-or-more tuned backends.

Each tuned JSON must contain a single throughput value per mode (write / read /
concurrent) and have `config.implementation` set to the backend name
(e.g., "threaded_tunable", "iouring") — used as the legend label.

The cpp baseline is read from two files:
  - <cpp_data.json>       — contains write + read (grid of threads x block_size)
  - <cpp_concurrent.json> — contains concurrent (grid)
For each mode, the best cpp throughput across the grid is used.

Usage:
    python plot_multi_backend.py <cpp_data.json> <cpp_concurrent.json> <tuned1.json> [<tuned2.json> ...]

The output filename includes every backend name involved, so multiple variants
coexist in the same plots/ directory.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cpp_best(results, metric):
    """Return max throughput across the grid for a given mode."""
    tp = results.get(f"{metric}_throughput_gbs", {})
    best = 0.0
    best_tc = ""
    best_bs = ""
    for tc, bs_map in tp.items():
        for bs, val in bs_map.items():
            if val > best:
                best = val
                best_tc = tc
                best_bs = bs
    return best, best_tc, best_bs


def tuned_value(results, metric):
    """Return the single (throughput, tc, bs) from a tuned result file."""
    tp = results.get(f"{metric}_throughput_gbs", {})
    for tc, bs_map in tp.items():
        for bs, val in bs_map.items():
            return val, tc, bs
    return 0.0, "", ""


def plot(cpp_data, cpp_concurrent, tuned_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    modes = ["write", "read", "concurrent"]

    # Gather cpp bests per mode
    cpp_values = {}
    cpp_labels = {}
    for mode in modes:
        src = cpp_concurrent if mode == "concurrent" else cpp_data
        val, tc, bs = cpp_best(src, mode)
        cpp_values[mode] = val
        cpp_labels[mode] = f"{bs}MB, {tc}t"

    # Gather tuned backends
    tuned_entries = []  # list of (backend_label, values_dict, labels_dict)
    for tpath in tuned_files:
        t = load_json(tpath)
        backend = t.get("config", {}).get("implementation", "tuned")
        values = {}
        labels = {}
        for mode in modes:
            val, tc, bs = tuned_value(t, mode)
            values[mode] = val
            labels[mode] = f"{bs}MB, {tc}"  # tc is queue_depth for iouring, thread_count for threads
        tuned_entries.append((backend, values, labels))

    n_groups = len(modes)
    n_bars = 1 + len(tuned_entries)  # cpp + each tuned

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    # cpp bars (reference)
    cpp_bars = ax.bar(
        x - (n_bars - 1) / 2 * bar_width,
        [cpp_values[m] for m in modes],
        width=bar_width,
        label="cpp (grid best)",
        color="#4477AA",
        edgecolor="black", linewidth=0.5,
    )
    for bar, mode in zip(cpp_bars, modes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{bar.get_height():.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2, -1.3,
            cpp_labels[mode], ha="center", va="top", fontsize=7, color="gray",
        )

    # Tuned bars
    palette = ["#228833", "#EE6677", "#CCBB44", "#66CCEE"]
    for i, (backend, values, labels) in enumerate(tuned_entries):
        offset = (i + 1 - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            [values[m] for m in modes],
            width=bar_width,
            label=f"{backend} (tuned)",
            color=palette[i % len(palette)],
            edgecolor="black", linewidth=0.5,
        )
        for bar, mode in zip(bars, modes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
            )
            # Annotate with delta vs cpp
            delta = (values[mode] - cpp_values[mode]) / cpp_values[mode] * 100
            color = "green" if delta > 0 else "#B44"
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{delta:+.1f}%", ha="center", va="center",
                fontsize=8, fontweight="bold", color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modes], fontsize=12)
    ax.set_ylabel("Throughput (GB/s)", fontsize=11)
    ax.set_ylim(0, max(
        max(cpp_values.values()),
        max((v[m] for _, v, _ in tuned_entries for m in modes)),
    ) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    # Context line in title
    cpp_cfg = cpp_data.get("config", {})
    total_gb = cpp_cfg.get("total_data_size_gb", "?")
    cluster = cpp_cfg.get("cluster", "unknown")
    storage = cpp_cfg.get("file_system", "unknown").split(" (")[0]
    iterations = cpp_cfg.get("num_iterations", "?")
    backends_str = " & ".join([b for b, _, _ in tuned_entries])
    ax.set_title(
        f"cpp vs {backends_str} (tuned) — {total_gb} GB, {iterations} iterations\n"
        f"{cluster} · {storage}",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()

    out_name = f"multi_backend_cpp_vs_{'_'.join(b for b, _, _ in tuned_entries)}_{total_gb}gb.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    cpp_data_path = sys.argv[1]
    cpp_concurrent_path = sys.argv[2]
    tuned_paths = sys.argv[3:]

    cpp_data = load_json(cpp_data_path)
    cpp_concurrent = load_json(cpp_concurrent_path)

    output_dir = os.path.join(os.path.dirname(cpp_data_path), "plots")
    plot(cpp_data, cpp_concurrent, tuned_paths, output_dir)


if __name__ == "__main__":
    main()
