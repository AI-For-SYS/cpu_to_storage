"""
Plot comparison bar charts: CPP best vs CPP same-config vs Tunable.

Usage:
    python plot_comparison.py <cpp_data.json> <tunable_data.json> [cpp_concurrent.json]

Examples:
    python plot_comparison.py results/data_*cpp*.json results/data_*tunable*.json
    python plot_comparison.py results/data_*cpp*.json results/data_*tunable*.json results/concurrent_*cpp*.json
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

 
def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_cpp_best(results, metric="write"):
    """Find the best throughput across all thread counts and block sizes."""
    tp = results.get(f"{metric}_throughput_gbs", {})
    best_val = 0
    best_tc = ""
    best_bs = ""
    for tc, block_sizes in tp.items():
        for bs, val in block_sizes.items():
            if val > best_val:
                best_val = val
                best_tc = tc
                best_bs = bs
    return best_val, best_tc, best_bs


def get_cpp_at_block_size(results, target_bs_mb, metric="write"):
    """Find the best throughput for a specific (or closest) block size across all thread counts."""
    tp = results.get(f"{metric}_throughput_gbs", {})

    # Collect all available block sizes
    all_bs = set()
    for tc, block_sizes in tp.items():
        all_bs.update(int(bs) for bs in block_sizes.keys())

    # If exact match not available, find closest
    actual_bs = target_bs_mb
    if all_bs and target_bs_mb not in all_bs:
        actual_bs = min(all_bs, key=lambda x: abs(x - target_bs_mb))
    actual_bs_str = str(actual_bs)

    best_val = 0
    best_tc = ""
    for tc, block_sizes in tp.items():
        if actual_bs_str in block_sizes:
            val = block_sizes[actual_bs_str]
            if val > best_val:
                best_val = val
                best_tc = tc
    return best_val, best_tc, actual_bs


def get_tunable_throughput(results, metric="write"):
    """Get tunable throughput (single value per metric)."""
    tp = results.get(f"{metric}_throughput_gbs", {})
    for tc, block_sizes in tp.items():
        for bs, val in block_sizes.items():
            return val, tc, bs
    return 0, "", ""


def plot_experiment(cpp_data, tunable_data, cpp_concurrent=None, output_dir="plots"):
    """Generate comparison bar charts for one experiment."""
    os.makedirs(output_dir, exist_ok=True)

    cpp_config = cpp_data.get("config", {})
    tunable_config = tunable_data.get("config", {})
    total_gb = cpp_config.get("total_data_size_gb", "?")

    # Extract tunable configs
    write_cfg = tunable_config.get("tunable_write_config", {})
    read_cfg = tunable_config.get("tunable_read_config", {})
    concurrent_cfg = tunable_config.get("tunable_concurrent_config", {})

    # Determine which metrics to plot
    metrics = ["write", "read"]
    has_concurrent = "concurrent_throughput_gbs" in tunable_data
    if has_concurrent:
        metrics.append("concurrent")

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get tunable result
        tunable_val, tunable_tc, tunable_bs = get_tunable_throughput(tunable_data, metric)

        # Get cpp data source (concurrent may be in separate file)
        if metric == "concurrent" and cpp_concurrent:
            cpp_source = cpp_concurrent
        else:
            cpp_source = cpp_data

        # Get cpp best overall
        cpp_best_val, cpp_best_tc, cpp_best_bs = get_cpp_best(cpp_source, metric)

        # Get cpp at same (or closest) block size as tunable
        tunable_bs_mb = int(tunable_bs) if tunable_bs else 0
        cpp_same_val, cpp_same_tc, actual_bs = get_cpp_at_block_size(cpp_source, tunable_bs_mb, metric)

        # Bar data
        closest_label = f"CPP @{actual_bs}MB" if actual_bs == tunable_bs_mb else f"CPP @{actual_bs}MB\n(closest to {tunable_bs}MB)"
        labels = [
            f"CPP best\n({cpp_best_bs}MB, {cpp_best_tc}t)",
            f"{closest_label}\n(best: {cpp_same_tc}t)",
            f"Tunable\n({tunable_bs}MB, {tunable_tc}t)",
        ]
        values = [cpp_best_val, cpp_same_val, tunable_val]
        colors = ["#4477AA", "#66CCEE", "#EE6677"]

        bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Throughput (GB/s)", fontsize=11)
        ax.set_title(f"{metric.capitalize()} Throughput", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    # Overall title
    cluster = cpp_config.get("cluster", "unknown")
    storage = cpp_config.get("file_system", "unknown")
    iterations = cpp_config.get("num_iterations", "?")
    fig.suptitle(
        f"CPP vs Threaded Tunable — {total_gb}GB, {iterations} iterations\n"
        f"Cluster: {cluster} | Storage: {storage}",
        fontsize=13, y=1.02
    )

    plt.tight_layout()

    # Save
    ts = tunable_data.get("config", {}).get("total_data_size_gb", "")
    output_path = os.path.join(output_dir, f"comparison_{total_gb}gb.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_comparison.py <cpp_data.json> <tunable_data.json> [cpp_concurrent.json]")
        sys.exit(1)

    cpp_data_path = sys.argv[1]
    tunable_data_path = sys.argv[2]
    cpp_concurrent_path = sys.argv[3] if len(sys.argv) > 3 else None

    cpp_data = load_json(cpp_data_path)
    tunable_data = load_json(tunable_data_path)
    cpp_concurrent = load_json(cpp_concurrent_path) if cpp_concurrent_path else None

    output_dir = os.path.join(os.path.dirname(cpp_data_path), "plots")
    plot_experiment(cpp_data, tunable_data, cpp_concurrent, output_dir)


if __name__ == "__main__":
    main()
