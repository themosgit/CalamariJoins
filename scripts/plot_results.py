import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_performance_results(json_path="results.json"):
    """
    Parses the performance results JSON file and creates a line chart
    comparing the median total runtime of different algorithms.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        print(
            "Please ensure you are running this script from the project's root directory."
        )
        return

    stats = data.get("statistics", {})
    if not stats:
        print("Error: 'statistics' key not found in the JSON file.")
        return

    executables = list(stats.keys())
    median_runtimes = [
        stats[exe]["total_runtime_stats"]["median"] for exe in executables
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(executables, median_runtimes, marker="o", linestyle="-", color="b")

    ax.set_ylabel("Median Total Runtime (ms)", fontsize=12)
    ax.set_title(
        "Algorithm Performance Comparison (Median of 20 Iterations)",
        fontsize=14,
        weight="bold",
    )
    ax.tick_params(axis="x", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=10)

    for i, txt in enumerate(median_runtimes):
        ax.annotate(
            f"{int(txt)}",
            (executables[i], median_runtimes[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=300)
    print("Plot saved as performance_comparison.png")
    plt.show()


def plot_per_query_performance(json_path="results.json"):
    """
    Parses performance results and creates a single image with a bar chart for each query,
    comparing the median runtime of different algorithms for that query.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        print(
            "Please ensure you are running this script from the project's root directory."
        )
        return

    stats = data.get("statistics", {})
    if not stats:
        print("Error: 'statistics' key not found in the JSON file.")
        return

    executables = list(stats.keys())
    queries = list(stats[executables[0]]["per_query_stats"].keys())

    num_queries = len(queries)
    cols = 4
    rows = (num_queries + cols - 1) // cols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        rows, cols, figsize=(20, 5 * rows), constrained_layout=True
    )
    fig.suptitle("Per-Query Performance Comparison", fontsize=16, weight="bold")
    axes = axes.flatten()

    for i, query in enumerate(queries):
        ax = axes[i]
        median_runtimes = []
        for exe in executables:
            median_runtimes.append(stats[exe]["per_query_stats"][query]["median"])

        colors = plt.cm.viridis(np.linspace(0, 1, len(executables)))
        bars = ax.bar(executables, median_runtimes, color=colors)

        ax.set_ylabel("Median Runtime (µs)", fontsize=10)
        ax.set_title(f"Query: {query}", fontsize=12, weight="bold")
        ax.tick_params(axis="x", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    for i in range(num_queries, len(axes)):
        axes[i].set_visible(False)

    plt.savefig("per_query_performance_grid.png", dpi=300)
    print("Per-query performance grid saved as per_query_performance_grid.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_performance_results()
    plot_per_query_performance()
