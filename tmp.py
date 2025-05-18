import matplotlib.pyplot as plt
import numpy as np


def plot_rejection_trends():
    # Data for each p-norm (rejected points per iteration and Hausdorff distance)
    data = {
        "p=1": {"rejected": [996, 483, 285, 1930, 2949], "hausdorff": 0.00015},
        "p=2": {"rejected": [960, 644, 195, 1920, 3068], "hausdorff": 0.00010},
        "p=∞": {"rejected": [957, 572, 231, 1945, 2916], "hausdorff": 0.00017},
    }

    plt.figure(figsize=(10, 6))

    # Plot cumulative rejected points for each p-norm
    for p_label, p_data in data.items():
        cumulative_rejected = np.cumsum(p_data["rejected"])
        iterations = range(1, len(cumulative_rejected) + 1)

        plt.plot(
            iterations,
            cumulative_rejected,
            marker="o",
            label=f'{p_label} (Hausdorff={p_data["hausdorff"]:.5f})',
        )

    plt.title("Cumulative Rejected Points Trend by p-norm")
    plt.xlabel("Iteration")
    plt.ylabel("Cumulative Rejected Points")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()


plot_rejection_trends()
