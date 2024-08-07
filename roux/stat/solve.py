"""For solving equations."""

import numpy as np
import matplotlib.pyplot as plt


def get_intersection_locations(
    y1: np.array, y2: np.array, test: bool = False, x: np.array = None
) -> list:
    """Get co-ordinates of the intersection (x[idx]).

    Args:
        y1 (np.array): vector.
        y2 (np.array): vector.
        test (bool, optional): test mode. Defaults to False.
        x (np.array, optional): vector. Defaults to None.

    Returns:
        list: output.
    """
    idxs = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    if test:
        x = range(len(y1)) if x is None else x
        plt.figure(figsize=[2.5, 2.5])
        ax = plt.subplot()
        ax.plot(x, y1, color="r", label="line1", alpha=0.5)
        ax.plot(x, y2, color="b", label="line2", alpha=0.5)
        _ = [ax.axvline(x[i], color="k") for i in idxs]
        _ = [
            ax.text(x[i], ax.get_ylim()[1], f"{x[i]:1.1f}", ha="center", va="bottom")
            for i in idxs
        ]
        ax.legend(bbox_to_anchor=[1, 1])
        ax.set(xlabel="x", ylabel="density")
    return idxs
