"""For workflow monitors."""

import pandas as pd
import matplotlib.pyplot as plt
## TODOs: use ## stats


def plot_workflow_log(dplot: pd.DataFrame) -> plt.Axes:
    """Plot workflow log.

    Args:
        dplot (pd.DataFrame): input data (dparam).

    Returns:
        plt.Axes: output.

    TODOs:
        1. use the statistics tagged as `## stats`.
    """
    parameters_count_max = (
        dplot.groupby(["function name"])
        .agg({"parameter name input list": lambda x: len(x) + 1})
        .max()
        .values[0]
    )
    plt.figure(
        figsize=[
            parameters_count_max * 1.5,  # *0.3,
            len(dplot) * 0.5 + 2,
        ]
    )
    ax = plt.subplot(1, 5, 2)
    #     ax=plt.subplot()
    elements = [
        "script",
        "function",
    ]
    for elementi, element in enumerate(elements):
        _ = dplot.apply(
            lambda x: ax.text(
                x[f"{element} x{''  if element!='parameter' else ' input'}"],
                x[f"{element} y{''  if element!='parameter' else ' input'}"],
                x[f"{element} name{''  if element!='parameter' else ' input'}"],
            ),
            axis=1,
        )
    _ = dplot.apply(
        lambda x: ax.annotate(
            x["parameter name output"],
            xy=(x["parameter x input"], x["parameter y input"]),
            xycoords="data",
            xytext=(x["parameter x output"], x["parameter y output"]),
            textcoords="data",
            #             size=20,
            va="center",
            ha="center",
            arrowprops=dict(
                arrowstyle="<|-",
                alpha=0.5,
                color="lime",
                lw=4,
                connectionstyle="arc3,rad=0.4",
            ),
        ),
        axis=1,
    )
    ax.set_ylim(len(dplot), 0)
    #     ax.set_xlim(0,parameters_count_max)
    ax.set_axis_off()
    return ax
