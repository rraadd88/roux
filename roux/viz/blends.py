"""Blends of plotting functions."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_ranks(
    data: pd.DataFrame,
    kws_plot: dict,
    col: str,
    colid: str,
    col_label: str = None,
    xlim_min: float = -20,
    ax=None,
):
    if col_label is None:
        col_label = colid
    if ax is None:
        ax = plt.gca()
    from roux.viz.scatter import plot_ranks

    ax, data1 = plot_ranks(
        data,
        col=col,
        colid=colid,
        ranks_on="x",
        ascending=False,
        line=True,
        show_topn=kws_plot["topn"],
        ax=ax,
    )
    from roux.viz.annot import annot_side_curved

    ax = annot_side_curved(
        data1.sort_values("rank").head(kws_plot["topn"]),
        colx="rank",
        coly=col,
        x=ax.get_xlim()[1],  # *0.3,
        ylim=[
            ax.get_ylim()[0]
            + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * kws_plot["ylim_fr"][0]),
            ax.get_ylim()[0]
            + ((ax.get_ylim()[1] - ax.get_ylim()[0]) * kws_plot["ylim_fr"][1]),
        ],
        col_label=col_label,
        ax=ax,
        ha="right",
        **kws_plot["annot_side_curved"],
    )
    ax.set(
        xlim=[
            xlim_min,
            ax.get_xlim()[1],
        ]
    )
    ax.set(xticks=[1, ax.get_xticks()[ax.get_xticks() < ax.get_xlim()[1]].max()])
    from roux.viz.ax_ import format_ax

    format_ax()
    return ax
