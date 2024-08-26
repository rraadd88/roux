"""For heatmaps."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## internal
import roux.lib.dfs as rd  # noqa
from roux.lib.str import linebreaker


def plot_table(
    df1: pd.DataFrame,
    xlabel: str = None,
    ylabel: str = None,
    annot: bool = True,
    cbar: bool = False,
    linecolor: str = "k",
    linewidths: float = 1,
    cmap: str = None,
    sorty: bool = False,
    linebreaky: bool = False,
    scales: tuple = [1, 1],
    ax: plt.Axes = None,
    **kws,
) -> plt.Axes:
    """Plot to show a table.

    Args:
        df1 (pd.DataFrame): input data.
        xlabel (str, optional): x label. Defaults to None.
        ylabel (str, optional): y label. Defaults to None.
        annot (bool, optional): show numbers. Defaults to True.
        cbar (bool, optional): show colorbar. Defaults to False.
        linecolor (str, optional): line color. Defaults to 'k'.
        linewidths (float, optional): line widths. Defaults to 1.
        cmap (str, optional): color map. Defaults to None.
        sorty (bool, optional): sort rows. Defaults to False.
        linebreaky (bool, optional): linebreak for y labels. Defaults to False.
        scales (tuple, optional): scale of the table. Defaults to [1,1].
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `sns.heatmap` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    # print(df1.index.name,df1.columns.name)
    if xlabel is None and df1.index.name is not None:
        ylabel = df1.index.name
    if ylabel is None and df1.columns.name is not None:
        xlabel = df1.columns.name
    #     print(xlabel,ylabel)
    from roux.viz.colors import make_cmap

    if sorty:
        df1 = df1.loc[df1.sum(axis=1).sort_values(ascending=False).index, :]
    if linebreaky:
        df1.index = [linebreaker(s, break_pt=35) for s in df1.index]
    if ax is None:
        fig, ax = plt.subplots(
            figsize=[(df1.shape[1] * 0.6) * scales[0], (df1.shape[0] * 0.5) * scales[1]]
        )
    ax = sns.heatmap(
        df1,
        cmap=make_cmap(["#ffffff", "#ffffff"]) if cmap is None else cmap,
        annot=annot,
        cbar=cbar,
        linecolor=linecolor,
        linewidths=linewidths,
        ax=ax,
        **kws,
    )
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("center")
    ax.patch.set_edgecolor("k")
    ax.patch.set_linewidth(1)
    # set_ylabel(ax=ax,s=df1.index.name if ylabel is None else ylabel,xoff=0.05,yoff=0.01)
    return ax
