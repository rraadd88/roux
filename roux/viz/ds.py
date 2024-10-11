"""For wrappers around pandas Series plotting attributes."""

import matplotlib.pyplot as plt
import pandas as pd

# ## internal
from roux.lib import to_rs

@to_rs
def hist(
    ds: pd.Series,
    ax: plt.Axes=None,
    kws_set_label_n={},
    **kws,
):
    if ax is None:
        ax=plt.gca()
    ax=ds.hist(
        **kws,
        ax=ax
    )
    if hasattr(ds,'name'):
        ax.set(
            xlabel=ds.name,
            ylabel='Count',
        )
    from roux.viz.ax_ import set_label
    set_label(
        **{
            **dict(
                ax=ax,
                x=1,
                y=0,
                s=f"n={len(ds)}",
                ha='right',
                va='bottom',
            ),
            **kws_set_label_n
        }
    )
    return ax