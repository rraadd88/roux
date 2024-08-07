"""For processing pandas Series."""

# ## logging
# import logging
## data
import pandas as pd


def get_near_quantile(
    x: pd.Series,
    q: float,
):
    """
    Retrieve the nearest value to a quantile.
    """
    return x[(x - x.quantile(q)).abs().argsort()[:1]].values[0]
