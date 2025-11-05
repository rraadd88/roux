"""For processing pandas Series."""

# ## logging
# import logging
## data
import pandas as pd

def to_cat(
    ds1: pd.Series,
    cats: list,
    ordered: bool=True,
    ):
    """To series containing categories.

    Parameters:
        ds1 (Series): input series.
        cats (list): categories.
        ordered (bool): if the categories are ordered (True).

    Returns:
        ds1 (Series): output series.
    """
    ds1 = ds1.astype("category")
    ds1 = ds1.cat.set_categories(new_categories=cats, ordered=ordered)
    assert not ds1.isnull().any(), ds1.isnull().sum()
    return ds1
    
def get_near_quantile(
    x: pd.Series,
    q: float,
):
    """
    Retrieve the nearest value to a quantile.
    """
    return x[(x - x.quantile(q)).abs().argsort()[:1]].values[0]
