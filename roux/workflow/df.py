"""For management of tables."""

import logging
import pandas as pd


## metadata to process the data
def exclude_items(
    df1: pd.DataFrame,
    metadata: dict,
) -> pd.DataFrame:
    """Exclude items from the table with the workflow info.

    Args:
        df1 (pd.DataFrame): input table.
        metadata (dict): metadata of the repository.

    Returns:
        pd.DataFrame: output.
    """
    for c in df1:
        if c in metadata["exclude"]:
            # info(c)
            shape0 = df1.shape
            df1 = df1.loc[~(df1[c].isin(metadata["exclude"][c])), :]
            if shape0 != df1.shape:
                logging.info(f"{c}:{df1[c].nunique()}")
    return df1
