"""For network related stats."""

import pandas as pd


def get_subgraphs(df1: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """Subgraphs from the the edge list.

    Args:
        df1 (pd.DataFrame): input dataframe containing edge-list.
        source (str): source node.
        target (str): taget node.

    Returns:
        pd.DataFrame: output.
    """
    import networkx as nx

    g = nx.from_pandas_edgelist(df1, source=source, target=target)
    ug = g.to_undirected()
    sgs = nx.connected_components(ug)
    dn2df = {}
    for sg in sgs:
        ns = sorted(sg)
        dn2df["--".join(ns)] = pd.Series(ns)
    return (
        pd.concat(dn2df, names=["subnetwork name"])
        .reset_index()
        .drop(["level_1"], axis=1)
        .rename(columns={0: "node name"})
    )
