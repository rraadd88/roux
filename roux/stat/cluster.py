"""For clustering data."""

## logging
import logging

## data
import numpy as np
import pandas as pd

## viz
import matplotlib.pyplot as plt

## stats
import scipy as sc
## internal


# scikit learn
def check_clusters(df: pd.DataFrame):
    """Check clusters.

    Args:
        df (DataFrame): dataframe.

    """
    return (
        df.groupby(["cluster #"]).agg({"silhouette value": np.max})["silhouette value"]
        >= df["silhouette value"].mean()
    ).all()


def get_clusters(
    X: np.array, n_clusters: int, random_state=88, params={}, test=False
) -> dict:
    """Get clusters.

    Args:
        X (np.array): vector
        n_clusters (int): int
        random_state (int, optional): random state. Defaults to 88.
        params (dict, optional): parameters for the `MiniBatchKMeans` function. Defaults to {}.
        test (bool, optional): test. Defaults to False.

    Returns:
        dict:
    """
    from sklearn import cluster, metrics

    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        **params,
    ).fit(X)
    clusters = kmeans.predict(X)
    ds = pd.Series(dict(zip(X.index, clusters)))
    ds.name = "cluster #"
    df = pd.DataFrame(ds)
    df.index.name = X.index.name
    df["cluster #"].value_counts()
    # Compute the silhouette scores for each sample
    df["silhouette value"] = metrics.silhouette_samples(X, clusters)
    #     if test:
    #         print(f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: {is_optimum}, random state {random_state}")
    if not check_clusters:
        logging.warning(
            f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: is_optimum, random state {random_state}"
        )
    dn2df = {
        "clusters": df.reset_index(),
        "inertia": kmeans.inertia_,
        "centers": pd.DataFrame(
            kmeans.cluster_centers_, index=range(n_clusters), columns=X.columns
        ).rename_axis(
            index="cluster #"
        ),  # .stack().reset_index().rename(columns={'level_1':'variable',0:'value'}),
    }
    return dn2df


def get_n_clusters_optimum(df5: pd.DataFrame, test=False) -> int:
    """Get n clusters optimum.

    Args:
        df5 (DataFrame): input dataframe
        test (bool, optional): test. Defaults to False.

    Returns:
        int: knee point.
    """
    from kneed import KneeLocator

    kn = KneeLocator(
        x=df5["total clusters"],
        y=df5["inertia"],
        curve="convex",
        direction="decreasing",
    )
    if test:
        import matplotlib.pyplot as plt

        kn.plot_knee()
        plt.title(f"knee point={kn.knee}")
    return kn.knee


def plot_silhouette(df: pd.DataFrame, n_clusters_optimum=None, ax=None):
    """Plot silhouette

    Args:
        df (DataFrame): input dataframe.
        n_clusters_optimum (int, optional): number of clusters. Defaults to None:int.
        ax (axes, optional): axes object. Defaults to None:axes.

    Returns:
        ax (axes, optional): axes object. Defaults to None:axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    ax = plt.subplot() if ax is None else ax
    ax = sns.violinplot(
        data=df.groupby(["total clusters", "cluster #"])
        .agg({"silhouette value": np.mean})
        .reset_index(),
        y="silhouette value",
        x="total clusters",
        color="salmon",
        alpha=0.7,
        ax=ax,
    )
    ax = sns.pointplot(
        data=df.groupby("total clusters")
        .agg({"silhouette value": np.mean})
        .reset_index(),
        y="silhouette value",
        x="total clusters",
        color="k",
        ax=ax,
    )
    if n_clusters_optimum is not None:
        ax.annotate(
            "optimum",
            xy=(
                n_clusters_optimum - int(ax.get_xticklabels()[0].get_text()),
                ax.get_ylim()[0],
            ),
            xycoords="data",
            xytext=(+50, +50),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="->", ec="k", connectionstyle="angle3,angleA=0,angleB=-90"
            ),
        )
    ax.set_xlabel("clusters")
    return ax


def get_clusters_optimum(
    X: np.array,
    n_clusters=range(2, 11),
    params_clustering={},
    test=False,
) -> dict:
    """Get optimum clusters.

    Args:
        X (np.array): samples to cluster in indexed format.
        n_clusters (int, optional): _description_. Defaults to range(2,11).
        params_clustering (dict, optional): parameters provided to `get_clusters`. Defaults to {}.
        test (bool, optional): test. Defaults to False.

    Returns:
        dict: _description_
    """
    dn2d = {}
    for n in n_clusters:
        dn2d[n] = get_clusters(X=X, n_clusters=n, test=test, params=params_clustering)
    df1 = (
        pd.DataFrame(
            pd.Series({k: dn2d[k]["inertia"] for k in dn2d}), columns=["inertia"]
        )
        .rename_axis(index="total clusters")
        .reset_index()
    )
    # TODO identify saturation point in the intertia plot for n_clusters_optimum
    n_clusters_optimum = get_n_clusters_optimum(df1, test=test)
    #
    dn2df = {
        dn: pd.concat(
            {k: dn2d[k][dn] for k in dn2d}, axis=0, names=["total clusters"]
        ).reset_index()
        for dn in ["clusters", "centers"]
    }
    if not check_clusters(dn2df["clusters"]):
        logging.warning("low silhoutte scores")
        return
    if test:
        import matplotlib.pyplot as plt

        plt.figure()
        plot_silhouette(df=dn2df["clusters"], n_clusters_optimum=None)
    # make output
    dn2df = {
        dn: dn2df[dn]
        .loc[(dn2df[dn]["total clusters"] == n_clusters_optimum), :]
        .drop(["total clusters"], axis=1)
        for dn in dn2df
    }
    return dn2df


def get_gmm_params(
    g,
    x,
    n_clusters=2,
    test=False,
):
    """Intersection point of the two peak Gaussian mixture Models (GMMs).

    Args:
        out (str): `coff` only or `params` for all the parameters.

    """
    assert n_clusters == 2
    weights = g.weights_
    means = g.means_
    covars = g.covariances_
    stds = np.sqrt(covars).ravel().reshape(n_clusters, 1)
    # logging.info(f'weights {weights}')
    f = x.reshape(-1, 1)
    x.sort()
    two_pdfs = sc.stats.norm.pdf(np.array([x, x]), means, stds)
    mix_pdf = np.matmul(weights.reshape(1, n_clusters), two_pdfs)
    return mix_pdf, two_pdfs, means, weights


def get_gmm_intersection(x, two_pdfs, means, weights, test=False):
    from roux.stat.solve import get_intersection_locations

    idxs = get_intersection_locations(
        y1=two_pdfs[0] * weights[0], y2=two_pdfs[1] * weights[1], test=False, x=x
    )
    x_intersections = x[idxs]
    if test:
        logging.info(f"intersections {x_intersections}")
    ms = sorted([means[0][0], means[1][0]])
    if len(x_intersections) > 1:
        if test:
            logging.info(x_intersections)
            logging.info(ms)
            logging.info([i for i in x_intersections if i > ms[0] and i < ms[1]])
        coffs_ = [i for i in x_intersections if i > ms[0] and i < ms[1]]
        if len(coffs_) != 0:
            coff = coffs_[0]
        else:
            coff = None
        if test:
            logging.info(coff)
    else:
        coff = x_intersections[0]
    return coff


def cluster_1d(
    ds: pd.Series,
    n_clusters: int,
    clf_type="gmm",
    random_state=1,
    test=False,
    returns=["coff"],
    **kws_clf,
) -> dict:
    """Cluster 1D data.

    Args:
        ds (Series): series.
        n_clusters (int): number of clusters.
        clf_type (str, optional): type of classification. Defaults to 'gmm'.
        random_state (int, optional): random state. Defaults to 88.
        test (bool, optional): test. Defaults to False.
        returns (list, optional): return format. Defaults to ['df','coff','ax','model'].
        ax (axes, optional): axes object. Defaults to None.

    Raises:
        ValueError: clf_type

    Returns:
        dict: _description_
    """
    assert not ds._is_view, "input series should be a copy not a view"
    x = ds.to_numpy()
    X = x.reshape(-1, 1)
    if clf_type.lower() == "gmm":
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    elif clf_type.lower() == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, **kws_clf).fit(
            X,
        )
    else:
        raise ValueError(clf_type)
    ## fit and predic
    labels = model.fit_predict(X)
    assert model.converged_
    df = pd.DataFrame({"value": x, "label": labels == 1})
    if clf_type == "gmm":
        mix_pdf, two_pdfs, means, weights = get_gmm_params(
            g=model,
            x=x,
            n_clusters=n_clusters,
            test=test,
        )
        coff = get_gmm_intersection(x, two_pdfs, means, weights, test=test)
    d = {}
    for k in returns:
        d[k] = locals()[k]
    if test:
        if clf_type == "gmm":
            from roux.viz.dist import plot_gmm

            ax = plot_gmm(
                x,
                coff,
                mix_pdf,
                two_pdfs,
                weights,
                n_clusters=n_clusters,
            )
        else:
            coffs = df.groupby("label")["value"].agg(min).values
            for c in coffs:
                if c > df["value"].quantile(0.05) and c < df["value"].quantile(0.95):
                    coff = c
                    break
            logging.info(f"coff:{c}; selected from {coffs}")
            ax.axvline(coff, color="k")
            ax.text(coff, ax.get_ylim()[1], f"{coff:.1f}", ha="center", va="bottom")
    return d


## umap
def get_pos_umap(df1, spread=100, test=False, k="", **kws) -> pd.DataFrame:
    """Get positions of the umap points.

    Args:
        df1 (DataFrame): input dataframe
        spread (int, optional): spead extent. Defaults to 100.
        test (bool, optional): test. Defaults to False.
        k (str, optional): number of clusters. Defaults to ''.

    Returns:
        DataFrame: output dataframe.
    """
    try:
        import umap
    except ImportError:
        logging.error(
            "umap package not installed. Installation command: pip install umap-learn"
        )
        return

    reducer = umap.UMAP(spread=spread, *kws)
    embedding = reducer.fit_transform(df1)
    if test:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=["r" if k in s else "k" for s in df1.index.get_level_values(0)],
            alpha=0.1,
        )
        plt.gca().set_aspect("equal", "datalim")
    df2 = pd.DataFrame(embedding, columns=["x", "y"])
    df2.index = df1.index
    return df2.reset_index()
