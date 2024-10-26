"""For processing binary data."""

import logging
import numpy as np
import pandas as pd


## overlap
def compare_bools_jaccard(x, y):
    """Compare bools in terms of the jaccard index.

    Args:
        x (list): list of bools.
        y (list): list of bools.

    Returns:
        float: jaccard index.
    """
    x = np.asarray(x, np.bool)
    y = np.asarray(y, np.bool)
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())


def compare_bools_jaccard_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise compare bools in terms of the jaccard index.

    Args:
        df (DataFrame): dataframe with boolean columns.

    Returns:
        DataFrame: matrix with comparisons between the columns.
    """
    from roux.stat.binary import compare_bools_jaccard

    dmetrics = pd.DataFrame(index=df.columns.tolist(), columns=df.columns.tolist())
    for c1i, c1 in enumerate(df.columns):
        for c2i, c2 in enumerate(df.columns):
            if c1i > c2i:
                dmetrics.loc[c1, c2] = compare_bools_jaccard(
                    df.dropna(subset=[c1, c2])[c1], df.dropna(subset=[c1, c2])[c2]
                )
            elif c1i == c2i:
                dmetrics.loc[c1, c2] = 1
    for c1i, c1 in enumerate(df.columns):
        for c2i, c2 in enumerate(df.columns):
            if c1i < c2i:
                dmetrics.loc[c1, c2] = dmetrics.loc[c2, c1]
    return dmetrics


def classify_bools(l: list) -> str:
    """Classify bools.

    Args:
        l (list): list of bools

    Returns:
        str: classification.
    """
    return "both" if all(l) else "either" if any(l) else "neither"


## aggregate
def frac(x: list) -> float:
    """Fraction.

    Args:
        x (list): list of bools.

    Returns:
        float: fraction of True values.
    """
    if len(x)==0:
        return np.nan
    else:
        return sum(x) / len(x)


def perc(x: list) -> float:
    """Percentage.

    Args:
        x (list): list of bools.

    Returns:
        float: Percentage of the True values
    """
    return frac(x) * 100


## confusion_matrix
def get_stats_confusion_matrix(
    df_: pd.DataFrame,
) -> pd.DataFrame:
    """Get stats confusion matrix.

    Args:
        df_ (DataFrame): Confusion matrix.

    Returns:
        DataFrame: stats.
    """
    d0 = {}
    d0["TP"] = df_.loc[True, True]
    d0["TN"] = df_.loc[False, False]
    d0["FP"] = df_.loc[False, True]
    d0["FN"] = df_.loc[True, False]
    # Sensitivity, hit rate, recall, or true positive rate
    d0["TPR"] = d0["TP"] / (d0["TP"] + d0["FN"])
    # Specificity or true negative rate
    d0["TNR"] = d0["TN"] / (d0["TN"] + d0["FP"])
    # Precision or positive predictive value
    d0["PPV"] = d0["TP"] / (d0["TP"] + d0["FP"])
    # Negative predictive value
    d0["NPV"] = d0["TN"] / (d0["TN"] + d0["FN"])
    # Fall out or false positive rate
    d0["FPR"] = d0["FP"] / (d0["FP"] + d0["TN"])
    # False negative rate
    d0["FNR"] = d0["FN"] / (d0["TP"] + d0["FN"])
    # False discovery rate
    d0["FDR"] = d0["FP"] / (d0["TP"] + d0["FP"])
    # Overall accuracy
    d0["ACC"] = (d0["TP"] + d0["TN"]) / (d0["TP"] + d0["FP"] + d0["FN"] + d0["TN"])
    df1 = pd.Series(d0).to_frame("value")
    df1.index.name = "variable"
    df1 = df1.reset_index()
    return df1


## thresholding
def get_cutoff(
    y_true,
    y_score,
    method,  #'roc','pr'
    show_diagonal=True,
    show_area=True,
    kws_area: dict = {},
    show_cutoff=True,
    plot_pr=True,
    color="k",
    returns=["ax"],
    ax=None,
):
    """
    Obtain threshold based on ROC or PR curve.

    Returns:
        Table:
            columns: values
                method: ROC, PR
                variable: threshold (index), TPR, FPR, TP counts, precision, recall
                values:
        Plots:
            AUC ROC,
            TPR vs TP counts
            PR
            Specificity vs TP counts
        Dictionary:
            Thresholds from AUC, PR

    TODOs:
        1. Separate the plotting functions.
    """
    if all((y_score) <= 0):
        negative_values = True
    elif any((y_score) < 0):
        raise ValueError("y_score should be all >=0 or <=0")
    else:
        negative_values = False
    if method.lower().startswith("roc"):
        columns_value = ["FPR (1-specificity)", "TPR (sensitivity)"]
        method = "roc_curve"
    elif method.lower().startswith("pr"):
        columns_value = ["precision", "recall"]
        method = "precision_recall_curve"
    else:
        raise ValueError(method)
    from sklearn import metrics

    df1 = pd.DataFrame(
        getattr(metrics, method)(
            list(y_true),
            y_score * (1 if not negative_values else -1),
        ),
        index=columns_value + ["threshold"],
    ).T

    if df1["threshold"].nunique() > 100:
        df1["threshold"] = (
            pd.cut(x=df1["threshold"], bins=100).apply(lambda x: x.mid).astype(float)
        )
        logging.warning(
            f"number of thresholds reduced from {df1['threshold'].nunique()} to 100"
        )
    df1["count TP"] = df1["threshold"].map(
        {i: sum(y_score > i) for i in df1["threshold"].unique()}
    )
    if show_cutoff != False:  # noqa
        df1 = df1.reset_index(drop=True)
        if method == "roc_curve":
            show_cutoff = {} if show_cutoff == True else show_cutoff  ##noqa

            ratios = df1[columns_value[1]] / (df1[columns_value[0]] + 0.01)
            if show_cutoff["maximize"].lower() in ["specificity", "tpr"]:
                ratio = ratios.max()
                cutoff_index = np.where(ratios == ratio)[0][0]
            else:
                # TODOs: test for sensitivity
                raise ValueError(show_cutoff["maximize"])
            cutoff = df1.iloc[cutoff_index, :]  # .iloc[0,:]
        else:
            df1 = df1.loc[(df1["recall"] != 0), :]
            # max pr
            cutoff = df1.loc[
                (df1[columns_value[0]] == df1[columns_value[0]].max()), :
            ].iloc[0, :]
            columns_value = columns_value[::-1]
    ## plot
    for xaxis in [columns_value[0]] + (["count TP"] if plot_pr else []):
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=[2, 2])
        ## line
        ax.plot(
            df1[xaxis],
            df1[columns_value[1]],
            color=color,
            zorder=2,
            solid_capstyle="butt",
            clip_on=False,
        )
        ## area for ROC-AUC
        if show_area != False and xaxis == columns_value[0] and method == "roc_curve":  # noqa
            ax.fill_between(
                df1[xaxis],
                df1[columns_value[1]],
                zorder=1,
                **kws_area,
            )
            auc = metrics.roc_auc_score(
                y_true=list(y_true),
                y_score=y_score,
            )
            ax.text(x=1, y=0, s="AUC=" + f"{auc:.2f}", ha="right", va="bottom")
            if show_diagonal != False and xaxis == columns_value[0]:  # noqa
                ax.plot(
                    [0, 1],
                    [0, 1],
                    ":",
                    color="gray",
                    zorder=1,
                )
                ax.set(
                    xticks=[0, 1],
                    yticks=[0, 1],
                )
        ## mark threshold
        if show_cutoff != False:  # noqa
            ax.scatter(
                [cutoff[xaxis]],
                [cutoff[columns_value[1]]],
                ec="k",
                fc="none",
                zorder=2,
            )
            ax.annotate(
                text=f"threshold={cutoff['threshold']:.2f}",
                xy=[cutoff[xaxis], cutoff[columns_value[1]]],
                xytext=[0.5, 1.1],
                ha="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color="k",
                    # relpos=(0,0.5),
                    shrinkA=0,
                    connectionstyle="arc3,rad=0.3",
                ),
                xycoords="data",
                textcoords="axes fraction",
            )
        ax.set(
            xlabel=xaxis,  #
            ylabel=columns_value[1],
        )
    d_ = {}
    for k in returns:
        d_[k] = locals()[k if k != "data" else "df1"]
    return d_
