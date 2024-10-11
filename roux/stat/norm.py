"""For normalisation."""

import logging

import numpy as np
import pandas as pd

import scipy as sc
from scipy import stats

## vector
## dist shape change, (ranks preserved)
def to_norm(
    x,
    off=1e-5, ## for 0 and 1 values, to prevent inf/-inf in the output
    ):
    """
    Normalise a vector bounded between 0 and 1.
    """
    import numpy as np
    from scipy.stats import norm
        
    assert x.min()>=0, 'values < 0 found'
    if x.min()==0:
        assert not any((x > 0) & (x < off)), 'need to decrease the off and retry'
        x[x == 0] = off
    assert x.max()<=1, 'values > 1 found'
    if x.max()==1:
        assert not any((x > 1-off) & (x < 1)), 'need to decrease the off and retry'
        x[x == 1] = 1-off

    # transformation
    xt = norm.ppf(x)
    assert not any(np.isinf(x)), "inf values found in the output"    
    
    return xt
    
## array
## variance normalization
def norm_by_quantile(X: np.array) -> np.array:
    """Quantile normalize the columns of X.

    Params:
        X : 2D array of float, shape (M, N). The input data, with M rows (genes/features) and N columns (samples).

    Returns:
        Xn : 2D array of float, shape (M, N). The normalized data.

    Notes:
        Faster processing (~5 times compared to other function tested) because of the use of numpy arrays.

    TODOs:
        Use `from sklearn.preprocessing import QuantileTransformer` with `output_distribution` parameter allowing rescaling back to the same distribution kind.
    """

    # compute the quantiles
    quantiles = np.mean(np.sort(X, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 1, the
    # second-smallest by 2, ..., and the largest by M, the number of rows.
    ranks = np.apply_along_axis(stats.rankdata, 0, X)

    # convert ranks to integer indices from 0 to M-1
    rank_indices = ranks.astype(int) - 1

    # index the quantiles for each rank with the ranks matrix
    Xn = quantiles[rank_indices]

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(Xn, columns=X.columns, index=X.index)
    else:
        return Xn


def norm_by_gaussian_kde(values: np.array) -> np.array:
    """Normalise matrix by gaussian KDE.

    Args:
        values (np.array): input matrix.

    Returns:
        np.array: output matrix.

    References:
        https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """
    kernel = stats.gaussian_kde(values)
    return pd.Series(
        {
            k: np.log(
                kernel.integrate_box_1d(-1e4, v) / kernel.integrate_box_1d(v, 1e4)
            )
            for k, v in values.to_dict().items()
        }
    )


## z-scores
def zscore(
    df: pd.DataFrame,
    cols: list = None,
) -> pd.DataFrame:
    """Z-score.

    Args:
        df (pd.DataFrame): input table.

    Returns:
        pd.DataFrame: output table.

    TODOs:
        1. Use scipy or sklearn's zscore because of it's additional options
            from scipy.stats import zscore
            df.apply(zscore)
    """
    if isinstance(df, pd.Series):
        logging.warning("isinstance(df, pd.Series)==True")
        return (df - df.mean()) / df.std()
    else:
        if cols is None:
            cols = df.columns
        return (df[cols] - df[cols].mean()) / df[cols].std()


def zscore_robust(a: np.array) -> np.array:
    """Robust Z-score.

    Args:
        a (np.array): input data.

    Returns:
        np.array: output.

    Example:
        t = sc.stats.norm.rvs(size=100, scale=1, random_state=123456)
        plt.hist(t,bins=40)
        plt.hist(apply_zscore_robust(t),bins=40)
        print(np.median(t),np.median(apply_zscore_robust(t)))
    """

    def get_zscore_robust(x, median, mad):
        return (x - median) / (mad * 1.4826)

    median = np.median(a)
    mad = sc.stats.median_abs_deviation(a)
    if mad == 0:
        logging.error("mad==0")
    return [get_zscore_robust(x, median, mad) for x in a]


## co-variance normalization
def norm_covariance_PCA(
    X: np.array,
    use_svd: bool = True,
    use_sklearn: bool = True,
    rescale_centered: bool = True,
    random_state: int = 0,
    test: bool = False,
    verbose: bool = False,
) -> np.array:
    """Covariance normalization by PCA whitening.

    Args:
        X (np.array): input array
        use_svd (bool, optional): use SVD method. Defaults to True.
        use_sklearn (bool, optional): use `skelearn` for SVD method. Defaults to True.
        rescale_centered (bool, optional): rescale to centered input. Defaults to True.
        random_state (int, optional): random state. Defaults to 0.
        test (bool, optional): test mode. Defaults to False.
        verbose (bool, optional): verbose. Defaults to False.

    Returns:
        np.array: transformed data.
    """
    if test:
        verbose = True
    np.random.seed(random_state)
    if verbose:
        logging.info(
            f"Covariance of the original data: {np.cov(X.T).round(2).tolist()}"
        )
    if test:
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(1, 2, figsize=[6, 3])
        from roux.viz.scatter import plot_scatter

        plot_scatter(
            data=pd.DataFrame(X).rename(
                columns={i: f"{i}" for i in range(2)}, errors="raise"
            ),
            x="0",
            y="1",
            stat_method="pearson",
            ax=axs[0],
        ).set(title="raw")

    X_centered = X - np.mean(X, axis=0)
    if verbose:
        logging.info(
            f"Covariance of the centered data: {np.cov(X_centered.T).round(2).tolist()}"
        )

    if use_svd:
        if use_sklearn:
            from sklearn.decomposition import PCA

            X_transformed = PCA(
                n_components=X.shape[1],
                whiten=True,
                svd_solver="full",
                random_state=random_state,
            ).fit_transform(X_centered)
        else:
            cov = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
            eigen_values, eigen_vectors, _ = np.linalg.svd(cov)
            mult = np.dot(np.diag(1.0 / np.sqrt(eigen_vectors + 1e-5)), eigen_values.T)
            if verbose:
                logging.info(
                    f"Covariance of the mult data: {np.cov(mult.T).round(2).tolist()}"
                )
            X_transformed = np.dot(X_centered, mult.T)
    else:
        cov = np.cov(X_centered.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        if rescale_centered == True:  # noqa
            X_decorrelated = X_centered.dot(eigen_vectors)
        elif rescale_centered == False:  # noqa
            X_decorrelated = X.dot(eigen_vectors)
        X_transformed = X_decorrelated / np.sqrt(eigen_values + 1e-5)
        if verbose:
            logging.info(
                f"Covariance of the X_decorrelated data: {np.cov(X_decorrelated.T).round(2).tolist()}"
            )

    if verbose:
        logging.info(
            f"Covariance of the white data: {np.cov(X_transformed.T).round(2).tolist()}"
        )
    assert X_transformed.shape == X.shape
    assert np.allclose(np.cov(X_transformed.T).round(2), np.identity(X.shape[1]))

    if test:
        plot_scatter(
            data=pd.DataFrame(X_transformed).rename(
                columns={i: f"{i}" for i in range(2)}, errors="raise"
            ),
            x="0",
            y="1",
            stat_method="pearson",
            ax=axs[1],
        ).set(title="transformed")
    return X_transformed
