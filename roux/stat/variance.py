"""For variance related stats."""

import numpy as np
import pandas as pd
import scipy.stats as sc

def confidence_interval_95(x: np.array) -> float:
    """95% confidence interval.

    Args:
        x (np.array): input vector.

    Returns:
        float: output.
    """
    return 1.96 * np.std(x) / np.sqrt(len(x))


def get_ci(rs, ci_type, outstr=False):
    if ci_type.lower() == "max":
        if np.isfinite(rs).all():
            ci = max([abs(r - np.mean(rs)) for r in rs])
        else:
            ci = None
    elif ci_type.lower() == "sd":
        ci = np.std(rs)
    elif ci_type.lower() == "ci":
        ci = confidence_interval_95(rs)
    else:
        raise ValueError("ci_type invalid")
        return
    if not outstr:
        return ci
    else:
        return "$\pm${ci:.2f}{ci_type if ci_type!='max' else ''}"


def get_variance_inflation(
    data,
    coly: str,
    cols_x: list = None,
):
    """
    Variance Inflation Factor (VIF). A wrapper around `statsmodels`'s '`variance_inflation_factor` function.

    Parameters:
        data (pd.DataFrame): input data.
        coly (str): dependent variable.
        cols_x (list): independent variables.

    Returns:
        pd.Series
    """
    from patsy import dmatrices
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if cols_x is None:
        cols_x = list(set(data.columns) - set([coly]))
    from roux.lib.str import replace_many, to_formula
    from roux.lib.df import renameby_replace

    df1 = renameby_replace(data, to_formula())
    # print(df1.columns)
    # print(f'{replace_many(coly,to_formula())} ~' + "+".join(cols_x))
    # design matrix
    y, X = dmatrices(
        f"{replace_many(coly,to_formula(),ignore=True)} ~"
        + "+".join([replace_many(s, to_formula(), ignore=True) for s in cols_x]),
        df1,
        return_type="dataframe",
    )
    return (
        pd.Series({k: variance_inflation_factor(X.values, i) for i, k in enumerate(X)})
        .to_frame("VIF")
        .rename_axis(["variable"], axis=0)
        .reset_index()
        .assign(
            **{
                "variable": lambda df: df["variable"].apply(
                    lambda x: replace_many(x, to_formula(reverse=True), ignore=True)
                    if x != "Intercept"
                    else "Intercept"
                ),
            },
        )
    )

def get_complexity(
    df: pd.DataFrame,
    method=[
        # 'entropy', ## not comparable across different ns
        'norm-entropy', 
        'gini',
        # 'perplexity', ## not comparable across different ns
    ]
) -> pd.DataFrame: 
    """
    Calculates complexity scores for a DataFrame of probability vectors.

    Args:
        df (pd.DataFrame): An m x n DataFrame where each row is a
                              probability vector (sums to 1).

    Returns:
        pd.DataFrame: A DataFrame with m rows and columns for each
                      complexity score.

    Notes:
        Specificity (Tau) ≈ 1 - Uncertainty (Entropy/Gini)
    """
    if isinstance(method,str): 
        method=[method]
        
    # 1. Extract the underlying NumPy array for calculation
    P = df.values
    m, n = P.shape

    scores={}
    if 'entropy' in method or 'norm-entropy' in method or 'perplexity' in method:
        scores['entropy'] = sc.entropy(P, axis=1, base=2)

    if 'norm-entropy' in method:
        # The calculation logic is the same
        # 2. Normalize the entropy to a [0, 1] scale
        # Handle the edge case where n=1 to avoid division by zero (log2(1)=0)
        max_entropy = np.log2(n) if n > 1 else 1
        scores['norm-entropy'] = scores['entropy'] / max_entropy
    
    if 'perplexity' in method:        
        scores['perplexity'] = np.power(2, scores['entropy'])

    
    if 'gini' in method:        
        scores['gini'] = 1 - np.sum(np.power(P, 2), axis=1)
    
    # 2. Create an output DataFrame from the results
    return pd.DataFrame(scores)