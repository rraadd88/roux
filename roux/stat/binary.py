from roux.lib.df import *

def log_likelihood(y_true: list, y_pred: list) -> float:
    """Log likelihood.

    Args:
        y_true (list): True
        y_pred (list): Predicted.

    Returns:
        float: log likelihood

    Reference: 
        1. https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l

def f_statistic(y_true, y_pred, n, p):
    """F-statistic.

    Args:
        y_true (list): True
        y_pred (list): Predicted.

    Returns:
        float: F-statistic

    Reference: 
        1. https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """
    msm = np.power(y_pred - y_true.mean(), 2).sum() / p
    mse = np.power(y_true - y_pred, 2).sum() / (n - p - 1)

    f = msm / mse

    f_pval = stats.f.sf(f, p, n - p - 1)

    return f, f_pval
    
def compare_bools_jaccard(x,y):
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
    dmetrics=pd.DataFrame(index=df.columns.tolist(),columns=df.columns.tolist())
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i>c2i:
                dmetrics.loc[c1,c2]=compare_bools_jaccard(df.dropna(subset=[c1,c2])[c1],df.dropna(subset=[c1,c2])[c2])
            elif c1i==c2i:
                dmetrics.loc[c1,c2]=1
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i<c2i:
                dmetrics.loc[c1,c2]=dmetrics.loc[c2,c1]
    return dmetrics

def classify_bools(l: list) -> str:
    """Classify bools.

    Args:
        l (list): list of bools

    Returns:
        str: classification.
    """
    return 'both' if all(l) else 'either' if any(l) else 'neither'

## agg
def frac(x: list) -> float:
    """Fraction.

    Args:
        x (list): list of bools.

    Returns:
        float: fraction of True values.
    """
    return (sum(x)/len(x))
def perc(x: list) -> float:
    """Percentage.

    Args:
        x (list): list of bools.

    Returns:
        float: Percentage of the True values
    """
    return frac(x)*100

## confusion_matrix
def get_stats_confusion_matrix(df_: pd.DataFrame) -> pd.DataFrame:
    """Get stats confusion matrix.

    Args:
        df_ (DataFrame): Confusion matrix.

    Returns:
        DataFrame: stats.
    """
    d0={}
    d0['TP']=df_.loc[True,True]
    d0['TN']=df_.loc[False,False]
    d0['FP']=df_.loc[False,True]
    d0['FN']=df_.loc[True,False]
    # Sensitivity, hit rate, recall, or true positive rate
    d0['TPR'] = d0['TP']/(d0['TP']+d0['FN'])
    # Specificity or true negative rate
    d0['TNR'] = d0['TN']/(d0['TN']+d0['FP']) 
    # Precision or positive predictive value
    d0['PPV'] = d0['TP']/(d0['TP']+d0['FP'])
    # Negative predictive value
    d0['NPV'] = d0['TN']/(d0['TN']+d0['FN'])
    # Fall out or false positive rate
    d0['FPR'] = d0['FP']/(d0['FP']+d0['TN'])
    # False negative rate
    d0['FNR'] = d0['FN']/(d0['TP']+d0['FN'])
    # False discovery rate
    d0['FDR'] = d0['FP']/(d0['TP']+d0['FP'])
    # Overall accuracy
    d0['ACC'] = (d0['TP']+d0['TN'])/(d0['TP']+d0['FP']+d0['FN']+d0['TN'])
    df1=pd.Series(d0).to_frame('value')
    df1.index.name='variable'
    df1=df1.reset_index()
    return df1