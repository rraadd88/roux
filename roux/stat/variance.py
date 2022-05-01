import numpy as np

def confidence_interval_95(x: np.array) -> float:
    """95% confidence interval.

    Args:
        x (np.array): input vector.

    Returns:
        float: output.
    """
    return 1.96*np.std(x)/np.sqrt(len(x))

def get_ci(rs,ci_type,outstr=False):
    if ci_type.lower()=='max':
        ci=max([abs(r-np.mean(rs)) for r in rs])
    elif ci_type.lower()=='sd':
        ci=np.std(rs)
    elif ci_type.lower()=='ci':
        ci=confidence_interval_95(rs)
    else:
        raise ValueError("ci_type invalid")
        return
    if not outstr:
        return ci
    else:
        return "$\pm${ci:.2f}{ci_type if ci_type!='max' else ''}"