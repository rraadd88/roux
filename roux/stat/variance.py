import numpy as np

def confidence_interval_95(x: np.array) -> float:
    """95% confidence interval.

    Args:
        x (np.array): input vector.

    Returns:
        float: output.
    """
    return 1.96*np.std(x)/np.sqrt(len(x))