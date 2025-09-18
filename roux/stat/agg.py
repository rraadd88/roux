import numpy as np
from typing import List, Union

# Define types for clarity
ArrayLike = Union[List[float], np.ndarray]

def agg_means(n: ArrayLike, means: ArrayLike) -> float:
    """
    Combines means of multiple groups, weighted by their sample sizes.
    """
    # --- Validation ---
    try:
        n = np.array(n, dtype=float)
        means = np.array(means, dtype=float)
    except ValueError:
        raise TypeError("Inputs 'n' and 'means' must be numeric and convertible to NumPy arrays.")

    if n.shape != means.shape:
        raise ValueError("Input arrays 'n' and 'means' must have the same shape.")
    if n.size == 0:
        return np.nan # Or raise ValueError("Input arrays cannot be empty.")
    if np.any(n < 0):
        raise ValueError("Sample sizes in 'n' cannot be negative.")
        
    # --- Calculation ---
    total_n = np.sum(n)
    if total_n == 0:
        return np.nan # Mean is undefined if there are no samples

    return np.sum(n * means) / total_n


def agg_vars(n: ArrayLike, means: ArrayLike, std_devs: ArrayLike) -> float:
    """
    Combines the variance of multiple groups with robust validation.
    """
    # --- Validation ---
    try:
        n = np.array(n, dtype=float)
        means = np.array(means, dtype=float)
        std_devs = np.array(std_devs, dtype=float)
    except ValueError:
        raise TypeError("All inputs must be numeric and convertible to NumPy arrays.")

    if not (n.shape == means.shape == std_devs.shape):
        raise ValueError("All input arrays ('n', 'means', 'std_devs') must have the same shape.")
    if n.size == 0:
        return np.nan # Or raise ValueError("Input arrays cannot be empty.")
    if np.any(n < 1):
        raise ValueError("All sample sizes in 'n' must be 1 or greater.")
    if np.any(std_devs < 0):
        raise ValueError("Standard deviations in 'std_devs' cannot be negative.")

    # --- Calculation ---
    total_n = np.sum(n)
    if total_n <= 1:
        # Combined variance is undefined for a single or zero total samples.
        return np.nan

    combined_mean = agg_means(n, means)
    if np.isnan(combined_mean): # Propagate NaN if means are undefined
        return np.nan

    # Calculate variances, handling the n=1 case where std_dev might be NaN
    # A group of size 1 has 0 variance, regardless of the input std_dev value.
    variances = std_devs**2
    variances[n == 1] = 0
    
    # Sum of squared deviations within groups
    within_group_variance = np.sum((n - 1) * variances)

    # Sum of squared deviations between groups
    between_group_variance = np.sum(n * (means - combined_mean)**2)

    return (within_group_variance + between_group_variance) / (total_n - 1)


def agg_stds(n: ArrayLike, means: ArrayLike, std_devs: ArrayLike) -> float:
    """
    Combines the standard deviations of multiple groups with robust validation.
    """
    combined_var = agg_vars(n, means, std_devs)
    if np.isnan(combined_var) or combined_var < 0:
        return np.nan
    return np.sqrt(combined_var)