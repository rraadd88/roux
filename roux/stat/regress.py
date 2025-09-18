import numpy as np

def regress_out(a, b, jitter=False):
    """
    Regresses b from a using the pseudo-inverse, preserving the original mean.
    
    This method is robust to perfect multicollinearity and can handle near-constant
    input arrays by adding optional jitter.

    Parameters
    ----------
    a : np.ndarray
        The dependent variable.
    b : np.ndarray or pd.DataFrame
        The independent variable(s) (covariates).
    jitter : bool, optional
        If True, adds a small amount of random noise to prevent zero variance.
        Defaults to False.

    Returns
    -------
    np.ndarray
        The residuals of a after regressing out b.

    Raises
    ------
    ValueError
        If the input array 'b' has a variance too close to zero, which
        prevents a stable regression calculation.
    """
    # Ensure b is a 2D array for the linear algebra operation
    if b.ndim == 1:
        b = b.reshape(-1, 1)

    # Add jitter if requested to handle near-constant data
    if jitter:
        a = a + np.random.normal(0, 1e-10, size=a.shape)
        b = b + np.random.normal(0, 1e-10, size=b.shape)

    # Check for near-zero variance in 'b' and raise an explicit error
    if np.any(np.std(b, axis=0) < 1e-10):
        raise ValueError("Input data for regression has near-zero variance. "
                         "Cannot perform a stable regression. Try setting jitter=True.")

    # Preserve the original mean
    a_mean = a.mean()
    a_centered = a - a_mean
    b_centered = b - b.mean(axis=0)
    
    # Calculate the regression coefficients using the pseudo-inverse
    coeffs = np.linalg.pinv(b_centered).dot(a_centered)
    
    # Get the predicted values
    a_predicted_centered = b_centered.dot(coeffs)
    
    # The residuals are the difference between actual and predicted
    residuals = a_centered - a_predicted_centered
    
    # Add the original mean back to the residuals
    return residuals + a_mean