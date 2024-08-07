"""For fitting data."""

import numpy as np
import pandas as pd
import roux.lib.dfs as rd  # noqa
import matplotlib.pyplot as plt

import scipy as sc


def fit_curve_fit(
    func,
    xdata: np.array = None,
    ydata: np.array = None,
    bounds: tuple = (-np.inf, np.inf),
    test=False,
    plot=False,
) -> tuple:
    """Wrapper around `scipy`'s `curve_fit`.

    Args:
        func (function): fitting function.
        xdata (np.array, optional): x data. Defaults to None.
        ydata (np.array, optional): y data. Defaults to None.
        bounds (tuple, optional): bounds. Defaults to (-np.inf, np.inf).
        test (bool, optional): test. Defaults to False.
        plot (bool, optional): plot. Defaults to False.

    Returns:
        tuple: output.
    """
    from scipy.optimize import curve_fit

    # Define the data to be fit with some noise:
    if xdata is None and ydata is None:
        xdata = np.linspace(1, 4, 50)
        y = func(xdata, -2.5)
        np.random.seed(1729)
        y_noise = 0.2 * np.random.normal(size=xdata.size)
        ydata = y + y_noise
    if test or plot:
        plt.plot(xdata, ydata, "b.", label="data")
        # Fit for the parameters a, b, c of the function func:

    popt, pcov = curve_fit(func, xdata, ydata)
    if test or plot:
        print(popt)
        plt.plot(
            xdata, func(xdata, *popt), "r-", label=f"non-bounded fit:\na={popt[0]}"
        )
    # Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
    popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds)

    if test or plot:
        print(popt)
        plt.plot(xdata, func(xdata, *popt), "g--", label=f"bounded fit:\na={popt[0]}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()
    return func(xdata, *popt), popt


def fit_gauss_bimodal(
    data: np.array,
    bins: int = 50,
    expected: tuple = (1, 0.2, 250, 2, 0.2, 125),
    test=False,
) -> tuple:
    """Fit bimodal gaussian distribution to the data in vector format.

    Args:
        data (np.array): vector.
        bins (int, optional): bins. Defaults to 50.
        expected (tuple, optional): expected parameters. Defaults to (1,.2,250,2,.2,125).
        test (bool, optional): test. Defaults to False.

    Returns:
        tuple: _description_

    Notes:
        Observed better performance with `roux.stat.cluster.cluster_1d`.
    """
    from scipy.optimize import curve_fit

    def gauss(x, mu, sigma, A):
        return A * np.exp(-((x - mu) ** 2) / 2 / sigma**2)

    def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
        return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x[:-1] + x[1:]) / 2
    params, cov = curve_fit(bimodal, x, y, expected)
    sigma = np.sqrt(np.diag(cov))
    if test:
        plt.figure()
        _ = plt.hist(data, bins, alpha=0.3, label="data", density=True)
        plt.plot(x, bimodal(x, *params), color="red", lw=3, label="model")
        plt.legend()
    return params, sigma


## 2D
def get_grid(
    x: np.array,
    y: np.array,
    z: np.array = None,
    off: int = 0,
    grids: int = 100,
    method="linear",
    test=False,
    **kws,
) -> tuple:
    """2D grids from 1d data.

    Args:
        x (np.array): vector.
        y (np.array): vector.
        z (np.array, optional): vector. Defaults to None.
        off (int, optional): offsets. Defaults to 0.
        grids (int, optional): grids. Defaults to 100.
        method (str, optional): method. Defaults to 'linear'.
        test (bool, optional): test. Defaults to False.

    Returns:
        tuple: output.
    """
    xoff = (np.max(x) - np.min(x)) * off
    yoff = (np.max(y) - np.min(y)) * off
    xi = np.linspace(np.min(x) - xoff, np.max(x) + xoff, grids)
    yi = np.linspace(np.min(y) - yoff, np.max(y) + yoff, grids)

    X, Y = np.meshgrid(xi, yi)
    if z is None:
        return X, Y
    else:
        Z = sc.interpolate.griddata(
            (x, y),
            z,
            (X, Y),
            method=method,
            fill_value=min(z),
        )
        return X, Y, Z


def fit_gaussian2d(
    x: np.array,
    y: np.array,
    z: np.array,
    grid=True,
    grids=20,
    method="linear",
    off=0,
    rescalez=True,
    test=False,
) -> tuple:
    """Fit gaussian 2D.

    Args:
        x (np.array): vector.
        y (np.array): vector.
        z (np.array): vector.
        grid (bool, optional): grid. Defaults to True.
        grids (int, optional): grids. Defaults to 20.
        method (str, optional): method. Defaults to 'linear'.
        off (int, optional): offsets. Defaults to 0.
        rescalez (bool, optional): rescalez. Defaults to True.
        test (bool, optional): test. Defaults to False.

    Returns:
        tuple: output.
    """
    if grid:
        xg, yg, zg = get_grid(x, y, z, grids=grids, method=method)
    else:
        xg, yg, zg = x, y, z
    from astropy.modeling import models, fitting

    if rescalez:
        from roux.stat.transform import rescale

        range1 = (np.min(zg), np.max(zg))
        zg_ = rescale(a=zg, range1=range1, range2=[0, 1])

        #         from sklearn.preprocessing import MinMaxScaler
        #         scaler = MinMaxScaler()
        #         scaler.fit(zg)
        #         zg_ = scaler.transform(zg)
        if test:
            print(np.min(zg), np.max(zg), zg.shape)
            print(np.min(zg_), np.max(zg_), zg_.shape)
    else:
        zg_ = zg

    m1 = models.Gaussian2D(
        amplitude=np.max(zg_),
        x_mean=np.mean(xg),
        y_mean=np.mean(yg),
        #       x_stddev=np.std(xg), y_stddev=np.std(yg),
        #       theta=0.
        cov_matrix=np.cov(np.vstack([xg.flatten(), yg.flatten()])),
        fixed={"x_mean": True, "y_mean": True},
        bounds={"amplitude": (np.min(zg_), np.max(zg_))},
    )
    #     z = m1(x, y)
    fitr = fitting.LevMarLSQFitter()
    mf1 = fitr(m1, xg, yg, zg_)
    if grid and off != 0:
        xg, yg = get_grid(x, y, grids=grids, method=method, off=off)
    zg_hat_ = mf1(xg, yg)
    print(mf1.theta)
    if rescalez:
        zg_hat = rescale(a=zg_hat_, range1=[0, 1], range2=range1)
        # for inverse transformation
        #         zg_hat = scaler.inverse_transform(zg_hat_)
        if test:
            print(np.min(zg_hat_), np.max(zg_hat_), zg_hat_.shape)
            print(np.min(zg_hat), np.max(zg_hat), zg_hat.shape)
    #         print(range1)
    else:
        zg_hat = zg_hat_
    return xg, yg, zg, zg_hat


def fit_2d_distribution_kde(
    x: np.array,
    y: np.array,
    bandwidth: float,
    xmin: float = None,
    xmax: float = None,
    xbins=100j,
    ymin: float = None,
    ymax: float = None,
    ybins=100j,
    test=False,
    **kwargs,
) -> tuple:
    """
    2D kernel density estimate (KDE).

    Notes:
        Cut off outliers:
            quantile_coff=0.01
            params_grid=merge_dicts([
            df01.loc[:,var2col.values()].quantile(quantile_coff).rename(index=flip_dict({f"{k}min":var2col[k] for k in var2col})).to_dict(),
            df01.loc[:,var2col.values()].quantile(1-quantile_coff).rename(index=flip_dict({f"{k}max":var2col[k] for k in var2col})).to_dict(),
                    ])

    Args:
        x (np.array): vector.
        y (np.array): vector.
        bandwidth (float): bandwidth
        xmin (float, optional): x minimum. Defaults to None.
        xmax (float, optional): x maximum. Defaults to None.
        xbins (_type_, optional): x bins. Defaults to 100j.
        ymin (float, optional): y minimum. Defaults to None.
        ymax (float, optional): y maximum. Defaults to None.
        ybins (_type_, optional): y bins. Defaults to 100j.
        test (bool, optional): test. Defaults to False.

    Returns:
        tuple: output.
    """
    from sklearn.neighbors import KernelDensity

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[
        x.min() if xmin is None else xmin : x.max() if xmax is None else xmax : xbins,
        y.min() if ymin is None else ymin : y.max() if ymax is None else ymax : ybins,
    ]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    zz = np.reshape(z, xx.shape)
    if test:
        fig = plt.figure(figsize=[5, 4])
        ax = plt.subplot()
        ax.scatter(x, y, s=1, fc="k")
        pc = ax.pcolormesh(xx, yy, zz, cmap="Reds")
        fig.colorbar(pc)
        ax.set(
            **{
                "xlim": [xmin, xmax],
                "ylim": [ymin, ymax],
                "title": f"bandwidth{bandwidth}_bins{xbins}",
            }
        )
    return xx, yy, zz


def check_poly_fit(
    d: pd.DataFrame, xcol: str, ycol: str, degmax: int = 5
) -> pd.DataFrame:
    """Check the fit of a polynomial equations.

    Args:
        d (pd.DataFrame): input dataframe.
        xcol (str): column containing the x values.
        ycol (str): column containing the y values.
        degmax (int, optional): degree maximum. Defaults to 5.

    Returns:
        pd.DataFrame: _description_
    """
    from scipy.stats import linregress

    ns = range(1, degmax + 1, 1)
    plt.figure(figsize=[9, 5])
    ax = plt.subplot(121)
    d.plot.scatter(x=xcol, y=ycol, alpha=0.3, color="gray", ax=ax)
    metrics = pd.DataFrame(index=ns, columns=["r", "standard error"])
    for n in ns:
        fit = np.polyfit(d[xcol], d[ycol], n)
        yp = np.poly1d(fit)
        _, _, r, _, e = linregress(yp(d[xcol]), d[ycol])
        metrics.loc[n, "r"] = r
        metrics.loc[n, "standard error"] = e
        ax.plot(
            d[xcol],
            yp(d[xcol]),
            "-",
            color=plt.get_cmap("hsv")(n / float(max(ns))),
            alpha=1,
            lw="4",
        )
    ax.legend(["degree=%d" % n for n in ns], bbox_to_anchor=(1, 1))

    metrics.index.name = "degree"
    ax = plt.subplot(122)
    ax = metrics.plot.barh(ax=ax)
    ax.legend(bbox_to_anchor=[1, 1])
    #     metrics.plot.barh('standard error')
    plt.tight_layout()
    return metrics


def mlr_2(df: pd.DataFrame, coly: str, colxs: list) -> tuple:
    """Multiple linear regression between two variables.

    Args:
        df (pd.DataFrame): input dataframe.
        coly (str): column  containing y values.
        colxs (list): columns containing x values.

    Returns:
        tuple: output.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    X = poly.fit_transform(df.loc[:, colxs])
    # pd.DataFrame()
    y = df.loc[:, coly].tolist()
    reg = LinearRegression().fit(X, y)
    label_score = f"$r^2$={reg.score(X, y):.2g}"
    label_eqn = (
        "$y$="
        + "".join(
            [
                f"{l[0]:+.2g}*{l[1]}"
                for l in zip(reg.coef_, ["$x_1$", "$x_2$", "$x_1$*$x_2$"])
            ]
        )[1:]
    )
    dplot = pd.DataFrame(
        {
            f"{coly}": y,
            f"{coly} predicted": reg.predict(X),
        }
    )
    return label_score, label_eqn, dplot


def get_mlr_2_str(df: pd.DataFrame, coly: str, colxs: list) -> str:
    """Get the result of the multiple linear regression between two variables as a string.

    Args:
        df (pd.DataFrame): input dataframe.
        coly (str): column  containing y values.
        colxs (list): columns containing x values.

    Returns:
        str: output.
    """
    label_score, label_eqn, dplot_reg = mlr_2(df, coly, colxs)
    label_eqn = (
        label_eqn.replace("$y$", "$z$").replace("$x_1$", "$x$").replace("$x_2$", "$y$")
    )
    return f"{label_eqn} ({label_score})"
