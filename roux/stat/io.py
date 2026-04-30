"""For input/output of stats."""

import numpy as np
import pandas as pd


# stats
def perc_label(a, b=None, bracket=True):
    from roux.lib.str import num2str

    if b is None:
        b = len(a)
        a = sum(a)
    ratio = a / b if b != 0 else np.nan
    return f"{(ratio)*100:.1f}%" + (f" ({num2str(a)}/{num2str(b)})" if bracket else "")


def pval2annot(
    pval: float,
    alternative: str = None,
    alpha: float = 0.05,
    fmt: str = None,
    power: bool = True,
    linebreak: bool = False,
    replace_prefix: str = None,
    pmin=1e-10,
):
    """
    P/Q-value to annotation.

    Parameters:
        fmt (str): 
            None: values without decimals, upto pmin
            *:
            <:

    Notes:
        Test:
        
        %run io.py
        for p in [
            1,
            0.1,
            0.05,
            0.06,
            1e-1,1e-2,1e-3,1e-4,
            2e-9,3e-10,5.239847e-11
        ]:
            print(
                p,
                pval2annot(
                    p,
                    fmt=None,
                )
            )
        
        1 $p$=1
        0.1 $p$=0.1
        0.05 $p$=0.05
        0.06 $p$=0.06
        0.1 $p$=0.1
        0.01 $p$=0.01
        0.001 $p$=0.001
        0.0001 $p$=0.0001
        2e-09 $p$=2x$10^{-9}$
        3e-10 $p$=3x$10^{-10}$
        5.239847e-11 $p$<5x$10^{-11}$
    """
    if alternative is None and alpha is None:
        raise ValueError("both alternative and alpha are None")
    if pd.isnull(pval):
        annot = ""
    elif pval < pmin:
        annot = (
            "****"
                if fmt == "*" else 
            f"$p$<\n{pmin:.0e}"
                if fmt == "<" else 
            f"$p$=\n{pval:.1g}"
                if len(f"$p$={pval:.1g}") >= 6 and linebreak else 
            f"$p$<{pmin:.1g}"
        )
    elif pval < 0.001:
        annot = (
            "***"
                if fmt == "*" else 
            f"$p$<\n{0.001:.0e}"
                if fmt == "<" else 
            f"$p$=\n{pval:.1g}"
                if len(f"$p$={pval:.1g}") >= 6 and linebreak else 
            f"$p$={pval:.1g}"
        )
    elif pval < 0.01:
        annot = (
            "**"
                if fmt == "*" else 
            "$p$<\n0.01"
                if fmt == "<" else 
            f"$p$=\n{pval:.1g}"
                if len(f"$p$={pval:.1g}") >= 6 and linebreak else 
            f"$p$={pval:.1g}"
        )
    elif pval < alpha:
        annot = (
            "*"
                if fmt == "*" else 
            f"$p$<\n{alpha}"
                if fmt == "<" else 
            f"$p$=\n{pval:.1g}"
                if len(f"$p$={pval:.1g}") >= 6 and linebreak else 
            f"$p$={pval:.1g}"
        )
    else:
        annot = (
            "ns"
                if fmt == "*" else 
            f"$p$=\n{pval:.1g}"
                if len(f"$p$={pval:.1g}") >= 6 and linebreak else 
            f"$p$={pval:.1g}"
        )
    annot = annot if linebreak else annot.replace("\n", "")
    if replace_prefix is not None:
        annot = annot.replace("$p$", replace_prefix)
    if power and "e" in annot:
        annot = annot.replace("e-0", "e-").replace("e", "x$10^{") + "}$"
    return annot
