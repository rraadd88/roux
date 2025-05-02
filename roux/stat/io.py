"""For input/output of stats."""

import pandas as pd


# stats
def perc_label(a, b=None, bracket=True):
    from roux.lib.str import num2str

    if b is None:
        b = len(a)
        a = sum(a)
    ratio = a / b if b != 0 else None
    return f"{(ratio)*100:.1f}%" + (f" ({num2str(a)}/{num2str(b)})" if bracket else "")


def pval2annot(
    pval: float,
    alternative: str = None,
    alpha: float = 0.05,
    fmt: str = "*",
    power: bool = True,
    linebreak: bool = False,
    replace_prefix: str = None,
):
    """
    P/Q-value to annotation.

    Parameters:
        fmt (str): *|<|'num'

    """
    if alternative is None and alpha is None:
        raise ValueError("both alternative and alpha are None")
    if pd.isnull(pval):
        annot = ""
    elif pval < 0.0001:
        annot = (
            "****"
            if fmt == "*"
            else f"$p$<\n{0.0001:.0e}"
            if fmt == "<"
            else f"$p$={pval:.1g}"
            if len(f"$p$={pval:.1g}") < 6
            else f"$p$=\n{pval:.1g}"
            if not linebreak
            else f"$p$={pval:.1g}"
        )
    elif pval < 0.001:
        annot = (
            "***"
            if fmt == "*"
            else f"$p$<\n{0.001:.0e}"
            if fmt == "<"
            else f"$p$={pval:.1g}"
            if len(f"$p$={pval:.1g}") < 6
            else f"$p$=\n{pval:.1g}"
            if not linebreak
            else f"$p$={pval:.1g}"
        )
    elif pval < 0.01:
        annot = (
            "**"
            if fmt == "*"
            else "$p$<\n0.01"
            if fmt == "<"
            else f"$p$={pval:.1g}"
            if len(f"$p$={pval:.1g}") < 6
            else f"$p$=\n{pval:.1g}"
            if not linebreak
            else f"$p$={pval:.1g}"
        )
    elif pval < alpha:
        annot = (
            "*"
            if fmt == "*"
            else f"$p$<\n{alpha}"
            if fmt == "<"
            else f"$p$={pval:.1g}"
            if len(f"$p$={pval:.1g}") < 6
            else f"$p$=\n{pval:.1g}"
            if not linebreak
            else f"$p$={pval:.1g}"
        )
    else:
        annot = (
            "ns"
            if fmt == "*"
            else f"$p$={pval:.1g}"
            if len(f"$p$={pval:.1g}") < 6
            else f"$p$=\n{pval:.1g}"
            if not linebreak
            else f"$p$={pval:.1g}"
        )
    annot = annot if linebreak else annot.replace("\n", "")
    if replace_prefix is not None:
        annot = annot.replace("$p$", replace_prefix)
    if power and "e" in annot:
        annot = annot.replace("e-0", "e-").replace("e", "x$10^{") + "}$"
    return annot
