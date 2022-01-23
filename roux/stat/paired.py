# import pandas as pd
from roux.global_imports import *

def get_stats_paired(df1,cols,
                     input_logscale,
                     prefix=None,
                     drop_cols=False,
                     unidirectional_stats=['min','max'],
                     fast=False):
    assert(len(cols)==2)
    if prefix is None:
        prefix=get_fix(*cols,common=True,clean=True)
        info(prefix)
    from roux.stat.diff import get_ratio_sorted,get_diff_sorted
    df1[f"{prefix} {'ratio' if not input_logscale else 'diff'}"]=getattr(df1,'parallel_apply' if fast else "apply")(lambda x: (get_ratio_sorted if not input_logscale else get_diff_sorted)(x[cols[0]],
                                                            x[cols[1]]),
                                                            axis=1)
    assert(not any(df1[f"{prefix} {'ratio' if not input_logscale else 'diff'}"]<0))
    for k in unidirectional_stats:
        df1[f'{prefix} {k}']=getattr(df1.loc[:,cols],k)(axis=1)
    if drop_cols:
        df1=df1.log.drop(labels=cols,axis=1)
    return df1