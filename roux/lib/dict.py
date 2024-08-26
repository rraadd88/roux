"""For processing dictionaries."""

from roux.lib.set import itertools, logging, np, pd
from roux.lib.str import dict2str, str2dict  # noqa


def head_dict(d, lines=5):
    return dict(itertools.islice(d.items(), lines))


def sort_dict(d1, by=1, ascending=True):
    """Sort dictionary by values.

    Parameters:
        d1 (dict): input dictionary.
        by (int): index of the value among the values.
        ascending (bool): ascending order.

    Returns:
        d1 (dict): output dictionary.
    """
    return dict(sorted(d1.items(), key=lambda item: item[1], reverse=not ascending))


def merge_dicts(
    l: list,  # noqa
) -> dict:
    """Merge dictionaries.

    Parameters:
        l (list): list containing the dictionaries.

    Returns:
        d (dict): output dictionary.

    TODOs:
        1. In python>=3.9, `merged = d1 | d2`?
    """
    from collections import ChainMap

    return dict(ChainMap(*l))


def merge_dicts_deep(left: dict, right: dict) -> dict:
    """
    Merge nested dictionaries. Overwrites left with right.

    Parameters:
        left (dict): dictionary #1
        right (dict): dictionary #2

    TODOs:
        1. In python>=3.9, `merged = d1 | d2`?
    """
    # logging.warning("`merge_dicts_deep` is under development.")
    from copy import deepcopy

    d1 = deepcopy(left)
    for k2, v2 in right.items():
        v1 = d1.get(k2)
        if isinstance(v1, dict) and isinstance(v2, dict):
            d1[k2] = merge_dicts_deep(v1, v2)  # recursive
        else:
            d1[k2] = deepcopy(v2)  # overwrite left with right
    return d1


def merge_dict_values(
    l,  # noqa
    test=False,
):
    """Merge dictionary values.
    Parameters:
        l (list): list containing the dictionaries.
        test (bool): verbose.

    Returns:
        d (dict): output dictionary.
    """
    for di, d_ in enumerate(l):
        if di == 0:
            d = d_
        else:
            d = {k: d[k] + d_[k] for k in d}
        if test:
            print(",".join([str(len(d[k])) for k in d]))
    return d


def flip_dict(d):
    """switch values with keys and vice versa.

    Parameters:
        d (dict): input dictionary.

    Returns:
        d (dict): output dictionary.
    """
    if all([not isinstance(s, list) for s in d.values()]):
        if len(np.unique(list(d.keys()))) != len(np.unique(list(d.values()))):
            logging.warning("values are list.")
            return pd.Series(d).reset_index().groupby(0)["index"].agg(list).to_dict()
        else:
            return {d[k]: k for k in d}
    # else:
    #     if not get_offdiagonal_values(intersections(d)).sum().sum()==0:
    #         raise ValueError('dict values should be mutually exclusive.')
    #     d_={}
    #     for k in d:
    #         for v in d[k]:
    #             d_[v]=k
    #     return d_
