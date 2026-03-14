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

def flatten_keys(
    stats,
    sep=':',
    clean=False,
    ): 
    from roux.lib.sys import to_path
    stats={to_path(k) if not isinstance(k,tuple) else sep.join([to_path(str(s)) for s in k]) :v for k,v in stats.items()}
    if clean:
        ## remove _s
        stats={k.replace(f'_{sep}',sep).replace(f'{sep}_',sep):v for k,v in stats.items()}
    return stats

def flatten_vals(d: dict, parent_key: str = '', sep: str = ':') -> dict:
    """
    Flatten a nested dictionary using a specified separator.
    
    Parameters:
        d (dict): Nested dictionary to flatten.
        parent_key (str): Prefix for the current level.
        sep (str): Separator between nested keys.
        
    Returns:
        dict: Flattened dictionary.
    """
    if isinstance(d,pd.DataFrame):
        df_=d
        if len(df_)==1:
            d=df_.iloc[0,:].to_dict()
        else:
            # d=df_.set_index(df_.rd.infer_index()).T.to_dict()
            d=pd.json_normalize(
                df_.rd.clean().set_index(df_.rd.infer_index()).T.to_dict(),
                sep=':',
            ).iloc[0,:].to_dict()

    assert isinstance(d,dict), d
    items = []
    for k, v in d.items():
        # g: construct the new key with the separator if a parent key exists
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # g: recursively flatten if the value is a dictionary
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_dict(d,sep=':'):
    d_flat=flatten_vals(d,sep=sep)
    # size=len(d_flat)
    _keys=list(d_flat.keys())
    
    d_flat=flatten_keys(d_flat,sep=sep)
    # assert len(d_flat)==size,(len(d_flat),size)
    assert len(d_flat)==len(_keys),f"\n{list(d_flat.keys())}\n{_keys}"
    
    return d_flat

## find
def contains_keys(
    obj,
    keys_to_find
    ):
    found = set()

    def recurse(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in keys_to_find:
                    found.add(k)
                recurse(v)
        elif isinstance(o, list):
            for item in o:
                recurse(item)

    recurse(obj)
    return all(k in found for k in keys_to_find)

def diff_dicts(
    d1, # old 
    d2, # new
    path="",
    ignore=[]
    ):
    """Recursively find differences between two dictionaries."""
    for k in d1:
        if k not in d2: # and k not in ignore:
            print(f"{path} ! : '{k}' not in d2")
        else:
            if isinstance(d1[k], dict) and isinstance(d2[k], dict):
                diff_dicts(d1[k], d2[k], f"{path} : {k}" if path else k, ignore=ignore)
            elif d1[k] != d2[k]:
                current_path = f"{path} : {k}" if path else k
                print(f"{current_path}:")
                print(f" - : {d1[k]}")
                print(f" + : {d2[k]}")

    for k in d2:
        if k not in d1: # and k not in ignore:
            print(f"{path} ! : '{k}' not in d1")