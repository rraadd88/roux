"""Dictionary processing"""
from roux.lib.set import *
import yaml
import json
from roux.lib.str import dict2str,str2dict

def head_dict(d, lines=5):
    return dict(itertools.islice(d.items(), lines))

def sort_dict(d1,by=1,ascending=True):
    """Sort dictionary by values.
    
    Parameters:
        d1 (dict): input dictionary.
        by (int): index of the value among the values.
        ascending (bool): ascending order.
        
    Returns:
        d1 (dict): output dictionary.
    """
    return dict(sorted(d1.items(), key=lambda item: item[1],reverse=not ascending))

def merge_dicts(
    l: list,
    )-> dict:
    """Merge dictionaries.
    
    Parameters:
        l (list): list containing the dictionaries.
        
    Returns:
        d (dict): output dictionary.
        
    TODOs:
        in python>=3.9, `merged = d1 | d2`
    """    
    from collections import ChainMap
    return dict(ChainMap(*l))

def merge_dict_values(l,test=False):
    """Merge dictionary values.
    
    Parameters:
        l (list): list containing the dictionaries.
        test (bool): verbose.
        
    Returns:
        d (dict): output dictionary.
    """
    for di,d_ in enumerate(l):
        if di==0:
            d=d_
        else:
            d={k:d[k]+d_[k] for k in d}
        if test:
            print(','.join([str(len(d[k])) for k in d]))
    return d        
        
def flip_dict(d):
    """switch values with keys and vice versa.
    
    Parameters:
        d (dict): input dictionary.
        
    Returns:
        d (dict): output dictionary.
    """
    if all([not isinstance(s,list) for s in d.values()]):
        if len(np.unique(d.keys()))!=len(np.unique(d.values())):
            logging.error('values should be unique to flip the dict')
            return
        else:
            return {d[k]:k for k in d}
    else:
        if not get_offdiagonal_values(intersections(d)).sum().sum()==0:
            raise ValueError('dict values should be mutually exclusive') 
        d_={}
        for k in d:
            for v in d[k]:
                d_[v]=k
        return d_
