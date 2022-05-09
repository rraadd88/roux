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

def merge_dicts(l):
    """Merge dictionaries.
    
    Parameters:
        l (list): list containing the dictionaries.
        
    Returns:
        d (dict): output dictionary.
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

def read_yaml(p):
    """Read `.yaml` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """
    with open(p,'r') as f:
        return yaml.safe_load(f)
def to_yaml(d,p,**kws): 
    """Save `.yaml` file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
        
    Keyword Arguments:
        kws (d): parameters provided to `yaml.safe_dump`.
        
    Returns:
        p (str): path.
    """
    with open(p,'w') as f:
        yaml.safe_dump(d,f,**kws)
    return p        
def read_json(path_to_file,encoding=None):    
    """Read `.json` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """
    with open(path_to_file,encoding=encoding) as p:
        return json.load(p)
def to_json(data,p):
    """Save `.json` file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
                
    Returns:
        p (str): path.
    """
    with open(p, 'w') as outfile:
        json.dump(data, outfile)
    return p

def read_pickle(p):
    """Read `.pickle` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """    
    import pickle
    return pickle.load(open(p,
               'rb'))
def is_dict(p):
    return p.endswith(('.yml','.yaml','.json','.joblib','.pickle'))
    
def read_dict(p,fmt='',**kws):
    """Read dictionary file.
    
    Parameters:
        p (str): path.
        fmt (str): format of the file.
        
    Keyword Arguments:
        kws (d): parameters provided to reader function.
        
    Returns:
        d (dict): output dictionary.
    """    
    if '*' in p:
        from roux.lib.io import basenamenoext
        from glob import glob
        return {basenamenoext(p):read_dict(p) for p in glob(p)}
    if p.endswith('.yml') or p.endswith('.yaml') or fmt=='yml' or fmt=='yaml':
        return read_yaml(p,**kws)
    elif p.endswith('.json') or fmt=='json':
        return read_json(p,**kws)
    elif p.startswith('https'):
        from urllib.request import urlopen
        try:
            return json.load(urlopen(p))
        except:
            print(logging.error(p))
#         return read_json(p,**kws)    
    elif p.endswith('.pickle'):
        return read_pickle(p,**kws)
    elif p.endswith('.joblib'):
        import joblib
        return joblib.load(p,**kws)
    else:
        logging.error(f'supported extensions: .yml .yaml .json .pickle .joblib')
        
def to_dict(d,p,**kws):
    """Save dictionary file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
                
    Keyword Arguments:
        kws (d): parameters provided to export function.
        
    Returns:
        p (str): path.
    """
    from roux.lib.sys import makedirs
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    makedirs(p)
    if p.endswith('.yml') or p.endswith('.yaml'):
        return to_yaml(d,p,**kws)
    elif p.endswith('.json'):
        return to_json(d,p,**kws)
    elif p.endswith('.pickle'):
        import pickle
        return pickle.dump(d, open(p, 'wb'),**kws)
    elif p.endswith('.joblib'):
        import joblib
        return joblib.dump(d, p,**kws)     
    else:
        raise ValueError(f'supported extensions: .yml .yaml .json .pickle .joblib')        
        
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
