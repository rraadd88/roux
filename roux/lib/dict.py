# from roux.global_imports import *
from roux.lib.set import *
# from os.path import basename,dirname,exists
# from os import makedirs
import yaml
import json

from roux.lib.str import dict2str

def pprint_dict(d, indent=0):
    """
    thanks to https://stackoverflow.com/a/3229493/3521099
    """
    for key, value in d.items():
        print('\t' * indent + str(key+":"))
        if isinstance(value, dict):
            pretty(value, indent+1)
    else:
         print('\t' * (indent+1) + str(value))

def sort_dict(d1,by=1,ascending=True):     
    return dict(sorted(d1.items(), key=lambda item: item[1],reverse=not ascending))

def merge_dict_values(l,test=False):
    for di,d_ in enumerate(l):
        if di==0:
            d=d_
        else:
            d={k:d[k]+d_[k] for k in d}
        if test:
            print(','.join([str(len(d[k])) for k in d]))
    return d    

def read_yaml(p):
    with open(p,'r') as f:
        return yaml.safe_load(f)
def to_yaml(d,p,**kws): 
    with open(p,'w') as f:
        yaml.safe_dump(d,f,**kws)
    return p        
def read_json(path_to_file,encoding=None):    
    with open(path_to_file,encoding=encoding) as p:
        return json.load(p)
def to_json(data,p):
    with open(p, 'w') as outfile:
        json.dump(data, outfile)
    return p
def read_pickle(p):
    import pickle
    return pickle.load(open(p,
               'rb'))

def read_dict(p,fmt='',**kws):
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
        d = read_pickle(p,**kws)
    elif p.endswith('.joblib'):
        import joblib
        d=joblib.load(p,**kws)
    else:
        logging.error(f'supported extensions: .yml .yaml .json .pickle .joblib')
        
def to_dict(d,p,**kws):
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
        ValueError(f'supported extensions: .yml .yaml .json .pickle .joblib')        
        
def groupby_value(d):
    d_={k:[] for k in unique_dropna(d.values())}
    for k in d:
        if d[k] in d_:
            d_[d[k]].append(k)
    return d_       

# def dictwithtuplekeys2nested(d):
#     #https://stackoverflow.com/a/40130494/3521099
#     from itertools import groupby
#     return {g: {k[1]: v for k, v in items} 
#            for g, items in groupby(sorted(d.items()), key=lambda kv: kv[0][0])}
def convert_tuplekeys2nested(d1): return {k1:{k[1]:d1[k] for k in d1 if k1 in k} for k1 in np.unique([k[0] for k in d1])}

def flip_dict(d):
    if all([not isinstance(s,list) for s in d.values()]):
        if len(np.unique(d.keys()))!=len(np.unique(d.values())):
            logging.error('values should be unique to flip the dict')
            return
        else:
            return {d[k]:k for k in d}
    else:
        if not get_offdiagonal_values(intersections(d)).sum().sum()==0:
            ValueError('dict values should be mutually exclusive') 
        d_={}
        for k in d:
            for v in d[k]:
                d_[v]=k
        return d_

from roux.lib.str import str2dict

def merge_dicts(l):    
    from collections import ChainMap
    return dict(ChainMap(*l))
# def merge_dict(d1,d2):
#     from itertools import chain
#     from collections import defaultdict
#     dict3 = defaultdict(list)
#     for k, v in chain(d1.items(), d2.items()):
#         dict3[k].append(v)
#     return dict3

def head_dict(d, lines=5):
    return dict(itertools.islice(d.items(), lines))

from roux.lib.set import group_list_bylen,sort_list_by_list
