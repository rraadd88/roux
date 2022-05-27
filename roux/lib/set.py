"""Sets."""
import itertools
import numpy as np
import pandas as pd
import logging

## io
def to_list(l1,p):
    """Save list.
    
    Parameters:
        l1 (list): input list.
        p (str): path.
    
    Returns:
        p (str): path.        
    """
    from roux.lib.sys import makedirs
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    makedirs(p)    
    with open(p,'w') as f:
        f.write('\n'.join(l1))
    return p

def read_list(p):
    """Read the lines in the file.

    Args:
        p (str): path.

    Returns:
        l (list): list.
    """
    with open(p,'r') as f:
        s=f.readlines()
    return s
# alias to be deprecated
read_lines=read_list

from functools import reduce
def union(l):
    """Union of lists.
    
    Parameters: 
        l (list): list of lists.
        
    Returns:
        l (list): list.
    """
    return reduce(np.union1d, (l))
def intersection(l):
    """Intersections of lists.
    
    Parameters: 
        l (list): list of lists.
        
    Returns:
        l (list): list.
    """
    return reduce(np.intersect1d, (l))

# aliases: to be deprecated in th future 
list2union=union
list2intersection=intersection

def nunion(l):
    """Count the items in union.
    
    Parameters: 
        l (list): list of lists.
        
    Returns:
        i (int): count.    
    """
    return len(union(l))

def nintersection(l):
    """Count the items in intersetion.
    
    Parameters: 
        l (list): list of lists.
        
    Returns:
        i (int): count.    
    """
    return len(intersection(l))

# lists mostly for agg
def dropna(x):
    """Drop `np.nan` items from a list.
    
    Parameters:
        x (list): list.
    
    Returns:
        x (list): list.
    """
    x_=[]
    for i in x:
        if not pd.isnull(i):
            x_.append(i)
    return x_

def unique(l):
    """Unique items in a list.
    
    Parameters:
        l (list): input list.
    
    Returns:
        l (list): list.
    """
    return list(np.unique(l))
    
def list2str(x,ignore=False):
    """Returns string if single item in a list.
    
    Parameters:
        x (list): list
        
    Returns:
        s (str): string.        
    """
    x=list(x)
    if not ignore:
        assert(len(x)==1)        
    if len(x)>1:
        logging.warning('more than 1 str value encountered, returning list')
        return x
    else:
        return x[0]

def unique_str(l,**kws):
    """Unique single item from a list.
    
    Parameters:
        l (list): input list.
    
    Returns:
        l (list): list.
    """
    return list2str(unique(dropna(l)))    

def nunique(l,**kws):
    """Count unique items in a list
    
    Parameters:
        l (list): list
    
    Returns:
        i (int): count.
    """
    return len(unique(l,**kws))

def flatten(l):
    """List of lists to list.
    
    Parameters:
        l (list): input list.
        
    Returns:
        l (list): output list.
    """    
    return list(np.hstack(np.array(l,dtype=object)))

def get_alt(l1,s,): 
    """Get alternate item between two.
        
    Parameters:
        l1 (list): list.
        s (str): item.
        
    Returns: 
        s (str): alternate item.
    """
    assert(s in l1)
    return [i for i in l1 if i!=s][0]

def jaccard_index(l1,l2):
    # if len(l1)==0 or len(l2)==0:
    #     return 0,0,0
    x1=len(set(l1).intersection(set(l2)))
    x2=len(set(l1).union(set(l2)))
    if x1==0 or len(l1)==0 or len(l2)==0:
        return 0,x1,x2
    else:
        return x1/x2,x1,x2
    
def intersections(dn2list,jaccard=False,count=True,fast=False,test=False):
    """Get intersections between lists.
    
    Parameters:
        dn2list (dist): dictionary mapping to lists.
        jaccard (bool): return jaccard indices.
        count (bool): return counts.
        fast (bool): fast.
        test (bool): verbose.
    
    Returns:
        df (DataFrame): output dataframe.
            
    TODOs: 
        1. feed as an estimator to `df.corr()`.
        2. faster processing by filling up the symetric half of the adjacency matrix.
    """
    dn2list={k:dropna(dn2list[k]) for k in dn2list}
    df=pd.DataFrame(index=dn2list.keys(),
                columns=dn2list.keys())
    if jaccard:
        dn2list={k:set(dn2list[k]) for k in dn2list}
    from tqdm import tqdm
    for k1i,k1 in tqdm(enumerate(dn2list.keys())):
#         if test:
#             print(f"{(k1i/len(dn2list.keys()))*100:.02f}")
        for k2i,k2 in enumerate(dn2list.keys()):
            if fast and k1i>=k2i:
                continue
            if jaccard:
                if len(dn2list[k1].union(dn2list[k2]))!=0:
                    l=jaccard_index(dn2list[k1],dn2list[k2])
                    # l=len(set(dn2list[k1]).intersection(dn2list[k2]))/len(dn2list[k1].union(dn2list[k2]))
                else:
                    l=np.nan
            else:
                l=list(set(dn2list[k1]).intersection(dn2list[k2]))
            if count:
                df.loc[k1,k2]=len(l)
            else:
                df.loc[k1,k2]=l
    return df

## ranges
def range_overlap(l1,l2):
    """Overlap between ranges.
    
    Parameters:
        l1 (list): start and end integers of one range.
        l2 (list): start and end integers of other range.
        
    Returns:
        l (list): overlapped range. 
    """
    return list(set.intersection(set(range(l1[0],l1[1]+1,1)),
                            set(range(l2[0],l2[1]+1,1))))

from roux.lib.df import to_ranges,to_boolean
def get_windows(a, 
                size = None, 
                overlap = None, 
                windows=None, 
                overlap_fraction=None,
                stretch_last=False,
                out_ranges=True,
               ):
    """Windows/segments from a range. 
    
    Parameters:
        a (list): range.
        size (int): size of the windows. 
        windows (int): number of windows.
        overlap_fraction (float): overlap fraction. 
        overlap (int): overlap length. 
        stretch_last (bool): stretch last window. 
        out_ranges (bool): whether to output ranges.
    
    Returns:
        df1 (DataFrame): output dataframe.
        
    Notes:
        1. For development, use of `int` provides `np.floor`.
    """
    if not windows is None and size is None:
        # TODOs
        size = int(len(a)/windows)
    if not overlap_fraction is None and overlap is None:
        overlap=int(size*overlap_fraction)
    shape = (a.size - size + 1, size)
    strides = a.strides * 2
    a2=np.lib.stride_tricks.as_strided(a, strides = strides, shape = shape)[0::overlap]
    if not out_ranges:
        df1=pd.DataFrame(a2.T)#.add_prefix('window#')
        df1.index.name='position in window'
        df1=df1.melt(ignore_index=False,value_name='position',var_name='window#').reset_index()
        if stretch_last:
            if df1.iloc[-1,:]['position']!=a[-1]:
                a3=range(df1.iloc[-1,:]['position']+1,a[-1]+1)
                df_=pd.DataFrame({
                                  'position':np.repeat(df1.iloc[-1,:]['position'],len(a3)),
                                  'window#':np.repeat(df1.iloc[-1,:]['window#'],len(a3)),
                                  'position':a3,})
                df1=pd.concat([df1,df_],axis=0)
        return df1
    else:
        df1=pd.DataFrame(
                    {'window#':range(a2.shape[0]),
                      'start':a2.min(axis=1),
                      'end':a2.max(axis=1),
                     },
                )
        if stretch_last:
            if df1.iloc[-1,:]['end']!=a[-1]:
                df1.iloc[-1,:]['end']=a[-1]
        return df1
    
def bools2intervals(v):
    """Convert bools to intervals.
    
    Parameters:
        v (list): list of bools.
        
    Returns:
        l (list): intervals.
    """
    return np.flatnonzero(np.diff(np.r_[0,v,0])!=0).reshape(-1,2) - [0,1]

def list2ranges(l):    
    ls=[]
    for l in zip(l[:-1],l[1:]):
        ls.append(l)
    return ls

# def dfbool2intervals(df,colbool):
#     """
#     ds contains bool values
#     """
#     df.index=range(len(df))
#     intervals=bools2intervals(df[colbool])
#     for intervali,interval in enumerate(intervals):
#         df.loc[interval[0]:interval[1],f'{colbool} interval id']=intervali
#         df.loc[interval[0]:interval[1],f'{colbool} interval start']=interval[0]
#         df.loc[interval[0]:interval[1],f'{colbool} interval stop']=interval[1]
#         df.loc[interval[0]:interval[1],f'{colbool} interval length']=interval[1]-interval[0]+1
#         df.loc[interval[0]:interval[1],f'{colbool} interval within index']=range(interval[1]-interval[0]+1)    
#     df[f'{colbool} interval index']=df.index    
#     return df

def get_pairs(items,
              items_with=None,
              size=2,
              # with_self=False, # itertools.combination does not pair self
             ):
    """
    Creates a dataframe with the paired items.
    
    Notes:
        1. the ids of the items are sorted e.g. 'a'-'b' not 'b'-'a'.
    """
    ## get lists
    items=list(set(items)) ## unique, sorted
    if items_with is None:
        items_with=[]
    else:
        items_with=list(set(items_with))
    ## arrage
    if len(items_with)==0:
        o1=itertools.combinations(items,size)
    else:
        assert len(set(items) & set(items_with))!=0, 'two lists should be non-overlapping, otherwise pairs with self would be created.'
        o1=itertools.product(items,items_with)
    # create dataframe
    df0=pd.DataFrame(o1,columns=range(1,size+1))
    return df0
    