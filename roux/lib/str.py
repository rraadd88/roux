#!usr/bin/python

"""Strings."""
import re
import logging
import numpy as np

# convert
def substitution(s,i,replaceby):
    """Substitute character in a string.
    
    Parameters:
        s (string): string.
        i (int): location.
        replaceby (string): character to substitute with.
        
    Returns:
        s (string): output string.
    """
    l=list(s)
    l[i]=replaceby
    return "".join(l)
# alias
replacebyposition=substitution

def replace_many(s,replaces,replacewith='',
               ignore=False):
    """Rename by replacing sub-strings.

    Parameters:
        s (str): input string.
        replaces (dict|list): from->to format or list containing substrings to remove.
        replacewith (str): replace to in case `replaces` is a list.
        ignore (bool): if True, not validate the successful replacements.
        
    Returns:
        s (DataFrame): output dataframe.
    """
    s_=s
    if isinstance(replaces,list):
        replaces={k:replacewith for k in replaces}
    if isinstance(replaces,dict):
        for k in replaces:
            s=s.replace(k,replaces[k])
    else:
        import inspect
        if inspect.isfunction(replaces):
            s=replaces(s_)
        else:
            raise ValueError(replaces)
    if not ignore: assert(s!=s_)
    return s
# alias
replacemany=replace_many

def tuple2str(tup,sep=' '):
    """Join tuple items.
    
    Parameters:
        tup (tuple|list): input tuple/list.
        sep (str): separator between the items.
        
    Returns:
        s (str): output string.
    """    
    if isinstance(tup,tuple):
        tup=[str(s) for s in tup if not s=='']
        if len(tup)!=1:
            tup=sep.join(list(tup))
        else:
            tup=tup[0]
    elif not isinstance(tup,str):
        logging.error("tup is not str either")
    return tup

# def normalisestr(s,):
#     """Normalise string.
    
#     Parameters:
#         s (string): input string.
    
#     Returns:
#         s (string): output string.
#     """
#     if not isinstance(s,str):
#         s=s.decode("utf-8")
#     import re
#     return re.sub('\W+','', s.lower()).replace('_','')

def linebreaker(text, width,
                sep='\n',
                **kws,
               ):
    """Insert `newline`s within a string. 
    
    Parameters:
        text (str): string.
        width (int): insert `newline` at this interval. 
        sep (string): separator to split the sub-strings.
        
    Returns:
        s (string): output string.
        
    References:
        1. `textwrap`: https://docs.python.org/3/library/textwrap.html
    """
    import textwrap
    return sep.join(textwrap.wrap(text, width,**kws,))
    # if len(i)>break_pt:
    #     i_words=i.split(sep)
    #     i_out=''
    #     line_len=0
    #     for w in i_words:
    #         line_len+=len(w)+1
    #         if i_words.index(w)==0:
    #             i_out=w
    #         elif line_len>break_pt:
    #             line_len=0
    #             i_out="%s\n%s" % (i_out,w)
    #         else:
    #             i_out="%s %s" % (i_out,w)    
    #     return i_out    
    # else:
    #     return i

# find
def findall(s,ss,outends=False,outstrs=False,
           suffixlen=0):
    """Find the substrings or their locations in a string.
   
    Parameters:
        s (string): input string.
        ss (string): substring.
        outends (bool): output end positions.
        outstrs (bool): output strings.
        suffixlen (int): length of the suffix.
    
    Returns:
        l (list): output list.
    """
    import re
    finds=list(re.finditer(ss, s))
    if outends or outstrs:
        locs=[(a.start(), a.end()) for a in finds]
        if not outstrs:
            return locs
        else:
            return [s[l[0]:l[1]+suffixlen] for l in locs]
    else:
        return [a.start() for a in finds]
        
def get_marked_substrings(s,leftmarker='{',rightmarker='}',
                          leftoff=0,rightoff=0):
    """Get the substrings flanked with markers from a string.
    
    Parameters:
        s (str): input string.
        leftmarker (str): marker on the left.
        rightmarker (str): marker on the right. 
        leftoff (int): offset on the left.
        rightoff (int): offset on the right.
    
    Returns:
        l (list): list of substrings.
    """
    filers=[]
    for ini, end in zip(findall(s,leftmarker,outends=False),findall(s,rightmarker,outends=False)):
        filers.append(s[ini+1+leftoff:end+rightoff])
    return filers    

getall_fillers=get_marked_substrings
###
def mark_substrings(s,ss,
                  leftmarker='(',rightmarker=')'):
    """Mark sub-string/s in a string.
    
    Parameters:
        s (str): input string.
        ss (str): substring.
        leftmarker (str): marker on the left.
        rightmarker (str): marker on the right. 
    
    Returns:
        s (str): string.
    """
    pos=s.find(ss)
    return f"{s[:pos]}{leftmarker}{s[pos:pos+len(ss)]}{rightmarker}"

def get_bracket(s,leftmarker='(',righttmarker=')'):
    """Get bracketed substrings.
    
    Parameters:
        s (string): string.
        leftmarker (str): marker on the left.
        rightmarker (str): marker on the right. 
    
    Returns:
        s (str): string.
        
    TODOs:
        1. Use `get_marked_substrings`.
    """
#     import re
#     re.search(r'{l}(.*?){r}', s).group(1)    
    if leftmarker in s and righttmarker in s:
        return s[s.find(leftmarker)+1:s.find(righttmarker)]
    else:
        return '' 
    
## split
def align(s1,s2,
          prefix=False,
          suffix=False,
          common=True):
    """Align strings.
        
    Parameters:
        s1 (str): string #1.
        s2 (str): string #2.
        prefix (str): prefix.
        suffix (str): suffix.
        common (str): common substring.
    
    Returns:
        l (list): output list.
        
    Notes:
        1. Code to test:
            [
            get_prefix(source,target,common=False),
            get_prefix(source,target,common=True),
            get_suffix(source,target,common=False),
            get_suffix(source,target,common=True),]
    """
    
    for i,t in enumerate(zip(list(s1),list(s2))):
        if t[0]!=t[1]:
            break
    if common:
        return [s1[:i],s2[:i]] if prefix else [s1[i+1:],s2[i+1:]] 
    else:
        return [s1[:i+1],s2[:i+1]] if prefix else [s1[i:],s2[i:]] 

from roux.lib.set import unique_str    
def get_prefix(s1,s2,common=True,clean=True): 
    """Get the prefix of the strings
    
    Parameters:
        s1 (str): 1st string.
        s2 (str): 2nd string.
        common (bool): get the common prefix (default:True).
        clean (bool): clean the leading and trailing whitespaces (default:True).
        
    Returns:
        s (str): prefix.
    """
    l1=align(s1,s2,prefix=True,common=common)
    if not common:
        return l1
    else:
        s3=unique_str(l1)
        if not clean:
            return s3
        else:
            return s3.strip().rsplit(' ', 1)[0]    
def get_suffix(s1,s2,common=True,clean=True):
    """Get the suffix of the strings
    
    Parameters:
        s1 (str): 1st string.
        s2 (str): 2nd string.
        common (bool): get the common prefix (default:True).
        clean (bool): clean the leading and trailing whitespaces (default:True).
        
    Returns:
        s (str): suffix.
    """    
    l1=align(s1,s2,suffix=True,common=common)
    if not common:
        if not clean:
            return l1
        else:
            split_pos=(max([s.count(' ') for s in l1])+1)*-1
            return [' '.join(s.split(' ')[split_pos:]) for s in [s1,s2]]
    else:
        s3=unique_str(l1)
        if not clean:
            return s3
        else:
            return s3.strip()#.rsplit(' ', 1)[0]
def get_fix(s1,s2,**kws):
    """Infer common prefix or suffix.
    
    Parameters:
        s1 (str): 1st string.
        s2 (str): 2nd string.
    
    Keyword parameters:
        kws: parameters provided to the `get_prefix` and `get_suffix` functions.
        
    Returns:
        s (str): prefix or suffix.
    """
    s3=get_prefix(s1,s2,**kws)
    s4=get_suffix(s1,s2,**kws)
    return s3 if len(s3)>=len(s4) else s4

def removesuffix(s1,suffix):
    """Remove suffix.
    
    Paramters:
        s1 (str): input string.
        suffix (str): suffix.
        
    Returns:
        s1 (str): string without the suffix.
        
    TODOs: 
        1. Deprecate in py>39 use .removesuffix() instead.
    """
    if s1.endswith(suffix):
        return s1[:s1.rfind(suffix)]
    else:
        return s1

# dict
# def str2dict(s): return dict(item.split("=") for item in s.split(";"))
def str2dict(s,sep=';',sep_equal='='):
    """String to dictionary.
    
    Parameters:
        s (str): string.
        sep (str): separator between entries (default:';').
        sep_equal (str): separator between the keys and the values (default:'=').
    
    Returns:
        d (dict): dictionary.
    
    References:
        1. https://stackoverflow.com/a/186873/3521099
    """
    return dict(item.split(sep_equal) for item in s.split(sep))

def dict2str(d1,sep=';',sep_equal='='): 
    """Dictionary to string.
    
    Parameters:
        d (dict): dictionary.
        sep (str): separator between entries (default:';').
        sep_equal (str): separator between the keys and the values (default:'=').
    
    Returns:
        s (str): string.
    """
    return sep.join([sep_equal.join([k,str(v)]) for k,v in d1.items()])

def str2num(s):
    """String to number.
    
    Parameters:
        s (str): string.
        
    Returns:
        i (int): number.
    """
    import re
    s1=" ".join(re.findall("[a-zA-Z]+", s))
    assert len(s1)==1, f"len({s1})!=1"
    assert s1==s[-1], f"not at the end"
    i1=" ".join(re.findall("[0-9]+", s))
    assert len(s)==len(s1)+len(i1), f"do not add up"
    return int(int(i1)*{'':1, 'K':1e3, 'M':1e6, 'G':1e9, 'T':1e12, 'P':1e15}[s1])

def num2str(num,magnitude=False,
           coff=10000,decimals=0):
    """Number to string.
    
    Parameters:
        num (int): number.
        magnitude (bool): use magnitudes (default:False).
        coff (int): cutoff (default:10000).
        decimals (int): decimal points (default:0).
        
    Returns:
        s (str): string.
        
    TODOs
        1. ~ if magnitude else not
    """
    if not magnitude:
        return f"{num:.1e}" if num>coff else f"{num}"
    else:
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        if decimals==0:
#             return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
            return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
        elif decimals==1:
#             return ('%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])).replace('.0','')
            return ('%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])).replace('.0','')

## ids
def encode(data,**kws):
    """Encode the data as a string.
    
    Parameters:
        data (dict|Series): input data. 
        
    Keyword parameters:
        kws: parameters provided to `encode`.
        
    Returns:
        s (string): output string. 
    """
    import pandas as pd
    if isinstance(data,pd.Series):
        data=data.to_dict()
    if isinstance(data,dict):
#         from roux.lib.str import dict2str
        data=dict2str(dict(sorted(data.items())), sep=';', sep_equal='=')
    if not isinstance(data,bytes):
        data=data.encode(encoding='utf8')
    import zlib
    from base64 import urlsafe_b64encode as b64e
    return b64e(zlib.compress(data, 9)).decode("utf-8",**kws)

def decode(s,out=None,**kws):
    """Decode data from a string.
    
    Parameters:
        s (string): encoded string. 
        out (str): output format (dict|df).
        
    Keyword parameters:
        kws: parameters provided to `dict2df`.
        
    Returns:
        d (dict|DataFrame): output data. 
    """
    import zlib
    from base64 import urlsafe_b64decode as b64d
    s2=zlib.decompress(b64d(s)).decode("utf-8")
    if out in ['dict','df']:
#         from roux.lib.str import str2dict
        d1=str2dict(s2, sep=';', sep_equal='=')
        if out=='dict':
            return d1
        elif out=='df':
            from roux.lib.df import dict2df
            return dict2df(d1,**kws)
    else:
        return s2