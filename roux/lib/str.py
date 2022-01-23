#!usr/bin/python

"""
================================
``io_strs``
================================
"""
import re
import logging
import numpy as np

# convert
def s2re(s,ss2re):
    for ss in ss2re:
        s=s.replace(ss,ss2re[ss])
    return s

def replacebyposition(s,i,replaceby):
    l=list(s)
    l[i]=replaceby
    return "".join(l)

def replace_many(s,replaces,replacewith='',
               ignore=False):
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
            ValueError()
    if not ignore: assert(s!=s_)
    return s
replacemany=replace_many

def replacelist(l,replaces,replacewith=''):
    lout=[]    
    for s in l:
        for r in replaces:
            s=s.replace(r,replacewith)
        lout.append(s) 
    return lout

def tuple2str(tup,sep=' '): 
    if isinstance(tup,tuple):
        tup=[str(s) for s in tup if not s=='']
        if len(tup)!=1:
            tup=sep.join(list(tup))
        else:
            tup=tup[0]
    elif not isinstance(tup,str):
        logging.error("tup is not str either")
    return tup

def isstrallowed(s,form):
    """
    Checks is input string conforms to input regex (`form`).

    :param s: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    import re
    match = re.match(form,s)
    return match is not None

def convertstr2format(col,form):
    """
    Convert input string to input regex (`form`).
    
    :param col: input string.
    :param form: eg. for hdf5: `"^[a-zA-Z_][a-zA-Z0-9_]*$"`
    """
    if not isstrallowed(col,form):
        col=col.replace(" ","_") 
        if not isstrallowed(col,form):
            chars_disallowed=[char for char in col if not isstrallowed(char,form)]
            for char in chars_disallowed:
                col=col.replace(char,"_")
    return col

def normalisestr(s):
    import re
    return re.sub('\W+','', s.lower()).replace('_','')


def remove_accents_df(df):
    cols=df.dtypes[(df.dtypes!=float) & (df.dtypes!=int) ].index.tolist()
    
    df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    return df

def make_pathable_string(s,replacewith='_'):
    """
    Removes symbols from a string to be compatible with directory structure.

    :param s: string
    """
    import re
    return re.sub(r'[^\w+/.]',replacewith, s).replace('+','_').replace('/My_Drive/','/My Drive/')
#     return re.sub('\W+',replacewith, s.lower() )

def linebreaker(i,break_pt=16,sep=' '):
    """
    used for adding labels in plots.

    :param l: list of strings
    :param break_pt: number, insert new line after this many letters 
    """
    if len(i)>break_pt:
        i_words=i.split(sep)
        i_out=''
        line_len=0
        for w in i_words:
            line_len+=len(w)+1
            if i_words.index(w)==0:
                i_out=w
            elif line_len>break_pt:
                line_len=0
                i_out="%s\n%s" % (i_out,w)
            else:
                i_out="%s %s" % (i_out,w)    
        return i_out    
    else:
        return i

def splitlabel(label,splitby=' ',ctrl='__'):
    """
    used for adding labels in plots.

    :param label: string
    :param splitby: string split the label by this character/string
    :param ctrl: string, marker that denotes a control condition  
    """
    splits=label.split(splitby)
    if len(splits)==2:
        return splits
    elif len(splits)==1:

        return splits+[ctrl]


def byte2str(b): 
    if not isinstance(b,str):
        return b.decode("utf-8")
    else:
        return b
        

# find
def findall(s,substring,outends=False,outstrs=False,
           suffixlen=0):
    import re
    finds=list(re.finditer(substring, s))
    if outends or outstrs:
        locs=[(a.start(), a.end()) for a in finds]
        if not outstrs:
            return locs
        else:
            return [s[l[0]:l[1]+suffixlen] for l in locs]
    else:
        return [a.start() for a in finds]
        
def getall_fillers(s,leftmarker='{',rightmarker='}',
                  leftoff=0,rightoff=0):
    filers=[]
    for ini, end in zip(findall(s,leftmarker,outends=False),findall(s,rightmarker,outends=False)):
        filers.append(s[ini+1+leftoff:end+rightoff])
    return filers    

###
def list2ranges(l):    
    ls=[]
    for l in zip(l[:-1],l[1:]):
        ls.append(l)
    return ls

def str2tiles(s,tilelen=10,test=False):
    tile2range={'tiles1': list(np.arange(0,len(s),tilelen)),
    'tiles2': list(np.arange(tilelen/2,len(s),tilelen))}

    for tile in tile2range:
        if len(tile2range[tile])%2!=0:
            tile2range[tile]=tile2range[tile]+[len(s)]
        tile2range[tile]=list2ranges(tile2range[tile])    
    range2tiles={}
    for rang in sorted(tile2range['tiles1']+tile2range['tiles2']):
        range2tiles[f"{int(rang[0])}_{int(rang[1])}"]=s[int(rang[0]):int(rang[1])]
    if test:
        print(tile2range)
    return range2tiles

def bracket(s,sbracket):
    pos=s.find(sbracket)
    return f"{s[:pos]}({s[pos:pos+len(sbracket)]})"

def get_bracket(s,l='(',r=')'):
#     import re
#     re.search(r'{l}(.*?){r}', s).group(1)    
    if l in s and r in s:
        return s[s.find(l)+1:s.find(r)]
    else:
        return '' 
    
## split
def align(s1,s2,
          prefix=False,
          suffix=False,
          common=True):
    """
    test:
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
    s3=get_prefix(s1,s2,**kws)
    s4=get_suffix(s1,s2,**kws)
    return s3 if len(s3)>=len(s4) else s4

def removesuffix(s1,suffix):
    """
    TODOS: 
    deprecate in py>39 use .removesuffix() instead.
    """
    if s1.endswith(suffix):
        return s1[:s1.rfind(suffix)]
    else:
        return s1

## filenames 
def strlist2one(l,label=''):
    # find unique prefix
    from roux.lib.set import unique
    from os.path import splitext
    for si in range(len(l[0]))[::-1]:
        s_i=l[0][:si]
        if all([s_i in s for s in l]):
    #         si=si+1
            break
    return f"{l[0][:si]}{''.join(list(unique([s[si] for s in l])))}{label}{splitext(l[0])[1]}"

def str_split(strng, sep, pos):
    """https://stackoverflow.com/a/52008134/3521099"""
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])



def str_ranges2list(s,func=int,range_marker='-',sep=',',replace=['[',']']):
    s=replacemany(s,replace)
    from roux.lib.set import flatten
    return flatten([list(range(func(i.split(range_marker)[0]),func(i.split(range_marker)[1])+1)) if '-' in i else int(i) for i in s.split(sep)])

def str2urlformat(s):
    import urllib
    return urllib.parse.quote_plus(s)

# dict
# def str2dict(s): return dict(item.split("=") for item in s.split(";"))
def str2dict(s,sep=';',sep_equal='='):
    """
    thanks to https://stackoverflow.com/a/186873/3521099
    """
    return dict(item.split(sep_equal) for item in s.split(sep))

def dict2str(d1,sep=';',sep2='='): return sep.join([sep2.join([k,str(v)]) for k,v in d1.items()])

# def s2dict(s,sep=';',sep_key=':',):
#     d={}
#     for pair in s.split(sep):
#         if pair!='':
#             d[pair.split(sep_key)[0]]=pair.split(sep_key)[1]
#     return d

# TODO: deprecate
# from roux.lib.sys import get_logger,get_datetime,get_time
def str2num(s):
    import re
    s1=" ".join(re.findall("[a-zA-Z]+", s))
    assert len(s1)==1, f"len({s1})!=1"
    assert s1==s[-1], f"not at the end"
    i1=" ".join(re.findall("[0-9]+", s))
    assert len(s)==len(s1)+len(i1), f"do not add up"
    return int(int(i1)*{'':1, 'K':1e3, 'M':1e6, 'G':1e9, 'T':1e12, 'P':1e15}[s1])

# from roux.lib.dict import str2dict
def num2str(num,magnitude=False,
           coff=10000,decimals=0):
    """
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
    import pandas as pd
    if isinstance(data,pd.Series):
        data=data.to_dict()
    if isinstance(data,dict):
#         from roux.lib.str import dict2str
        data=dict2str(dict(sorted(data.items())), sep=';', sep2='=')
    if not isinstance(data,bytes):
        data=data.encode(encoding='utf8')
    import zlib
    from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d
    return b64e(zlib.compress(data, 9)).decode("utf-8")

def decode(obscured,out=None,**kws):
    import zlib
    from base64 import urlsafe_b64encode as b64e, urlsafe_b64decode as b64d
    s2=zlib.decompress(b64d(obscured)).decode("utf-8")
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