"""For processing individual pandas DataFrames/Series"""
import pandas as pd
import numpy as np
import logging
from icecream import ic
from roux.lib import to_rd

@to_rd
def get_name(df1,cols=None,coff=2,out=None):
    """Gets the name of the dataframe. 
    
    Especially useful within `groupby`+`pandarellel` context.

    Parameters:
        df1 (DataFrame): input dataframe.
        cols (list): list groupby columns. 
        coff (int): cutoff of unique values to infer the name.
        out (str): format of the output (list|not).

    Returns:
        name (tuple|str|list): name of the dataframe. 
    """    
    if hasattr(df1,'name'):
        name=df1.name
        name=name if isinstance(name,str) else list(name)
    elif not cols is None:
        name=df1.iloc[0,:][cols]
    else:
        l1=get_constants(df1.select_dtypes(object))
        if len(l1)<=coff:
            name=sorted(l1)
        elif len(l1)==0:
            return
        else:
            logging.warning(f"possible names in here?: {','.join(l1)}")
            return
    if out=='list':
        if isinstance(name,str):
            name=[name]
    return name
            
@to_rd            
def get_groupby_columns(df_): 
    """Get the columns supplied to `groupby`.

    Parameters:
        df_ (DataFrame): input dataframe.

    Returns:
        columns (list): list of columns.
    """
    return df_.apply(lambda x: all(x==df_.name)).loc[lambda x: x].index.tolist()

@to_rd
def get_constants(df1):
    """Get the columns with a single unique value.

    Parameters:
        df1 (DataFrame): input dataframe.

    Returns:
        columns (list): list of columns.
    """
    return df1.nunique().loc[lambda x: x==1].index.tolist()

## delete unneeded columns
@to_rd
def drop_unnamedcol(df):
    """Deletes the columns with "Unnamed" prefix.

    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)
### alias
delunnamedcol=drop_unnamedcol

@to_rd
def drop_levelcol(df):
    """Deletes the potentially temporary columns names with "level" prefix.

    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.
    """
    cols_del=[c for c in df.columns if 'level' in c]
    return df.drop(cols_del,axis=1)

@to_rd
def drop_constants(df):
    """Deletes columns with a single unique value.

    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.
    """    
    cols_del=get_constants(df)
    logging.warning(f"dropped columns: {', '.join(cols_del)}")
    return df.drop(cols_del,axis=1)

@to_rd
def dropby_patterns(df1,l1=None,test=False):
    """Deletes columns containing substrings i.e. patterns.

    Parameters:
        df1 (DataFrame): input dataframe.
        l1 (list): list of substrings.
        test (bool): verbose. 
        
    Returns:
        df1 (DataFrame): output dataframe.
    """
    if l1 is None or l1==[]:
        return df1
    if isinstance(l1,str):
        l1=[l1]
    s0='|'.join(l1).replace('(','\(').replace(')','\)')
    s1=f"^.*({s0}).*$"
    cols=df1.filter(regex=s1).columns.tolist()
    if test: info(s1)
    assert(len(cols)!=0)
    logging.info('columns dropped:'+','.join(cols))
    return df1.log.drop(labels=cols,axis=1)

@to_rd
def drop_inflates(df1,
                    col,
                    cols_index,
                  test=False,
                   ):
    """Deletes columns with high number of duplicates.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        col (str): column with values.
        cols_index (list): index columns.
        test (bool): verbose. 
        
    Returns:
        df1 (DataFrame): output dataframe.
    
    Notes:
        Under development.
    """
    logging.warning("UNDER DEVELOPMENT")
    df_=df1.rd.check_duplicated(cols_index,out='df')
    ## how many times duplicated?
    ds1_=df_[col].value_counts()
    ## unique number of times duplicated? e.g. x times whereas others are consistently y times
    ds2_=ds1_.value_counts()
    if test: 
        print(ds1_.head())
        print(ds2_)
    if len(ds2_)==1:
        logging.warning("skipping, inflates are consistent")
        logging.info(ds2_)
        return df1
    if len(ds2_.value_counts())==1:
        logging.warning("skipping, inflates are consistent")
        logging.info(ds2_)
        return df1        
    ## drop the unique number of times duplicated
    dropped_items=ds1_.loc[ds1_.isin(ds2_.loc[lambda x: x==1].index)].index.tolist()
    logging.info(f"dropped_items: {', '.join(dropped_items)}")
    df1=df1.loc[~(df1[col].isin(dropped_items)),:]
    assert(all(df1.rd.check_inflation([col])<1))
    return df1

@to_rd
def flatten_columns(df,**kws):
    """Multi-index columns to single-level.

    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.

    Keyword Arguments:
        kws (dict): parameters provided to `coltuples2str` function.    
    """    
    df.columns=coltuples2str(df.columns,**kws)
    return df

@to_rd
def lower_columns(df):
    """Column names of the dataframe to lower-case letters.

    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.
    """    
    df.columns=df.columns.str.lower()
    return df

@to_rd
def renameby_replace(df,replaces,ignore=True,**kws):
    """Rename columns by replacing sub-strings.

    Parameters:
        df (DataFrame): input dataframe.
        replaces (dict|list): from->to format or list containing substrings to remove.
        ignore (bool): if True, not validate the successful replacements.
        
    Returns:
        df (DataFrame): output dataframe.

    Keyword Arguments:
        kws (dict): parameters provided to `replacemany` function.    
    """    
    from roux.lib.str import replacemany
    df.columns=[replacemany(c,replaces,ignore=ignore,**kws) for c in df]
    return df


@to_rd
def clean_columns(df):
    """Standardise columns.

    Steps:
        1. Strip flanking white-spaces.
        2. Lower-case letters.
    
    Parameters:
        df (DataFrame): input dataframe.
        
    Returns:
        df (DataFrame): output dataframe.
    """
    df.columns=df.columns.str.strip().str.rstrip().str.lower()
    return df

@to_rd
def clean(df,cols=[],
          drop_constants=False,
          drop_unnamed=True,
          verb=False,
         ):
    """Deletes potentially temporary columns.

    Steps:
        1. Strip flanking white-spaces.
        2. Lower-case letters.
    
    Parameters:
        df (DataFrame): input dataframe.
        drop_constants (bool): whether to delete the columns with a single unique value. 
        drop_unnamed (bool): whether to delete the columns with 'Unnamed' prefix. 
        verb (bool): verbose.
        
    Returns:
        df (DataFrame): output dataframe.
    """
    cols_del=df.filter(regex="^(?:index|level|Unnamed|chunk|_).*$").columns.tolist()+df.filter(regex="^.*(?:\.1)$").columns.tolist()+cols
    # exceptions 
    cols_del=[c for c in cols_del if not c.endswith('0.1')]
    if drop_constants:
        df=df.rd.drop_constants()
    if not drop_unnamed:
        cols_del=[c for c in cols_del if not c.startswith('Unnamed')]
    if any(df.columns.duplicated()):
#         from roux.lib.set import unique
        if verb: logging.warning(f"duplicate column/s dropped:{df.loc[:,df.columns.duplicated()].columns.tolist()}")
        df=df.loc[:,~(df.columns.duplicated())]
    if len(cols_del)!=0:
        if verb: logging.warning(f"dropped columns: {', '.join(cols_del)}")
        return df.drop(cols_del,axis=1)
    else:
        return df
    
@to_rd
def compress(df1,coff_categories=20,test=False):
    """Compress the dataframe by converting columns containing strings/objects to categorical.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        coff_categories (int): if the number of unique values are less than cutoff the it will be converted to categories. 
        test (bool): verbose.
        
    Returns:
        df1 (DataFrame): output dataframe.
    """    
    if test: ini=df1.memory_usage().sum()
    ds=df1.select_dtypes('object').nunique()
    for c in ds[ds<=coff_categories].index:
        df1[c]=df1[c].astype('category')
    if test: logging.info(f"compression={((ini-df1.memory_usage().sum())/ini)*100:.1f}%")
    return df1

@to_rd
def clean_compress(df,kws_compress={},**kws_clean): 
    """`clean` and `compress` the dataframe.
    
    Parameters:
        df (DataFrame): input dataframe.
        kws_compress (int): keyword arguments for the `compress` function. 
        test (bool): verbose.
        
    Keyword Arguments:
        kws_clean (dict): parameters provided to `clean` function.
        
    Returns:
        df1 (DataFrame): output dataframe.

    See Also:
        `clean`
        `compress`
    """    
    return df.rd.clean(**kws_clean).rd.compress(**kws_compress)

## nans:    
@to_rd
def check_na(df,
             subset=None,
             perc=False,
            ):
    """Number/percentage of missing values in columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.
        
    Returns:
        ds (Series): output stats.
    """    
    if subset is None: subset=df.columns.tolist()
    if not perc:
        return df.loc[:,subset].isnull().sum()
    else:
        return (df.loc[:,subset].isnull().sum()/df.loc[:,subset].agg(len))*100

@to_rd
def validate_no_na(df,subset=None):
    """Validate no missing values in columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.
        
    Returns:
        ds (Series): output stats.
    """
    if subset is None: subset=df.columns.tolist()
    return not df.loc[:,subset].isnull().any().any()

@to_rd
def assert_no_na(df,subset=None):
    """Assert that no missing values in columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.
        
    Returns:
        ds (Series): output stats.
    """    
    assert validate_no_na(df,subset=subset), check_na(df,subset=subset) 
    return df

## nunique:
@to_rd
def check_nunique(df,subset=None,perc=False):
    """Number/percentage of unique values in columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.
        
    Returns:
        ds (Series): output stats.
    """
    if subset is None:
        subset=df.select_dtypes((object,bool)).columns.tolist()
    if not perc:
        return df.loc[:,subset].nunique()
    else:
        return (df.loc[:,subset].nunique()/df.loc[:,subset].agg(len))*100


## nunique:
@to_rd
def check_inflation(df1,subset=None):
    """Occurances of values in columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        
    Returns:
        ds (Series): output stats.
    """
    if subset is None: subset=df1.columns.tolist()    
    if subset is None:
        subset=df1.columns.tolist()
    return df1.loc[:,subset].apply(lambda x: (x.value_counts().values[0]/len(df1))*100).sort_values(ascending=False)
    
## duplicates:
@to_rd
def check_dups(df,subset=None,perc=False):
    """Check duplicates.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.
        
    Returns:
        ds (Series): output stats.
    """
    if subset is None: subset=df.columns.tolist()
    df1=df.loc[df.duplicated(subset=subset,keep=False),:].sort_values(by=subset)
    from roux.viz.annot import perc_label
    logging.info("duplicate rows: "+perc_label(len(df1),len(df)))
    if not perc:
        return df1
    else:
        return 100*(len(df1)/len(df))

@to_rd
def check_duplicated(df,subset=None,perc=False):
    """Check duplicates (alias of `check_dups`)    
    """
    return check_dups(df,subset=subset,perc=perc)

@to_rd
def validate_no_dups(df,subset=None,):
    """Validate that no duplicates.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
    """
    if subset is None: subset=df.columns.tolist()
    out=not df.duplicated(subset=subset).any()
    if not out: logging.warning('duplicate rows found')
    return out

@to_rd
def validate_no_duplicates(df,subset=None,):
    """Validate that no duplicates (alias of `validate_no_dups`)
    """
    return validate_no_dups(df,subset=subset,)

@to_rd
def assert_no_dups(df,subset=None):
    """Assert that no duplicates
    """    
    assert validate_no_dups(df,subset=subset), check_dups(df,subset=subset,perc=False)
    return df

## asserts        
@to_rd
def validate_dense(df01,subset=None,duplicates=True,na=True,message=None,):
    """Validate no missing values and no duplicates in the dataframe.
    
    Parameters:
        df01 (DataFrame): input dataframe.
        subset (list): list of columns.
        duplicates (bool): whether to check duplicates.
        na (bool): whether to check na.
        message (str): error message
        
    """    
    if subset is None:
        subset=df01.columns.tolist()
    if duplicates: df01.rd.validate_no_dups(subset=subset)#, 'duplicates found' if message is None else message
    if na: df01.rd.validate_no_na(subset=subset)# if message is None else message
    return True

@to_rd
def assert_dense(df01,subset=None,duplicates=True,na=True,message=None):
    """Alias of `validate_dense`.
    
    Notes:
        to be deprecated in future releases.
    """
    assert validate_dense(df01,subset=subset,duplicates=duplicates,na=na,message=message)
    return df01

## mappings
@to_rd
def classify_mappings(
    df1: pd.DataFrame,
    col1: str,
    col2: str,
    clean: bool=False,
    ) -> pd.DataFrame:
    """Classify mappings between items in two columns.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        col1 (str): column #1.
        col2 (str): column #2.
        clean (str): drop columns with the counts.
        
    Returns:
        (pd.DataFrame): output.
    """    
    df1[col2+' count']=df1.groupby(col1)[col2].transform('nunique')
    df1[col1+' count']=df1.groupby(col2)[col1].transform('nunique')
    df1['mapping']=df1.apply(lambda x: "1:1" if (x[col1+' count']==1) and (x[col2+' count']==1) else \
                                        "1:m" if (x[col1+' count']==1) else \
                                        "m:1" if (x[col2+' count']==1) else "m:m",
                                        axis=1)
    if clean:
        df1=df1.drop([col1+' count',col2+' count'],axis=1)
    return df1

@to_rd        
def check_mappings(df,
                   subset=None,
                   out='full',
                  ):
    """Mapping between items in two columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (str): format of the output.
        
    Returns:
        ds (Series): output stats.
    """
    if subset is None: subset=df.columns.tolist()
    import itertools
    d={}
    for t in list(itertools.permutations(subset,2)):
        d[t]=df.groupby(t[0])[t[1]].nunique().value_counts()
    df2=pd.concat(d,axis=0,ignore_index=False,names=['from','to','map to']).to_frame('map from').sort_index().reset_index(-1).loc[:,['map from','map to']]
    if out=='full':
        return df2
    else:
        return df2.loc[tuple(subset),:]#'map to'].item()

@to_rd        
def validate_1_1_mappings(df,
                   subset=None,
                  ):
    """Validate that the papping between items in two columns is 1:1.
    
    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (str): format of the output.
        
    """
    df1=check_mappings(df,
                   subset=subset,
                  )
    assert all(df1['map to']==1), df1
    
@to_rd
def get_mappings(df1,
                 subset=None,
                 keep='1:1',
                 clean=False,
                 cols=None,
                ):
    """Classify the mapapping between items in two columns.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        subset (list): list of columns.
        keep (str): type of mapping (1:1|1:m|m:1).
        clean (bool): whether remove temporary columns. 
        cols (list): alias of `subset`.
        
    Returns:
        df (DataFrame): output dataframe.
    """
    if not cols is None and not subset is None: 
        logging.error(f"cols and subset are alias, both cannot be used.")
        return
    if cols is None and not subset is None: cols=subset        
    if cols is None: cols=df1.columns.tolist()
    if not df1.rd.validate_no_dups(cols):
        df1=df1.loc[:,cols].log.drop_duplicates()
    if len(cols)==2:
        from roux.lib.set import get_alt
        df2=df1.copy()
        cols2=[]
        for c in cols:
            d1=df2.groupby(c)[get_alt(cols,c)].nunique().to_dict()
            c2=f"{c}:{get_alt(cols,c)}"
            df2[c2]=df2[c].map(d1)
            cols2.append(c2)
        df2['mapping']=df2.loc[:,cols2].apply(lambda x: ':'.join(["1" if i==1 else 'm' for i in x]),axis=1)
        if keep=='1:1':
            df2=df2.rd.filter_rows({'mapping':'1:1'})
        else:
            logging.info(df2['mapping'].value_counts())
        if clean:
            df2=df2.drop(cols2+['mapping'],axis=1)
        return df2
    else:
        d1={'1:1':df1.copy(),
           }
        if keep!='1:1':
            d1['not']=pd.DataFrame()
        import itertools
    #     for t in list(itertools.permutations(cols,2)):
        for c in cols:
            d1['1:1']=d1['1:1'].groupby(c).filter(lambda df: len(df)==1)
            if keep!='1:1':
                d1['not']=d1['not'].append(df1.copy().groupby(c).filter(lambda df: len(df)!=1))
        if keep=='1:1':
            logging.info(df1.shape)
            logging.info(d1['1:1'].shape)
            return d1['1:1']
        else:
            assert(len(df1)==len(d1['1:1'])+len(d1['not']))
            return pd.concat(d1,axis=0,names=['mapping']).reset_index()


@to_rd
def groupby_filter_fast(df1,col,fun,how,coff):
    """Groupby and filter fast.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        col (str): column name.
        fun (object): function to filter with.
        how (str): greater or less than `coff` (>|<).  
        coff (float): cut-off.    
        
    Returns:
        df1 (DataFrame): output dataframe.
    
    Todo:
        Deprecation if `pandas.core.groupby.DataFrameGroupBy.filter` is faster.
    """
    ds1=df1.groupby(col).transform(fun)
    if how=='<':
        return df1.loc[(ds1<coff),:]
    elif how=='>':
        return df1.loc[(ds1>coff),:]
    else:
        raise ValueError(how)
        
@to_rd
def to_map_binary(df,colgroupby=None,colvalue=None):
    """Convert linear mappings to a binary map
    
    Parameters:
        df (DataFrame): input dataframe.
        colgroupby (str): name of the column for groupby.
        colvalue (str): name of the column containing values.
        
    Returns:
        df1 (DataFrame): output dataframe.
    """
    colgroupby=[colgroupby] if isinstance(colgroupby,str) else colgroupby
    colvalue=[colvalue] if isinstance(colvalue,str) else colvalue
    if not df.rd.validate_no_dups(colgroupby+colvalue):
        logging.warning('duplicates found')
        df=df.log.drop_duplicates(subset=colgroupby+colvalue)
    return (df
            .assign(_value=True)
            .pivot(index=colvalue,columns=colgroupby,values='_value')
            .fillna(False)
            )

## intersections 
@to_rd        
def check_intersections(df,
                        colindex=None, # 'samples'
                        colgroupby=None, # 'yticklabels'
                        plot=False,
                        **kws_plot):
    """Check intersections.
    Linear dataframe to is converted to a binary map and then to a series using `groupby`.

    Parameters:
        df (DataFrame): input dataframe.
        colindex (str): name of the index column.
        colgroupby (str): name of the groupby column.
        plot (bool): plot or not.
    
    Returns:
        ds1 (Series): output Series.

    Keyword Arguments:
        kws_plot (dict): parameters provided to the plotting function.
    """
    # if isinstance(colindex,str):
    #     colindex=[colindex]
    if isinstance(df,pd.DataFrame):
        if not (colgroupby is None or colindex is None) :
            if not all(df.dtypes==bool): 
#             if isinstance(colgroupby,str):
                # lin
                df1=to_map_binary(df,colgroupby=colgroupby,colvalue=colindex)
                ds=df1.groupby(df1.columns.to_list()).size()
            elif isinstance(colgroupby,(str,list)):
                assert(not df.rd.check_duplicated([colindex]+colgroupby))
                # map
                # df=df.set_index(colindex).loc[:,colgroupby] 
                # ds=df.groupby(df.columns.tolist()).size()
                ds=df.groupby(colgroupby).nunique(colindex)
            else:
                logging.error('colgroupby should be a str or list')
        else:
            # map
            ds=map2groupby(df)
    elif isinstance(df,pd.Series):
        ds=df
    # elif isinstance(df,dict):
    #     ds=dict2df(d1).rd.check_intersections(colindex='value',colgroupby='key')
    else:
        raise ValueError("data type of `df`")
    ds.name=colindex if isinstance(colindex, str) else ','.join(colindex) if isinstance(colindex, list) else None
    if plot:
        from roux.viz.bar import plot_intersections
        return plot_intersections(ds,**kws_plot)
    else:
        return ds

def get_totals(ds1):
    """Get totals from the output of `check_intersections`.
    
    Parameters:
        ds1 (Series): input Series.
    
    Returns:
        d (dict): output dictionary.
    """
    col=ds1.name if not ds1.name is None else 0
    df1=ds1.to_frame().reset_index()
    return {c:df1.loc[df1[c],col].sum() for c in ds1.index.names}
    
#filter df
@to_rd
def filter_rows(df,
                d,
                sign='==',
                logic='and',
                drop_constants=False,
                test=False,
                verb=True,
               ):
    """Filter rows using a dictionary.
    
    Parameters:
        df (DataFrame): input dataframe.
        d (dict): dictionary.
        sign (str): condition within mappings ('==').
        logic (str): condition between mappings ('and').
        drop_constants (bool): to drop the columns with single unique value (False).
        test (bool): testing (False).
        verb (bool): more verbose (True).
                
    Returns:
        df (DataFrame): output dataframe.
    
    """
    if verb: logging.info(df.shape)    
    assert(all([isinstance(d[k],(str,list)) for k in d]))
    qry = f" {logic} ".join([f"`{k}` {sign} "+(f"'{v}'" if isinstance(v,str) else f"{v}") for k,v in d.items()])
    df1=df.query(qry)
    if test:
        logging.info(df1.loc[:,list(d.keys())].drop_duplicates())
        logging.warning('may be some column names are wrong..')
        logging.warning([k for k in d if not k in df])
    if verb: logging.info(df1.shape)
    if drop_constants:
        df1=df1.rd.drop_constants()
    return df1

## conversion to type
@to_rd
def to_dict(df,cols,drop_duplicates=False):
    """DataFrame to dictionary.
    
    Parameters:
        df (DataFrame): input dataframe.
        cols (list): list of two columns: 1st contains keys and second contains value.
        drop_duplicates: whether to drop the duplicate values (False).
        
    Returns:
        d (dict): output dictionary.
    """
    df=df.log.dropna(subset=cols)
    if drop_duplicates:
        df=df.loc[:,cols].drop_duplicates()
    if not df[cols[0]].duplicated().any():
        return df.set_index(cols[0])[cols[1]].to_dict()
    else:
        logging.warning('format: {key:list}')
        assert df[cols[1]].dtype=='O', df[cols[1]].dtype
        return df.groupby(cols[0])[cols[1]].unique().to_dict()        

## to avoid overlap with `io_dict.to_dict`
del to_dict

## conversion
@to_rd
def get_bools(df,cols,drop=False):
    """Columns to bools. One-hot-encoder (`get_dummies`).
    
    Parameters:
        df (DataFrame): input dataframe. 
        cols (list): columns to encode.
        drop (bool): drop the `cols` (False).
    
    Returns: 
        df (DataFrame): output dataframe.
    """
    for c in cols:
        df_=pd.get_dummies(df[c],
                                  prefix=c,
                                  prefix_sep=": ",
                                  dummy_na=False)
        df_=df_.replace(1,True).replace(0,False)
        df=df.join(df_)
        if drop:
            df=df.drop([c],axis=1)
    return df

@to_rd
def agg_bools(df1,cols):
    """Bools to columns. Reverse of one-hot encoder (`get_dummies`). 
    
    Parameters:
        df1 (DataFrame): input dataframe.
        cols (list): columns. 
    
    Returns:
        ds (Series): output series.
    """
    col='+'.join(cols)
#     print(df1.loc[:,cols].T.sum())
    assert(all(df1.loc[:,cols].T.sum()==1))
    for c in cols:
        df1.loc[df1[c],col]=c
    return df1[col]     

## paired dfs
@to_rd
def melt_paired(df,
                cols_index=None, # paired
                suffixes=None,
                cols_value=None,
                ):
    """Melt a paired dataframe.
    
    Parameters:
        df (DataFrame): input dataframe.
        cols_index (list): paired index columns (None).
        suffixes (list): paired suffixes (None).
        cols_value (list): names of the columns containing the values (None).

    Notes:
        Partial melt melts selected columns `cols_value`.
    
    Examples:
        Paired parameters:
            cols_value=['value1','value2'],
            suffixes=['gene1','gene2'],
    """
    if cols_value is None:
        assert not (cols_index is None and suffixes is None), "either cols_index or suffixes needed" 
        if suffixes is None and not cols_index is None:
            from roux.lib.str import get_suffix
            suffixes=get_suffix(*cols_index,common=False, clean=True)
            
        # both suffixes should not be in any column name
        assert not any([all([s in c for s in suffixes]) for c in df]), "both suffixes should not be in a single column name"
        assert not any([c==s for s in suffixes for c in df]), "suffix should not be the column name"
        assert all([any([s in c for c in df]) for s in suffixes]), "both suffix should be in the column names"
        
        cols_common=[c for c in df if not any([s in c for s in suffixes])]
        dn2df={}
        for s in suffixes:
            cols=[c for c in df if s in c]
            dn2df[s]=df.loc[:,cols_common+cols].rename(columns={c:c.replace(s,'') for c in cols},
                                                      errors='raise')
        df1=pd.concat(dn2df,axis=0,names=['suffix']).reset_index(0)
        df2=df1.rename(columns={c: c[:-1] if c.endswith(' ') else c[1:] if c.startswith(' ') else c for c in df1},
                         errors='raise')
        if '' in df2:
            df2=df2.rename(columns={'':'id'},
                   errors='raise')
        assert len(df2)==len(df)*2
        return df2
    else:
        assert not suffixes is None
        import itertools
        df2=pd.concat({c: df.rename(columns={f"{c} {s}":f"value {s}" for s in suffixes},errors='raise') for c in cols_value},
                        axis=0,names=['variable'],
                        ).reset_index(0)
        if len(cols_value)>1:
            df2=df2.drop([f"{c} {s}" for c,s in itertools.product(cols_value,suffixes)],axis=1)
        assert len(df2)==len(df)*len(cols_value)
        return df2

@to_rd
def get_chunks(df1,colindex,colvalue,bins=None,value='right'):
    """Get chunks of a dataframe.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        colindex (str): name of the index column.
        colvalue (str): name of the column containing values [0-100]
        bins (int): number of bins.
        value (str): value to use as the name of the chunk ('right').
        
    Returns: 
        ds (Series): output series.
    """
    from roux.lib.set import unique,nunique
    if bins==0:
        df1['chunk']=bins
        logging.warning("bins=0, so chunks=1")
        return df1['chunk']
    elif bins is None:
        bins=int(np.ceil(df1.memory_usage().sum()/1e9))
    df2=df1.loc[:,[colindex,colvalue]].drop_duplicates()
    from roux.stat.transform import get_bins
    d1=get_bins(df2.set_index(colindex)[colvalue],
                 bins=bins,
                 value=value,
                ignore=True)
    ## number bins
    d_={k:f"chunk{ki+1:08d}_upto{int(k):03d}" for ki,k in enumerate(sorted(np.unique(list(d1.values()))))}
    ## rename bins
    d2={k:d_[d1[k]] for k in d1}
    assert(nunique(d1.values())==nunique(d2.values()))
    df1['chunk']=df1[colindex].map(d2)
    return df1['chunk']

## GROUPBY
# aggregate dataframes
def get_group(groups,i=None,verbose=True):
    """Get a dataframe for a group out of the `groupby` object.

    Parameters:
        groups (object): groupby object.
        i (int): index of the group (None).
        verbose (bool): verbose (True).
        
    Returns: 
        df (DataFrame): output dataframe.
        
    Notes: 
        Useful for testing `groupby`.
    """
    if not i is None: 
        dn=list(groups.groups.keys())[i]
    else:
        dn=groups.size().sort_values(ascending=False).index.tolist()[0]
    logging.info(dn)
    df=groups.get_group(dn)
    df.name=dn
    return df
        
# multiindex
def coltuples2str(cols,sep=' '):
    """Convert list of tuples (e.g. MultiIndex) to list of strings
    
    Parameters:
        cols (list): list of tuples.
        sep (str): separator within the joined tuples (' ').
        
    Returns:
        cols_str (list): list of strings.
    """
    from roux.lib.str import tuple2str
    cols_str=[]
    for col in cols:
        cols_str.append(tuple2str(col,sep=sep))
    return cols_str

@to_rd
def to_multiindex_columns(df,suffixes,test=False):
    """Single level columns to multiindex.
    
    Parameters:
        df (DataFrame): input dataframe.
        suffixes (list): list of suffixes.
        test (bool): verbose (False).
        
    Returns:
        df (DataFrame): output dataframe.
    """
    cols=[c for c in df if c.endswith(f' {suffixes[0]}') or c.endswith(f' {suffixes[1]}')]
    if test:
        logging.info(cols)
    df=df.loc[:,cols]
    df=df.rename(columns={c: (s,c.replace(f' {s}','')) for s in suffixes for c in df if c.endswith(f' {s}')},
                errors='raise')
    df.columns=pd.MultiIndex.from_tuples(df.columns)
    return df

## ranges
@to_rd
def to_ranges(df1,colindex,colbool,sort=True):
    """Ranges from boolean columns.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        colindex (str): column containing index items.
        colbool (str): column containing boolean values.
        sort (bool): sort the dataframe (True).
        
    Returns:
        df1 (DataFrame): output dataframe.
        
    TODO:
        compare with io_sets.bools2intervals.
    """
    import scipy as sc
    if sort:
        df1=df1.sort_values(by=colindex)
    df1['group']=sc.ndimage.measurements.label(df1[colbool].astype(int))[0]
    return df1.loc[(df1['group']!=0),:].groupby('group')[colindex].agg([min,max]).reset_index()

@to_rd
def to_boolean(df1):
    """Boolean from ranges.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        
    Returns:
        ds (Series): output series.
        
    TODO:
        compare with io_sets.bools2intervals.
    """    
    low, high = np.array(df1).T[:,:, None]
    a = np.arange(high.max() + 1)
    return ((a >= low) & (a <= high)).any(axis=0)


## sorting
def to_cat(ds1,cats,ordered = True):
    """To series containing categories.
    
    Parameters:
        ds1 (Series): input series.
        cats (list): categories.
        ordered (bool): if the categories are ordered (True).

    Returns:
        ds1 (Series): output series.
    """
    ds1=ds1.astype('category')
    ds1=ds1.cat.set_categories(new_categories = cats, ordered = ordered)
    assert(not ds1.isnull().any())
    return ds1

@to_rd
def sort_valuesby_list(df1,by,cats,**kws):
    """Sort dataframe by custom order of items in a column.
        
    Parameters:
        df1 (DataFrame): input dataframe.
        by (str): column.
        cats (list): ordered list of items.

    Keyword parameters:
        kws (dict): parameters provided to `sort_values`.
    
    Returns:
        df (DataFrame): output dataframe.
    """
    df1[by]=to_cat(df1[by],cats,ordered = True)
    return df1.sort_values(by=by, **kws)

## apply_agg
def agg_by_order(x,order):
    """Get first item in the order.

    Parameters: 
        x (list): list.
        order (list): desired order of the items.

    Returns:
        k: first item.
        
    Notes:
        Used for sorting strings. e.g. `damaging > other non-conserving > other conserving`

    TODO: 
        Convert categories to numbers and take min
    """
    if len(x)==1:
#         print(x.values)
        return list(x.values)[0]
    for k in order:
        if k in x.values:
            return k
def agg_by_order_counts(x,order):
    """Get the aggregated counts by order*.

    Parameters:
        x (list): list.
        order (list): desired order of the items.
        
    Returns:
        df (DataFrame): output dataframe.
    
    Examples:
        df=pd.DataFrame({'a1':['a','b','c','a','b','c','d'],
        'b1':['a1','a1','a1','b1','b1','b1','b1'],})
        df.groupby('b1').apply(lambda df : agg_by_order_counts(x=df['a1'],
                                                       order=['b','c','a'],
                                                       ))
    """    
    ds=x.value_counts()
    ds=ds.add_prefix(f"{x.name}=")
    ds[x.name]=agg_by_order(x,order)
    return ds.to_frame('').T

@to_rd
def groupby_sort_values(df,col_groupby,col_sortby,
                 subset=None,
                 col_subset=None,
                 func='mean',ascending=True):
    """Sort groups. 
    
    Parameters:
        df (DataFrame): input dataframe.
        col_groupby (str|list): column/s to groupby with.
        col_sortby (str|list): column/s to sort values with.
        subset (list): columns (None).
        col_subset (str): column containing the subset (None).
        func (str): aggregate function, provided to numpy ('mean').
        ascending (bool): sort values ascending (True).
        
    Returns:
        df (DataFrame): output dataframe.
    """
    gs=df.groupby(col_groupby)
    if subset is None:
        df1=gs.agg({col_sortby:getattr(np,func)}).reset_index()
        df2=df.merge(df1,
                on=col_groupby,how='inner',suffixes=['',f' per {col_groupby}'])
        logging.warning(f'column added to df: {col_sortby} per {col_groupby}')
        return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)
    else:
        df1=df.groupby(col_subset).get_group(subset)
        df2=df1.groupby(col_groupby).agg({col_sortby:getattr(np,func)}).reset_index()
        return sort_col_by_list(df, 
                                col_groupby,
                                df2.sort_values(col_sortby,
                                                ascending=ascending)[col_groupby])
#         return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)
sort_values_groupby=groupby_sort_values

@to_rd
def swap_paired_cols(df_,suffixes=['gene1', 'gene2']):
    """Swap suffixes of paired columns.
    
    Parameters:
        df_ (DataFrame): input dataframe.
        suffixes (list): suffixes.
    
    Returns: 
        df (DataFrame): output dataframe.    
    """
    rename={c:c.replace(suffixes[0],suffixes[1]) if (suffixes[0] in c) else c.replace(suffixes[1],suffixes[0]) if (suffixes[1] in c) else c for c in df_}
    return df_.rename(columns=rename,errors='raise')

@to_rd
def sort_columns_by_values(df,cols_sortby=['mutation gene1','mutation gene2'],
                            suffixes=['gene1','gene2'], # no spaces
                           clean=False,
                            ):
    """Sort the values in columns in ascending order.
    
    Parameters:
        df (DataFrame): input dataframe.
        cols_sortby (list): (['mutation gene1','mutation gene2'])
        suffixes (list): suffixes, without no spaces. (['gene1','gene2'])
        
    Returns:
        df (DataFrame): output dataframe.
        
    Notes:
        In the output dataframe, `sorted` means values are sorted because gene1>gene2.
    """
    df.rd.assert_no_na(subset=cols_sortby)
    suffixes=[s.replace(' ','') for s in suffixes]
    dn2df={}
    # keys: (equal, to be sorted)
    dn2df[(False,False)]=df.loc[(df[cols_sortby[0]]<df[cols_sortby[1]]),:]
    dn2df[(False,True)]=df.loc[(df[cols_sortby[0]]>df[cols_sortby[1]]),:]
    dn2df[(True,False)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    dn2df[(True,True)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    ## rename columns of of to be sorted
    ## TODO: use swap_paired_cols
    rename={c:c.replace(suffixes[0],suffixes[1]) if (suffixes[0] in c) else c.replace(suffixes[1],suffixes[0]) if (suffixes[1] in c) else c for c in df}
    for k in [True, False]:
        dn2df[(k,True)]=dn2df[(k,True)].rename(columns=rename,
                                              errors='raise')
        
    df1=pd.concat(dn2df,names=['equal','sorted']).reset_index([0,1])
    logging.info(df1.groupby(['equal','sorted']).size())
    if clean:
        df1=df1.drop(['equal','sorted'],axis=1)
    return df1
    
## ids
@to_rd
def make_ids(df,cols,ids_have_equal_length,sep='--',sort=False):
    """Make ids by joining string ids in more than one columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        cols (list): columns. 
        ids_have_equal_length (bool): ids have equal length, if True faster processing.
        sep (str): separator between the ids ('--').
        sort (bool): sort the ids before joining (False).
    
    Returns:
        ds (Series): output series.
    """
    get_ids=lambda x: '--'.join(x)
    get_ids_sorted=lambda x: '--'.join(sorted(x))
    if ids_have_equal_length:
        logging.warning("ids should be of equal character length and should not contain non-alphanumeric characters e.g. '.'")
        return np.apply_along_axis(get_ids if not sort else get_ids_sorted, 1, df.loc[:,cols].values)
    else:
        return df.loc[:,cols].agg(lambda x: sep.join(x if not sort else sorted(x)),axis=1)

@to_rd
def make_ids_sorted(df,cols,ids_have_equal_length,sep='--'):
    """Make sorted ids by joining string ids in more than one columns.
    
    Parameters:
        df (DataFrame): input dataframe.
        cols (list): columns. 
        ids_have_equal_length (bool): ids have equal length, if True faster processing.
        sep (str): separator between the ids ('--').
    
    Returns:
        ds (Series): output series.
    """    
    return make_ids(df,cols,ids_have_equal_length,sep=sep,sort=True)
    
def get_alt_id(s1='A--B',s2='A',sep='--'): 
    """Get alternate/partner id from a paired id.
    
    Parameters:
        s1 (str): joined id.
        s2 (str): query id. 
        
    Returns:
        s (str): partner id.
    """    
    return [s for s in s1.split(sep) if s!=s2][0]

@to_rd    
def split_ids(df1,col,sep='--',prefix=None):
    """Split joined ids to individual ones.
    
    Parameters:
        df1 (DataFrame): input dataframe.
        col (str): column containing the joined ids.
        sep (str): separator within the joined ids ('--').
        prefix (str): prefix of the individual ids (None).
    
    Return:
        df1 (DataFrame): output dataframe.
    """
    # assert not df1._is_view, "input series should be a copy not a view"
    df=df1[col].str.split(sep,expand=True)
    for i in range(len(df.columns)):
        df1[f"{col} {i+1}"]=df[i].copy()
    if not prefix is None:
        df1=df1.rd.renameby_replace(replaces={f"{col} ":prefix})
    return df1

## tables io
def dict2df(d,colkey='key',colvalue='value'):
    """Dictionary to DataFrame.
    
    Parameters:
        d (dict): dictionary.
        colkey (str): name of column containing the keys.
        colvalue (str): name of column containing the values.
    
    Returns:
        df (DataFrame): output dataframe.
    """
    if not isinstance(list(d.values())[0],list):
        return pd.DataFrame({colkey:d.keys(), colvalue:d.values()})
    else:
        return pd.DataFrame(pd.concat({k:pd.Series(d[k]) for k in d})).droplevel(1).reset_index().rename(columns={'index':colkey,0:colvalue},
                                                                                                        errors='raise')
def log_shape_change(d1,fun=''):
    """Report the changes in the shapes of a DataFrame.

    Parameters:
        d1 (dic): dictionary containing the shapes.
        fun (str): name of the function.
    """
    if d1['from']!=d1['to']:
        prefix=f"{fun}: " if fun!='' else ''
        if d1['from'][0]==d1['to'][0]:
            logging.info(f"{prefix}shape changed: {d1['from']}->{d1['to']}, length constant")
        elif d1['from'][1]==d1['to'][1]:
            logging.info(f"{prefix}shape changed: {d1['from']}->{d1['to']}, width constant")        
        else:
            logging.info(f"{prefix}shape changed: {d1['from']}->{d1['to']}")
## log
def log_apply(df, fun, 
              validate_equal_length=False,
              validate_equal_width=False,
              validate_equal_shape=False,
              validate_no_decrease_length=False,
              validate_no_decrease_width=False,
              validate_no_increase_length=False,
              validate_no_increase_width=False,
              *args, **kwargs):
    """Report (log) the changes in the shapes of the dataframe before and after an operation/s.
    
    Parameters:
        df (DataFrame): input dataframe.
        fun (object): function to apply on the dataframe.
        validate_equal_length (bool): Validate that the number of rows i.e. length of the dataframe remains the same before and after the operation. 
        validate_equal_width (bool): Validate that the number of columns i.e. width of the dataframe remains the same before and after the operation. 
        validate_equal_shape (bool): Validate that the number of rows and columns i.e. shape of the dataframe remains the same before and after the operation. 
    
    Keyword parameters:
        args (tuple): provided to `fun`.
        kwargs (dict): provided to `fun`.
    
    Returns:
        df (DataFrame): output dataframe.
    """
    d1={}
    d1['from']=df.shape
    if isinstance(fun,str):
        df = getattr(df, fun)(*args, **kwargs)
    else:
        df = fun(df,*args, **kwargs)
    d1['to']=df.shape
    log_shape_change(d1,fun=fun)
    if validate_equal_length: assert d1["from"][0]==d1["to"][0], (d1["from"][0],d1["to"][0])
    if validate_equal_width: assert d1["from"][1]==d1["to"][1], (d1["from"][1],d1["to"][1])
    if validate_no_decrease_length: assert d1["from"][0]<=d1["to"][0], (d1["from"][0],d1["to"][0])
    if validate_no_decrease_width: assert d1["from"][1]<=d1["to"][1], (d1["from"][1],d1["to"][1])
    if validate_no_increase_length: assert d1["from"][0]>=d1["to"][0], (d1["from"][0],d1["to"][0])
    if validate_no_increase_width: assert d1["from"][1]>=d1["to"][1], (d1["from"][1],d1["to"][1])
    if validate_equal_shape: assert d1["from"]==d1["to"], (d1["from"],d1["to"])
    return df

@pd.api.extensions.register_dataframe_accessor("log")
class log:
    """Report (log) the changes in the shapes of the dataframe before and after an operation/s.
        
    TODO:
        Create the attribures (`attr`) using strings e.g. setattr.
        import inspect
        fun=inspect.currentframe().f_code.co_name
    """    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    def __call__(self,col=None):
        if not col is None and col in self._obj:
            logging.info(f"nunique('{col}') = {self._obj[col].nunique()}")
        logging.info(f"shape = {self._obj.shape}")
        return self._obj
    def dropna(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='dropna',**kws)
    def drop_duplicates(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='drop_duplicates',**kws)    
    def drop(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='drop',**kws)    
    def query(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='query',**kws)    
    def filter_(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='filter',**kws)    
    def pivot(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='pivot',**kws)
    def pivot_table(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='pivot_table',**kws)    
    def melt(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='melt',**kws)
    def stack(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='stack',**kws)
    def unstack(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='unstack',**kws)
    def explode(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='explode',**kws)
    def merge(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='merge',**kws)
    def join(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='join',**kws)    
    def groupby(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun='groupby',**kws)
    ## rd
    def clean(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun=clean,**kws)
    def filter_rows(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun=filter_rows,**kws)
    def melt_paired(self,**kws):
        from roux.lib.df import log_apply
        return log_apply(self._obj,fun=melt_paired,**kws)