"""For processing individual pandas DataFrames/Series. Mainly used in piped operations."""

## logging
import logging

## data
import numpy as np
import pandas as pd

## internal
from roux.lib import to_rd

## beacuse NA in pandas
@to_rd
def reset_index(
    df,
    **kws,
):
    """
    Named levels only.
    """
    return df.reset_index(
        [i for i in df.index.names if i],
        **kws
    )

@to_rd
def loca(
    df,
    index=None,
    columns=None,
    # errors='raise', 
    verbose=False,
    ):
    """
    loc_available
    """
    if (index is None and columns is None):
        if verbose:
            logging.warning('(index is None and columns is None)')
        return df
        
    if index is None:
        index=df.index.tolist()
    if columns is None:
        columns=df.columns.tolist()
        
    d1 = {}
    d1["from"] = (len(index),len(columns))
    
    df=df.loc[
        (df.index.intersection(index),
        df.columns.intersection(columns)) 
    ]
    
    d1["to"] = df.shape
    log_shape_change(
        d1,
        fun='loca',
        # label=label
        )
    if d1["to"]==(0,0):
        logging.warning('loca: na')
    return df
   
@to_rd
def get_name(
    df1: pd.DataFrame,
    cols: list = None,
    coff: float = 2,
    out=None,
):
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
    if hasattr(df1, "name") and cols is None:
        name = df1.name
        name = name if isinstance(name, str) else list(name)
    elif cols is not None:
        name = df1.iloc[0, :][cols]
    else:
        l1 = get_constants(df1.select_dtypes(object))
        if len(l1) == 1:
            from roux.lib.set import list2str

            name = list2str(df1[l1[0]].unique())
        elif len(l1) <= coff:
            name = sorted(l1)
        elif len(l1) == 0:
            return
        else:
            logging.warning(f"possible names in here?: {','.join(l1)}")
            return
    if out == "list":
        if isinstance(name, str):
            name = [name]
    elif out == False:
        logging.info(name)
        return df1
    return name


@to_rd
def log_name(
    df1: pd.DataFrame,
    **kws_get_name,
):
    return get_name(df1, out=False, **kws_get_name)


@to_rd
def get_groupby_columns(df_):
    """Get the columns supplied to `groupby`.

    Parameters:
        df_ (DataFrame): input dataframe.

    Returns:
        columns (list): list of columns.
    """
    return df_.apply(lambda x: all(x == df_.name)).loc[lambda x: x].index.tolist()


@to_rd
def get_constants(df1):
    """Get the columns with a single unique value.

    Parameters:
        df1 (DataFrame): input dataframe.

    Returns:
        columns (list): list of columns.
    """
    return df1.nunique().loc[lambda x: x == 1].index.tolist()


## delete unneeded columns
@to_rd
def drop_unnamedcol(df):
    """Deletes the columns with "Unnamed" prefix.

    Parameters:
        df (DataFrame): input dataframe.

    Returns:
        df (DataFrame): output dataframe.
    """
    cols_del = [c for c in df.columns if "Unnamed" in c]
    return df.drop(cols_del, axis=1)


### alias
delunnamedcol = drop_unnamedcol


@to_rd
def drop_levelcol(df):
    """Deletes the potentially temporary columns names with "level" prefix.

    Parameters:
        df (DataFrame): input dataframe.

    Returns:
        df (DataFrame): output dataframe.
    """
    cols_del = [c for c in df.columns if "level" in c]
    return df.drop(cols_del, axis=1)


@to_rd
def drop_constants(
    df,
    verbose=True,
    ):
    """Deletes columns with a single unique value.

    Parameters:
        df (DataFrame): input dataframe.

    Returns:
        df (DataFrame): output dataframe.
    """
    if len(df) <= 1:
        logging.warning(f"skipped drop_constants because len(df)=={len(df)}")
        return df
    cols_del = get_constants(df)
    if len(cols_del) > 0:
        if verbose:
            logging.warning(f"dropped columns: {', '.join(cols_del)}")
        return df.drop(cols_del, axis=1)
    else:
        return df


@to_rd
def dropby_patterns(
    df1,
    patterns=None,
    strict=False,
    test=False,
    verbose=True,
    errors="raise",
):
    """Deletes columns containing substrings i.e. patterns.

    Parameters:
        df1 (DataFrame): input dataframe.
        patterns (list): list of substrings.
        test (bool): verbose.

    Returns:
        df1 (DataFrame): output dataframe.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    if patterns is None or patterns == []:
        return df1
    s0 = "|".join(patterns).replace("(", r"\(").replace(")", r"\)")
    s1 = f"{'^' if strict else ''}.*({s0}).*{'$' if strict else ''}"
    cols = df1.filter(regex=s1).columns.tolist()
    if test:
        logging.info(s1)
    if errors == "raise":
        assert len(cols) != 0
    if verbose:
        logging.info("columns dropped:" + ",".join(cols))
        return df1.log.drop(labels=cols, axis=1)
    else:
        return df1.drop(labels=cols, axis=1)


## columns reformatting
@to_rd
def flatten_columns(
    df: pd.DataFrame,
    sep: str = " ",
    **kws,
) -> pd.DataFrame:
    """Multi-index columns to single-level.

    Parameters:
        df (DataFrame): input dataframe.
        sep (str): separator within the joined tuples (' ').

    Returns:
        df (DataFrame): output dataframe.

    Keyword Arguments:
        kws (dict): parameters provided to `coltuples2str` function.
    """
    from roux.lib.str import tuple2str

    cols_str = []
    for col in df.columns:
        cols_str.append(tuple2str(col, sep=sep))
    df.columns = cols_str
    return df


@to_rd
def lower_columns(df):
    """Column names of the dataframe to lower-case letters.

    Parameters:
        df (DataFrame): input dataframe.

    Returns:
        df (DataFrame): output dataframe.
    """
    df.columns = df.columns.str.lower()
    return df


@to_rd
def renameby_replace(
    df: pd.DataFrame,
    replaces: dict,
    errors: bool = 'raise',
    **kws,
) -> pd.DataFrame:
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
    df.columns = [replacemany(c, replaces, errors=errors, **kws) for c in df]
    return df


@to_rd
def clean_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardise columns.

    Steps:
        1. Strip flanking white-spaces.
        2. Lower-case letters.

    Parameters:
        df (DataFrame): input dataframe.

    Returns:
        df (DataFrame): output dataframe.
    """
    df.columns = df.columns.str.strip().str.rstrip().str.lower()
    return df


@to_rd
def clean(
    df: pd.DataFrame,
    cols: list = [],
    drop_constants: bool = False,
    drop_unnamed: bool = True,
    verb: bool = False,
) -> pd.DataFrame:
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
    cols_del = (
        df.filter(
            regex="^(?:index|level|temporary|Unnamed|chunk|_).*$"
        ).columns.tolist()
        + df.filter(regex=r"^.*(?:\.1)$").columns.tolist()
        + cols
    )
    # exceptions
    cols_del = [c for c in cols_del if not c.endswith("0.1")]
    if drop_constants:
        df = df.rd.drop_constants()
    if not drop_unnamed:
        cols_del = [c for c in cols_del if not c.startswith("Unnamed")]
    if any(df.columns.duplicated()):
        #         from roux.lib.set import unique
        if verb:
            logging.warning(
                f"duplicate column/s dropped:{df.loc[:,df.columns.duplicated()].columns.tolist()}"
            )
        df = df.loc[:, ~(df.columns.duplicated())]
    if len(cols_del) != 0:
        if verb:
            logging.warning(f"dropped columns: {', '.join(cols_del)}")
        return df.drop(cols_del, axis=1)
    else:
        return df


@to_rd
def compress(
    df1: pd.DataFrame,
    coff_categories: int = None,
    verbose: bool = True,
):
    """Compress the dataframe by converting columns containing strings/objects to categorical.

    Parameters:
        df1 (DataFrame): input dataframe.
        coff_categories (int): if the number of unique values are less than cutoff the it will be converted to categories.
        verbose (bool): verbose.

    Returns:
        df1 (DataFrame): output dataframe.
    """
    if verbose:
        ini = df1.memory_usage().sum()
    ds = df1.select_dtypes("object").nunique()
    if coff_categories is not None:
        cols = ds[ds <= coff_categories].index
    else:
        cols = ds.index.tolist()
    for c in cols:
        logging.info(f"compressing '{c}'")
        df1[c] = df1[c].astype("category")
    if verbose:
        logging.info(f"compression={((ini-df1.memory_usage().sum())/ini)*100:.1f}%")
    return df1


@to_rd
def clean_compress(
    df: pd.DataFrame,
    kws_compress: dict = {},
    **kws_clean,
):
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

@to_rd
def to_sparse(
    df1,
    cols=None,
    dtypes=None,
    ):
    """
    
    if dtype=='object':
        df=(
            df
            .replace(
                [None,b'None'],
                np.nan,
            )
        )
    
    """
    to_fill_value={
        'float': np.nan,
        'int': 0,
        'bool':False,
    }
    if cols:
        if not dtypes:
            dtypes={c:v.name for c,v in df1[cols].dtypes.to_dict().items()}
            logging.info(dtypes)
        sdtypes={c: pd.SparseDtype(v, fill_value=to_fill_value.get(dtype,np.nan)) for c,v in dtypes.items()}
    else:
        if not dtypes:
            dtypes=df1.dtypes.unique()
            assert len(dtypes)==1,dtypes
            dtypes=dtypes[0].name
            logging.info(dtypes)
        sdtypes=pd.SparseDtype(dtypes, fill_value=to_fill_value.get(dtypes,np.nan))
        
    df1=(
        df1
        .astype(
              sdtypes
        )
    )
    sparse_density=df1.sparse.density
    getattr(logging,'info' if sparse_density<1 else 'warning')(f"sparse_density: {sparse_density}")
    # assert sparse_density<1, (sparse_density,sdtypes)
    return df1

## nans:
@to_rd
def check_na(
    df,
    subset=None,
    out=True,
    perc=False,
    log=True,
):
    """Number of missing values in columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (bool): output, else not which can be applicable in chained operations.

    Returns:
        ds (Series): output stats.
    """
    ## input parameters
    if subset is None:
        subset = df.columns.tolist()
    if isinstance(subset, str):
        subset = [subset]

    ds = df.loc[:, subset].isnull().sum()
    if perc:
        ds = (ds / len(df)) * 100
    if not out:
        str_log = to_str(ds)
        if log:
            logging.info(str_log)
            return df
        else:
            return str_log
    else:
        return ds


@to_rd
def validate_no_na(df, subset=None):
    """Validate no missing values in columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.

    Returns:
        ds (Series): output stats.
    """
    if subset is None:
        subset = df.columns.tolist()
    return not df.loc[:, subset].isnull().any().any()


@to_rd
def assert_no_na(df, subset=None):
    """Assert that no missing values in columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.

    Returns:
        ds (Series): output stats.
    """
    assert validate_no_na(df, subset=subset), check_na(df, subset=subset)
    return df


## nunique:
def to_str(
    data,
    log=False,
):
    if isinstance(data, pd.Series):
        data.index = [str(i) for i in data.index]
        return (
            data.to_csv(sep="\t")
            .split("\n", 1)[1]
            .rsplit("\n", 1)[0]
            .replace("\t", " = ")
            .replace("\n", "; ")
        )
    elif isinstance(data, (list, tuple)):
        return data[0] if not log else f'"{data[0]}"' if len(data) == 1 else str(data)
    else:
        # raise ValueError(type(data))
        raise ValueError(data)


@to_rd
def check_nunique(
    df: pd.DataFrame,
    subset: list = None,
    groupby: str = None,
    # expr: str = None,
    perc: bool = False,
    auto=True,
    out=True,
    log=True,
    plot=False,
    errors='raise', # keyerror
) -> pd.Series:
    """Number/percentage of unique values in columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.

    Returns:
        ds (Series): output stats.
    """
    ## TODO: need disconnected copy 
    # if expr is not None:
    #     assert not out, f"for clarity, filter separately. expr={expr} .. "
    #     df=df.copy(deep=True).query(expr=expr)
    if subset is None and auto:
        subset = df.select_dtypes((object, bool)).columns.tolist()
        logging.warning(f"Auto-detected columns (subset): {subset}")
    if isinstance(subset, str):
        subset = [subset]
    if errors=='raise':
        assert len(set(subset) - set(df.columns.tolist())) == 0, 'else use errors=None'
    else:
        subset=df.columns.intersection(subset).tolist()
        assert len(subset)!=0
        logging.warning(f'subset reduced to {subset}')
    if groupby is None:
        if not perc:
            ds_ = df.loc[:, subset].nunique()
        else:
            ds_ = (df.loc[:, subset].nunique() / df.loc[:, subset].agg(len)) * 100
    else:
        if isinstance(groupby, str):
            groupby = [groupby]

        if len(subset) == 1:
            ds_ = df.groupby(groupby)[subset].nunique()[subset[0]]
        else:
            ds_ = df.groupby(groupby).apply(
                lambda df: len(df.loc[:, subset].drop_duplicates())
            )
        
    ds_=ds_.sort_values(ascending=False)
    if plot!=False:
        ds_.sort_values(ascending=True).plot.barh(
            **({} if plot==True else plot)
        )
    if out:
        ## no logging
        return ds_
    else:
        str_log = (
            # f"{expr if expr is not None else ' ':}: "
            f"{'by '+to_str(groupby,log=True)+', nunique '+to_str(subset,log=True)+':' if groupby is not None else 'nunique:'} {to_str(ds_)}"
            )
        if log:
            logging.info(str_log)
            return df  # input
        else:
            return str_log


## nunique:
@to_rd
def check_inflation(
    df1,
    subset=None,
):
    """Occurances of values in columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.

    Returns:
        ds (Series): output stats.
    """
    if subset is None:
        subset = df1.columns.tolist()
    if subset is None:
        subset = df1.columns.tolist()
    return (
        df1.loc[:, subset]
        .apply(lambda x: (x.value_counts().values[0] / len(df1)) * 100)
        .sort_values(ascending=False)
    )


## duplicates:
@to_rd
def check_dups(
    df,
    subset=None,
    perc=False,
    out=True,
):
    """Check duplicates.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        perc (bool): output percentages.

    Returns:
        ds (Series): output stats.
    """
    if subset is None:
        subset = df.columns.tolist()
    df1 = df.loc[df.duplicated(subset=subset, keep=False), :].sort_values(by=subset)
    from roux.stat.io import perc_label  # noqa

    logging.info("duplicate rows: " + perc_label(len(df1), len(df)))
    if not out:
        return df
    elif not perc:
        return df1
    else:
        return 100 * (len(df1) / len(df))


@to_rd
def check_duplicated(
    df,
    **kws,
):
    """Check duplicates (alias of `check_dups`)"""
    return check_dups(df, **kws)


@to_rd
def validate_no_dups(
    df,
    subset=None,
    log: bool=True,
):
    """Validate that no duplicates.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
    """
    if subset is None:
        subset = df.columns.tolist()
    out = not df.duplicated(subset=subset).any()
    if not out and log:
        logging.warning("duplicate rows found")
    return out


@to_rd
def validate_no_duplicates(
    df,
    subset=None,
    **kws,
):
    """Validate that no duplicates (alias of `validate_no_dups`)"""
    return validate_no_dups(
        df,
        subset=subset,
        **kws,
    )


@to_rd
def assert_no_dups(df, subset=None):
    """Assert that no duplicates"""
    assert validate_no_dups(df, subset=subset), (
        check_dups(
            df, subset=subset, perc=False
        )
        .dropna(how='all',axis=1)
        .pipe(drop_constants)
    )
    return df

@to_rd
def drop_dups_by_agg(
    df1,
    subset,
    col_agg,
    std_max=0.05, ## standard deviation
    agg_func='mean',
    **kws_drop_duplicates,
    ):
    df2=df1.rd.check_dups(
        subset=subset
    )
    if len(df2)==0:
        return df1
    else:
        std_max_data=df2.groupby(subset)[col_agg].std().max()
        assert std_max_data<=std_max, std_max_data
        logging.info(f'std max found in data={std_max_data}')
        return pd.concat(
            [
                df1.log().drop_duplicates(
                    subset=subset,
                    keep=False,
                    **kws_drop_duplicates
                ),
                df2.groupby(subset)[col_agg].agg(agg_func).reset_index(),
            ],
            axis=0,
            ).log()

## asserts
@to_rd
def validate_dense(
    df01: pd.DataFrame,
    subset: list = None,
    duplicates: bool = True,
    na: bool = True,
    message=None,
    out_fmt=bool,
) -> pd.DataFrame:
    """Validate no missing values and no duplicates in the dataframe.

    Parameters:
        df01 (DataFrame): input dataframe.
        subset (list): list of columns.
        duplicates (bool): whether to check duplicates.
        na (bool): whether to check na.
        message (str): error message

    """
    if subset is None:
        subset = df01.columns.tolist()
    valids = {}
    if duplicates:
        valids['no_dups']=df01.rd.validate_no_dups(subset=subset)
    if na:
        valids['no_na']=df01.rd.validate_no_na(subset=subset)
    if out_fmt==bool:
        return all(list(valids.values()))
    else:
        return valids

@to_rd
def assert_dense(
    df01: pd.DataFrame,
    subset: list = None,
    duplicates: bool = True,
    na: bool = True,
    message=None,
) -> pd.DataFrame:
    """Alias of `validate_dense`.

    Notes:
        to be deprecated in future releases.
    """
    valids=validate_dense(
        df01,
        subset=subset,
        duplicates=duplicates,
        na=na,
        message=message,
        out_fmt=dict,
        )
    if all(list(valids.values())):
        return df01
    else:
        ## diagnose
        if valids.get('no_dups')==False:
            logging.error(
                check_dups(
                    df01,
                    subset=subset,
                    out=False,
                )
            )
        if valids.get('no_na')==False:            
            logging.error(
                check_na(
                    df01,
                    subset=subset,
                    out=False,
                )
            )
        assert False, valids

## counts
@to_rd
def assert_shape(
    df: pd.DataFrame,
    shape: int,
) -> pd.DataFrame:
    """Validate shape in pipe'd operations.

    Example:
        (
            df
            .rd.assert_shape((2,10))
        )
    """
    assert df.shape == shape, df.shape
    return df
    
@to_rd
def assert_len(
    df: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    """Validate length in pipe'd operations.

    Example:
        (
            df
            .rd.assert_len(10)
        )
    """
    assert len(df) == count, len(df)
    return df


@to_rd
def assert_nunique(
    df: pd.DataFrame,
    col: str,
    count: int,
) -> pd.DataFrame:
    """Validate unique counts in pipe'd operations.

    Example:
        (
            df
            .rd.assert_nunique('id',10)
        )
    """
    assert df[col].nunique() == count, df[col].nunique()
    return df


## mappings
@to_rd
def classify_mappings(
    df: pd.DataFrame,
    subset: list,
    clean: bool = False,
) -> pd.DataFrame:
    """Classify mappings between items in two columns using a vectorized approach.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): A list containing two column names to map.
        clean (bool): If True, drop the intermediate count columns.

    Returns:
        pd.DataFrame: The DataFrame with a new 'mapping' column.
    """
    assert len(subset) == 2, "Subset must contain exactly two column names."
    col1, col2 = subset

    count1_col = f"{col1} count"
    count2_col = f"{col2} count"

    # Use .assign() for a more functional, chainable style.
    # The lambda functions ensure that the operations are performed on the
    # DataFrame being passed through the chain.
    df_with_counts = df.assign(
        **{
            count1_col: lambda d: d.groupby(col2)[col1].transform("nunique"),
            count2_col: lambda d: d.groupby(col1)[col2].transform("nunique"),
        }
    )

    # Define the conditions for each mapping type based on the new columns.
    conditions = [
        (df_with_counts[count1_col] == 1) & (df_with_counts[count2_col] == 1),
        (df_with_counts[count1_col] == 1) & (df_with_counts[count2_col] > 1),
        (df_with_counts[count1_col] > 1) & (df_with_counts[count2_col] == 1),
    ]
    choices = ["1:1", "1:m", "m:1"]

    # Use np.select for vectorized conditional assignment and assign the new column.
    df_final = df_with_counts.assign(
        mapping=np.select(conditions, choices, default="m:m")
    )

    if clean:
        df_final = df_final.drop(columns=[count1_col, count2_col])

    return df_final


@to_rd
def check_mappings(
    df: pd.DataFrame,
    subset: list = None,
    out=True,
) -> pd.DataFrame:
    """Mapping between items in two columns.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (str): format of the output.

    Returns:
        ds (Series): output stats.
    """
    if subset is None:
        subset = df.columns.tolist()
    df1 = (
        classify_mappings(df, subset=subset, clean=False)
        .drop_duplicates(subset=subset)
        .groupby(["mapping", f"{subset[0]} count", f"{subset[1]} count"])
        .size()
        .to_frame("mappings count")
    )
    if out:
        return df1
    else:
        logging.info(f"mappings: {df1.to_string(index=False)}")
        return df

@to_rd
def assert_mappings(
    df: pd.DataFrame,
    validate='1:1',
    subset: list = None,
) -> pd.DataFrame:
    """Validate that the papping between items in two columns is 1:1.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (str): format of the output.

    """
    df1 = classify_mappings(
        df,
        subset=subset,
    )
    assert all(df1["mapping"] == validate), df1
    return df


@to_rd
def assert_1_1_mappings(
    df: pd.DataFrame,
    subset: list = None,
    **kws,
) -> pd.DataFrame:
    """Validate that the papping between items in two columns is 1:1.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): list of columns.
        out (str): format of the output.

    """
    return assert_mappings(
        df,
        subset=subset,
        validate="1:1",
        **kws,
    )

@to_rd
def get_mappings(
    df1: pd.DataFrame,
    subset=None,
    keep="all",
    clean=False,
    cols=None,
) -> pd.DataFrame:
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
    if cols is not None and subset is not None:
        logging.error("cols and subset are alias, both cannot be used.")
        return
    if cols is None and subset is not None:
        cols = subset
    if cols is None:
        cols = df1.columns.tolist()
    if not df1.rd.validate_no_dups(cols):
        df1 = df1.loc[:, cols].log.drop_duplicates()
    query_expr = None
    if isinstance(keep, str) and keep != "all":
        ## filter
        if keep in ["1:1", "1:m", "m:1", "m:m"]:
            query_expr = f"`mapping` == '{keep}'"
        elif ":" in keep:
            query_expr = f"`{subset[0]} count`=={keep.split(':')[0]} and `{subset[1]} count`=={keep.split(':')[1]}"
            clean = False  # override
        else:
            raise ValueError(keep)
    df2 = classify_mappings(df1, subset=cols, clean=clean)
    if query_expr is not None:
        df2 = df2.log.query(expr=query_expr)
    return df2


## binary
@to_rd
def to_map_binary(df: pd.DataFrame, colgroupby=None, colvalue=None) -> pd.DataFrame:
    """Convert linear mappings to a binary map

    Parameters:
        df (DataFrame): input dataframe.
        colgroupby (str): name of the column for groupby.
        colvalue (str): name of the column containing values.

    Returns:
        df1 (DataFrame): output dataframe.
    """
    colgroupby = [colgroupby] if isinstance(colgroupby, str) else colgroupby
    colvalue = [colvalue] if isinstance(colvalue, str) else colvalue
    if not df.rd.validate_no_dups(colgroupby + colvalue):
        logging.warning("duplicates found")
        df = df.log.drop_duplicates(subset=colgroupby + colvalue)
    return (
        df.assign(_value=True)
        .pivot(index=colvalue, columns=colgroupby, values="_value")
        .fillna(False)
    )


## intersections
@to_rd
def check_intersections(
    df: pd.DataFrame,
    colindex=None,  # 'samples'
    colgroupby=None,  # 'yticklabels'
    plot=False,
    **kws_plot,
) -> pd.DataFrame:
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
    if isinstance(df, pd.DataFrame):
        if not (colgroupby is None or colindex is None):
            if not all(df.dtypes == bool):
                #             if isinstance(colgroupby,str):
                # lin
                df1 = to_map_binary(df, colgroupby=colgroupby, colvalue=colindex)
                ds = df1.groupby(df1.columns.to_list()).size()
            elif isinstance(colgroupby, (str, list)):
                assert not df.rd.check_duplicated([colindex] + colgroupby)
                # map
                # df=df.set_index(colindex).loc[:,colgroupby]
                # ds=df.groupby(df.columns.tolist()).size()
                ds = df.groupby(colgroupby).nunique(colindex)
            else:
                logging.error("colgroupby should be a str or list")
        # else:
        #     # map
        #     ds=map2groupby(df)
    elif isinstance(df, pd.Series):
        ds = df
    # elif isinstance(df,dict):
    #     ds=dict2df(d1).rd.check_intersections(colindex='value',colgroupby='key')
    else:
        raise ValueError("data type of `df`")
    ds.name = (
        colindex
        if isinstance(colindex, str)
        else ",".join(colindex)
        if isinstance(colindex, list)
        else None
    )
    if plot:
        from roux.viz.bar import plot_intersections

        return plot_intersections(ds, **kws_plot)
    else:
        return ds


def get_totals(ds1):
    """Get totals from the output of `check_intersections`.

    Parameters:
        ds1 (Series): input Series.

    Returns:
        d (dict): output dictionary.
    """
    col = ds1.name if ds1.name is not None else 0
    df1 = ds1.to_frame().reset_index()
    return {c: df1.loc[df1[c], col].sum() for c in ds1.index.names}


# filter df
@to_rd
def query(
    df : pd.DataFrame,
    expr : str = None,  
    ## safety
    errors='raise',    
    ## log
    subset=None,
    groupby=None,
    kws_log={},
    
    **kws,
    ) -> pd.DataFrame:
    """
    Can query safely and log clause-wise
    """    
    if expr in [None,'']:
        # return safely (as opposed to base pandas)
        logging.warning(f"skipped because expr={expr}")
        return df
        
    assert ' and ' not in expr, expr 
    assert ' or ' not in expr, expr 
    assert not (expr.count('&')  > 0 and expr.count('|')  > 0), expr

    if subset is not None:
        kws_log['subset']=subset
    if groupby is not None:
        kws_log['groupby']=groupby
    
    from roux.lib.str import get_fills
    import re

    # Pre-calculate once outside the loop
    df_columns_set = set(df.columns)
    
    # Get all column names from the full expression
    expr_columns = set(get_fills(expr, "`"))
    
    # Check if any columns are missing
    missing_columns = expr_columns - df_columns_set
    
    if len(missing_columns) > 0:
        if errors=='raise':
            raise ValueError(f"expr '{expr}' contains missing columns: {list(missing_columns)}")
        # If there are missing columns, split and filter the expression
        clauses_and_ops = re.split(r'\s*(&|\|)\s*', expr)
        clauses = clauses_and_ops[0::2]
        operators = [''] + clauses_and_ops[1::2]
        
        valid_clauses = []
        
        for op, clause in zip(operators, clauses):
            # Optimization: only check if a clause is valid if it contains a column
            # that is NOT in the missing_columns set.
            # This is a key simplification, as we don't need to get all columns for a clause again.
            
            # The logic here is tricky. The simplest way to check for a missing column
            # within a clause is to get the columns for that clause again.
            # So, the original call `get_fills(clause, '`')` is necessary.
            
            clause_columns = set(get_fills(clause, '`'))
            if clause_columns.isdisjoint(missing_columns):
                valid_clauses.append(op + ' ' + clause.strip())
            else:
                missing_in_clause = clause_columns & missing_columns
                logging.warning(f"Query clause '{clause.strip()}' was omitted due to missing columns: {list(missing_in_clause)}")
    
        if not valid_clauses:
            logging.warning("nothing to query, returning the input!")
            return df
    
        final_query_expr = ' '.join(valid_clauses).strip()
        
        if final_query_expr.startswith(('&', '|')):
            final_query_expr = final_query_expr.split(' ', 1)[1]
        logging.warning(f"expr = '{final_query_expr}'")
    else:
        final_query_expr = expr
    
    if len(kws_log)==0:
        return df.log.query(
            expr=final_query_expr,
            **kws
        )
    else:
        # print(kws_log)
        df.log(label='(init)')
        for e in final_query_expr.split('&'):
            df=(
                df
                .query(
                    expr=e,
                    **kws
               )
               .log(
                   # label=f"(queried by: {get_bracket(e,'`','`')})",
                   label=f"({e.strip()})",
                   **kws_log,
               ) 
               )
        return df

@to_rd
def filter_rows(
    df,
    
    expr,
    mode='keep',
    
    ## by df
    
    ## by dict
    logic="&",
    
    drop_constants=False,
    
    fast=False,
    
    test=False,
    verbose=True,
):
    """Filter rows using a dataframe or dictionary.

    Parameters:
        df (DataFrame): input dataframe.
        expr (dict|pd.DataFrame): filter by dataframe or dictionary.
        sign (str): condition within mappings ('==').
        logic (str): condition between mappings ('and').
        drop_constants (bool): to drop the columns with single unique value (False).
        test (bool): testing (False).
        verbose (bool): more verbose (True).

    Returns:
        df (DataFrame): output dataframe.
    """
    # Quality check for the mode parameter
    if mode not in ['drop', 'keep']:
        raise ValueError(f"mode must be either 'drop' or 'keep', but got '{mode}'")
    
    if isinstance(expr,pd.DataFrame):
        df=df.log().reset_index(drop=True)
        # This mask will identify all rows that match at least one rule
        combined_mask_of_matches = pd.Series(False, index=df.index)
        
        # Identify the columns that define the hierarchy
        expr=(
            expr
                .drop_duplicates()
        )
        cols_id = expr.columns.tolist()        
        if len(set(cols_id) - set(df.columns.tolist()))>0:
            expr=(
                expr
                    ## only the common columns
                    .loc[
                        :,
                        list(set(cols_id) & set(df.columns.tolist()))
                    ]
            )
            cols_id = expr.columns.tolist()
            logging.warning(f"using cols_id: {cols_id}")

        # --- Modification: Pre-filter expr to keep only relevant rules ---
        # 1. Get unique values from the data
        relevant_values_ = {col: set(df[col].unique()) for col in cols_id}
    
        # 2. Build a boolean mask to identify relevant rules without iteration
        mask_relevant_rules_ = pd.Series(True, index=expr.index)
        for col, data_vals in relevant_values_.items():
            # Find rules for this column that have values not present in the data
            is_irrelevant_ = ~expr[col].isin(data_vals) & expr[col].notna()
            # Update the mask to exclude these irrelevant rules
            mask_relevant_rules_[is_irrelevant_] = False

        _len=len(expr)
        expr = expr[mask_relevant_rules_]
        if _len > len(expr):
            logging.warning(f"expr reduced to remove irrelevant rules: {_len} -> {len(expr)}")
        
        if expr.empty:
            return df.copy() if mode == 'drop' else df.iloc[0:0]
        # --- End Modification ---

        if not fast:
            # slow
            # Iterate over each rule in the rules DataFrame
            for _, rule in expr.iterrows():
                # Start with a mask of all True for the current rule
                current_rule_mask = pd.Series(True, index=df.index)
                
                # Sequentially apply filters for each non-NaN value in the rule
                for col in cols_id:
                    if pd.notna(rule[col]):
                        current_rule_mask &= (df[col] == rule[col])
                
                # Add the rows matching this rule to the combined mask of matches
                combined_mask_of_matches |= current_rule_mask
        else:
            # 1. Build a list of query strings, one for each rule
            rule_queries = []
            for _, rule in expr.iterrows():
                conditions = []
                for col, val in rule.items():
                    if pd.notna(val):
                        # repr(val) correctly handles strings vs. numbers
                        conditions.append(f"`{col}` == {repr(val)}")
                
                if conditions:
                    # Join conditions for a single rule with ' & '
                    rule_queries.append(f"({' & '.join(conditions)})")
                    
            if not rule_queries:
                return df.copy() if mode == 'drop' else df.iloc[0:0]
        
            # 2. Combine all rule queries with ' | ' to find any match
            full_query = ' | '.join(rule_queries)
        
            # 3. Use the high-performance `eval` engine to get a boolean mask in one pass
            combined_mask_of_matches = df.eval(full_query, engine='numexpr')
    
        # Apply the final filter based on the selected mode
        if mode == 'keep':
            # Return only the rows that match at least one rule
            return df[combined_mask_of_matches].log()
        else: # mode == 'keep'
            # Return rows that do NOT match any rule
            return df[~combined_mask_of_matches].log()
        
    elif isinstance(expr,dict):
        if verbose:
            logging.info(df.shape)

        # if mode=='keep':
        #     sign=" == "
        # else:
        #     sign=" != "            
        # assert all([isinstance(expr[k], (str, list)) for k in expr])
        # qry = f" {logic} ".join(
        #     [
        #         f"`{k}` {sign} " + (f'"{v}"' if isinstance(v, str) else f"{v}")
        #         for k, v in expr.items()
        #     ]
        # )
        
        from roux.lib.str import to_expr
        expr=to_expr(
            expr,
            mode=mode,
            logic=logic, 
        )
        
        df1 = df.query(expr=expr)
        if test:
            logging.info(df1.loc[:, list(expr.keys())].drop_duplicates())
            logging.warning("may be some column names are wrong..")
            logging.warning([k for k in d if k not in df])
        if verbose:
            logging.info(df1.shape)
        if drop_constants:
            df1 = df1.rd.drop_constants()
        return df1
    else:
        raise ValueError(expr)
        
@to_rd
def tri(df,k=-1,**kws_np_tri):
    """
    Get tringle of adjacency matrix.
    """
    return df.where(np.tri(*(df.shape),k,**kws_np_tri).astype('bool'))

## conversion to type
@to_rd
def to_dict(
    df: pd.DataFrame,
    cols: list = None,
    drop_duplicates: bool = False,
):
    """DataFrame to dictionary.

    Parameters:
        df (DataFrame): input dataframe.
        cols (list): list of two columns: 1st contains keys and second contains value.
        drop_duplicates: whether to drop the duplicate values (False).

    Returns:
        d (dict): output dictionary.
    """
    if cols is None and df.shape[1] == 2:
        cols = df.columns.tolist()
    df = df.log.dropna(subset=cols)
    if drop_duplicates:
        df = df.loc[:, cols].drop_duplicates()
    if not df[cols[0]].duplicated().any():
        return df.set_index(cols[0])[cols[1]].to_dict()
    else:
        logging.warning("format: {key:list}")
        assert df[cols[1]].dtype == "O", df[cols[1]].dtype
        return df.groupby(cols[0])[cols[1]].unique().to_dict()


## to avoid overlap with `io_dict.to_dict`
del to_dict

## conversion
## deprecated: use pd.get_dummies(.. columns=)
# @to_rd
# def get_bools(df,cols,drop=False):
#     """Columns to bools. One-hot-encoder (`get_dummies`).

#     Parameters:
#         df (DataFrame): input dataframe.
#         cols (list): columns to encode.
#         drop (bool): drop the `cols` (False).

#     Returns:
#         df (DataFrame): output dataframe.
#     """
#     df=df.reset_index(drop=True) # because .join later
#     for c in cols:
#         df_=pd.get_dummies(
#             df[c],
#               prefix=c,
#               prefix_sep=": ",
#               dummy_na=False,
#         )
#         df_=df_.replace(1,True).replace(0,False)
#         df=df.join(df_)
#         if drop:
#             df=df.drop([c],axis=1)
#     return df


@to_rd
def agg_bools(df1, cols):
    """Bools to columns. Reverse of one-hot encoder (`get_dummies`).

    Parameters:
        df1 (DataFrame): input dataframe.
        cols (list): columns.

    Returns:
        ds (Series): output series.
    """
    col = "+".join(cols)
    #     print(df1.loc[:,cols].T.sum())
    assert all(df1.loc[:, cols].T.sum() == 1)
    for c in cols:
        df1.loc[df1[c], col] = c
    return df1[col]

## reshape
@to_rd
def explode(
    df1,
    col
):
    """
    Notes:
        -[x] dicts in a col.
    """
    df1=df1.reset_index(drop=True)
    return df1.join(pd.DataFrame(df1[col].tolist())).drop(columns=col)

## paired dfs
@to_rd
def melt_paired(
    df: pd.DataFrame,
    cols_index: list = None,  # paired
    suffixes: list = None,
    cols_value: list = None,
    clean: bool = False,
) -> pd.DataFrame:
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
        assert not (
            cols_index is None and suffixes is None
        ), "either cols_index or suffixes needed"
        if suffixes is None and cols_index is not None:
            from roux.lib.str import get_suffix

            suffixes = get_suffix(*cols_index, common=False, clean=True)

        # both suffixes should not be in any column name
        assert not any(
            [all([s in c for s in suffixes]) for c in df]
        ), "both suffixes should not be in a single column name"
        assert not any(
            [c == s for s in suffixes for c in df]
        ), "suffix should not be the column name"
        assert all(
            [any([s in c for c in df]) for s in suffixes]
        ), "both suffix should be in the column names"

        cols_common = [c for c in df if not any([s in c for s in suffixes])]
        dn2df = {}
        for s in suffixes:
            cols = [c for c in df if s in c]
            dn2df[s] = df.loc[:, cols_common + cols].rename(
                columns={c: c.replace(s, "") for c in cols}, errors="raise"
            )
        df1 = pd.concat(dn2df, axis=0, names=["suffix"]).reset_index(0)
        df2 = df1.rename(
            columns={
                c: c[:-1] if c.endswith(" ") else c[1:] if c.startswith(" ") else c
                for c in df1
            },
            errors="raise",
        )
        if "" in df2:
            df2 = df2.rename(columns={"": "id"}, errors="raise")
        assert len(df2) == len(df) * 2
        if clean:
            df2 = df2.drop(["suffix"], axis=1)
        return df2
    else:
        assert suffixes is not None
        import itertools

        df2 = pd.concat(
            {
                c: df.rename(
                    columns={f"{c} {s}": f"value {s}" for s in suffixes}, errors="raise"
                )
                for c in cols_value
            },
            axis=0,
            names=["variable"],
        ).reset_index(0)
        if len(cols_value) > 1:
            df2 = df2.drop(
                [f"{c} {s}" for c, s in itertools.product(cols_value, suffixes)], axis=1
            )
        assert len(df2) == len(df) * len(cols_value)
        return df2

@to_rd
def replace_inf(
    df,
    subset,
    value=None, #np.nan
    ):    
    """
    Replaces with the closest np.inf:max, -np.inf:min
    
    if not pandas.options.mode.use_inf_as_na = True
    """
    if isinstance(subset,str):
        subset=[subset]    
        
    for c in subset:    
        if df[c].max()==np.inf:
            logging.info(f"{c} == inf count = {sum(df[c]==np.inf)}")            
            df[c]=df[c].replace(
                np.inf,
                value if value is not None else df[c].replace([np.inf], np.nan).max()
            )
        if df[c].min()== -np.inf:
            logging.info(f"{c} == -inf count = {sum(df[c]== -np.inf)}")
            df[c]=df[c].replace(
                -np.inf,
                value if value is not None else df[c].replace([-np.inf], np.nan).min()
            )
    return df
    
## helper to get_bins

@to_rd
def get_qbins(df: pd.DataFrame, col: str, bins: list, labels: list = None, **kws_qcut):
    return df.assign(
        **{
            f"{col} bin": lambda df: pd.qcut(
                df[col], q=bins, labels=labels, **kws_qcut
            ),
        },
    )
    
@to_rd
def get_bins(
    df: pd.DataFrame,
    col: str,
    bins: list,
    kind: str= '',
    
    labels: list = None,
    labels_fmt: str = None,  # e.g. min

    dtype=None, # =labels_fmt
    
    **kws_cut,
):
    """
    kind: quantile
    """
    if dtype is not None:
        if labels_fmt is not None:
            raise ValueError(
                'remove deprec.d arg: dtype'
            )
        else:
            ## bc
            labels_fmt=dtype
            del dtype
    
    df=replace_inf(
        df,
        subset=col,
    )
    
    if kind.startswith('q'):
        return get_qbins(
            df,#: pd.DataFrame,
            col=col,#: str,
            bins=bins,#: list,
            labels=labels,#: list = None,
            **{k:v for k,v in kws_cut.items() if k not in ['include_lowest']}, ## **kws_qcut
        )
        ## label auto
    else:
        if isinstance(bins,int):
            ## for labeling
            bins=list(
                    np.linspace(
                        df[col].min(),
                        df[col].max(),
                        bins+1,
                    )
                )
            logging.info(f"bins={bins}" )
        if labels is None:
            from roux.lib.str import get_bin_labels
            labels = get_bin_labels(
                bins=bins,
                fmt=labels_fmt,
            )
        return df.assign(
            **{
                f"{col} bin": lambda df: pd.cut(
                    df[col], bins=bins,
                    labels=labels,
                    **kws_cut
                ),
            },
        )


@to_rd
def get_chunks(
    df,
    size=None,
    n=None,
    ):
    assert not (size is None and n is None)
    assert (size is not None or n is not None)
    if n is None:
        n=(len(df)//size)+1
    if size is None:
        size=(len(df)//n)+1
        
    return (
        df
        .assign(
            chunk=np.array([np.repeat(i,size) for i in range(n)]).ravel()[:len(df)],
        )
        .log('chunk')
    )


@to_rd
def sample_near_quantiles(
    data: pd.DataFrame,
    col: str,
    n: int,
    clean: bool = False,
):
    """
    Get rows with values closest to the quantiles.
    """
    dfs = {}
    for q in np.linspace(0, 1, n):
        dfs[q] = data.iloc[(data[col] - data[col].quantile(q)).abs().argsort()[:1]]
    df1 = pd.concat(dfs, axis=0, names=["q"])
    if clean:
        df1 = df1.reset_index(drop=True)
    return df1


## GROUPBY
# aggregate dataframes
def get_group(
    groups,
    i: int = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Get a dataframe for a group out of the `groupby` object.

    Parameters:
        groups (object): groupby object.
        i (int): index of the group. default None returns the largest group.
        verbose (bool): verbose (True).

    Returns:
        df (DataFrame): output dataframe.

    Notes:
        Useful for testing `groupby`.
    """
    if i is not None:
        dn = list(groups.groups.keys())[i]
    else:
        dn = groups.size().sort_values(ascending=False).index.tolist()[0]
    logging.info(f"sampled group name: {dn}")
    df = groups.get_group(dn)
    df.name = dn
    return df


@to_rd
def groupby_sample(
    df: pd.DataFrame,
    groupby: list,
    i: int = None,
    **kws_get_group,
) -> pd.DataFrame:
    """
    Samples a group (similar to .sample)

    Parameters:
        df (pd.DataFrame): input dataframe.
        groupby (list): columns to group by.
        i (int): index of the group. default None returns the largest group.

    Keyword arguments:
        keyword parameters provided to the `get_group` function

    Returns:
        pd.DataFrame
    """
    return get_group(df.groupby(by=groupby), **kws_get_group)

@to_rd
def groupby_sort_values(
    df: pd.DataFrame,
    groupby: str,
    col: str,
    func: str,
    col_temp : str ='temp',
    ascending=True,
    **kws_sort_values,
) -> pd.DataFrame:
    """
    Groupby and sort

    Parameters:
        df (pd.DataFrame): input dataframe.
        groupby (list): columns to group by.

    Keyword arguments:
        keyword parameters provided to the `.sort_values` attribute

    Returns:
        pd.DataFrame
    """
    return (
        df
        .assign(
            **{
                col_temp:lambda df: df.groupby(groupby)[col].transform(func),
            },
        )
        .sort_values(
            col_temp,
            ascending=ascending,
            **kws_sort_values,
            )
        .drop(
            [col_temp],
            axis=1
        )
    )

@to_rd
def groupby_agg_nested(
    df1: pd.DataFrame,
    groupby: list,
    subset: list,
    func: dict = None,
    cols_value: list = None,
    verbose: bool = False,
    **kws_agg,
) -> pd.DataFrame:
    """
    Aggregate serially from the lower level subsets to upper level ones.

    Parameters:
        df1 (pd.DataFrame): input dataframe.
        groupby (list): groupby columns i.e. list of columns to be used as ids in the output.
        subset (list): nested groups i.e. subsets.
        func (dict): map betweek columns with value to aggregate and the function for aggregation.
        cols_value (list): columns with value to aggregate, (optional).
        verbose (bool): verbose.

    Keyword arguments:
        kws_agg : keyword arguments provided to pandas's `.agg` function.

    Returns:
        output dataframe with the aggregated values.
    """

    def _agg(
        df2,
        cols_groupby,
        func,
        **kws_agg,
    ):
        if df2.loc[:, cols_groupby].duplicated().any():
            return (
                df2.groupby(cols_groupby)
                .agg(func=func, **kws_agg)
                .reset_index()
                .log(suffix="after a round of aggregation.")
            )
        else:
            return df2

    ## infer inputs
    if func is None:
        if cols_value is not None:
            func = {c: np.mean for c in cols_value}
    else:
        cols_value = list(func.keys())

    ds_ = df1.log(subset).loc[:, subset].nunique().sort_values(ascending=False)
    cols_groupby = list(set(groupby + subset))
    if verbose:
        logging.info(cols_groupby)

    df2 = _agg(
        df1,
        cols_groupby,
        func,
        **kws_agg,
    )
    for col in ds_.index:
        cols_groupby = list(set(cols_groupby) - set([col]))
        if verbose:
            logging.info(cols_groupby)
        df2 = _agg(
            df2,
            cols_groupby,
            func,
            **kws_agg,
        )
        if set(groupby) == set(cols_groupby):
            break
    return df2


@to_rd
def groupby_filter_fast(
    df1: pd.DataFrame,
    col_groupby,
    fun_agg,
    expr,
    col_agg: str = "temporary",
    **kws_query,
) -> pd.DataFrame:
    """Groupby and filter fast.

    Parameters:
        df1 (DataFrame): input dataframe.
        by (str|list): column name/s to groupby with.
        fun (object): function to filter with.
        how (str): greater or less than `coff` (>|<).
        coff (float): cut-off.

    Returns:
        df1 (DataFrame): output dataframe.

    Todo:
        Deprecation if `pandas.core.groupby.DataFrameGroupBy.filter` is faster.
    """
    assert col_agg in expr, f"{col_agg} not found in {expr}"
    df1[col_agg] = df1.groupby(col_groupby).transform(fun_agg)
    return df1.log.query(expr=expr, **kws_query)


# index
@to_rd
def infer_index(
    data: pd.DataFrame,
    cols_drop=[],
    include=object,
    exclude=None,
) -> list:
    """
    Infer the index (id) of the table.


    """    
    cols=(
        data
        .drop(cols_drop, axis=1)
        .select_dtypes(
            include='object',
            exclude=None,
        )
        .nunique()
        .sort_values(ascending=False)
        .to_frame('nunique')
        .reset_index()
        .query(expr="`nunique`>1")
        ['index'].tolist()
    )
    
    cols_id=[]
    for c in cols:
        cols_id+=[c]
        if data.rd.validate_no_dups(
            subset=cols_id,
            log=False,
            ):
            return cols_id

## multiindex
@to_rd
def to_multiindex_columns(df, suffixes, test=False):
    """Single level columns to multiindex.

    Parameters:
        df (DataFrame): input dataframe.
        suffixes (list): list of suffixes.
        test (bool): verbose (False).

    Returns:
        df (DataFrame): output dataframe.
    """
    cols = [
        c for c in df if c.endswith(f" {suffixes[0]}") or c.endswith(f" {suffixes[1]}")
    ]
    if test:
        logging.info(cols)
    df = df.loc[:, cols]
    df = df.rename(
        columns={
            c: (s, c.replace(f" {s}", ""))
            for s in suffixes
            for c in df
            if c.endswith(f" {s}")
        },
        errors="raise",
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


## ranges
@to_rd
def to_ranges(
    df1,
    colindex,
    colbool=None,
    sort=True,
    agg: dict={},
    interval=1,
    return_ranges_only=False,
    clean=True,
    verbose=True,
    ):
    """Ranges from boolean columns.

    Parameters:
        df1 (DataFrame): input dataframe.
        colindex (str): column containing index items.
        colbool (str): column containing boolean values.
        sort (bool): sort the dataframe (True).
        agg (bool): extra columns to agg. format: {col_renamed: (col,agg_func)}. (defaults to {}).

    Returns:
        df1 (DataFrame): output dataframe.

    TODO:
        compare with io_sets.bools2intervals.
    """        
    # import scipy as sc

    if sort:
        df1 = df1.sort_values(by=colindex)

    if colbool is None:
        if verbose:
            logging.info("setting consecutive values as ranges ..")
        colbool='_range'
        df1=df1.assign(
            **{
                colbool:lambda df : (df[colindex]+interval)==df[colindex].shift(-interval),
            },
        )
        
    col_group='_group'  
    
    df1=df1.assign(
        **{
            # col_group: sc.ndimage.measurements.label(df1[colbool].astype(int))[0],
            col_group: lambda df: (df[colindex].diff() != interval).cumsum()
        },
    )
    
        
    df2=(
        df1
        .groupby(col_group)
        .agg(
            **{
                **{
                    f"{colindex} min": (colindex, 'min'),
                    f"{colindex} max": (colindex, 'max'),
                },
                **agg
            }
        )
        .reset_index()
    )
        
    if return_ranges_only:
        if verbose:
            logging.info("returning ranges only ..")        
        df2=df2.query(expr=f"`{colindex} min` !=0 `{colindex} max`")
        
    if clean:
        df2=df2.drop([col_group],axis=1)
        
    return df2

## Assign bools
@to_rd
def assignby_expr(
    df,
    expr, # .query-style
    col=None,
    clean=True,
    verbose=False,
):
    if isinstance(expr,dict):
        ## recurse
        if col is not None and clean:
            ## -> one col
            sep_temp='__'
        else:
            sep_temp=''
        cols_temp=[]
        for c,e in expr.items():
            col_temp=f"{sep_temp}{c}"
            df=assignby_expr(
                df,
                expr=e, # .query-style
                col=col_temp,
                verbose=verbose,
            )
            cols_temp.append(col_temp)
        if col is None:
            return df
        else:
            df=df.reset_index(drop=True)
            ## collapse dummies
            df[col]=pd.from_dummies(df.loc[:,cols_temp])[''].str.replace(sep_temp,'')
            if clean:
                df=df.drop(cols_temp,axis=1)
            return df
            
    if col is None:
        col=expr
    df=df.reset_index(drop=True)
    df_=df.query(expr=expr)
    df[col]=False
    df.loc[df_.index.tolist(),col]=True
    if verbose:
        df.log(groupby=col)
    return df

@to_rd
def assign_bool(
    df1,
    expr, # {True: expr, False: expr}
    col, #output
    fillna=None, ## unassignd -> np.nan
    clean=True,
    verbose=False,
    # validate=False,
    ):
    """
    """ 
    if fillna is None:
       fillna=np.nan 
    
    if isinstance(expr,str):
        ## for True
        expr={
            True: expr,
        }        
    assert isinstance(expr,dict), expr
    
    for k in [True,False]:
        if str(k) in expr and k not in expr:
            expr[k]=expr[str(k)]
            del expr[str(k)]
            
    ## pre. to expr
    if True not in expr:
        if len(expr)==1:
            # col: [list -> True] 
            expr_in=expr.copy() 
            del expr
            
            col_in=list(expr_in.keys())[0]
            
            cats_in=list(expr_in.values())[0]
            if isinstance(cats_in,str):
                cats_in=[cats_in]
            expr={
                True: f"`{col_in}` == {cats_in}",
                # False: f"`{col_in}` == {list(set(df1[col_in].unique()) - set(cats_in))}",      
            }
    assert len(expr)<=2, expr
    if len(expr)==2:
        assert len(set(expr.values()))==2, expr
        
    if verbose:
        logging.info(f"expr(s):{expr}")
        
    for b,e in expr.items():
        df1=(
            df1
            .rd.assignby_expr(
                expr=e,
                col=f"_{b}",
                verbose=verbose,
            )
        )
    if '_False' not in df1:
        df1=df1.assign(
            **{
                '_False': lambda df: ~(df['_True']),
            }
        )
    ## combine the True and False
    df1=(
        df1
        .assign(
            **{
                col: lambda df: df.apply(
                    lambda x: (
                        True if x['_True'] and not x['_False'] else 
                        False if x['_False'] and not x['_True'] else 
                        fillna
                    ),
                    axis=1,
                )
            }   
        )
    )
    if clean:
        df1=df1.drop(['_True','_False'],axis=1,errors='ignore')
    if verbose:
        df1.log(groupby=col)        
    return df1
    
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
    low, high = np.array(df1).T[:, :, None]
    a = np.arange(high.max() + 1)
    return ((a >= low) & (a <= high)).any(axis=0)

## sorting
@to_rd
def sort_values(
    df,
    by: dict,
    **kws,
):
    if isinstance(by,dict): 
        kws['by'],kws['ascending']=list(by.keys()),list(by.values())
    return df.sort_values(**kws)

@to_rd
def astype_cat(
    df1: pd.DataFrame,
    col: str,
    cats: list,
):
    from roux.lib.ds import to_cat
    return df1.assign(
        **{
            col: lambda df: to_cat(df[col], cats=cats, ordered=True),
        }
    )


@to_rd
def sort_valuesby_list(
    df1: pd.DataFrame, by: str, cats: list, by_more: list = [], **kws
):
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
    return astype_cat(df1, col=by, cats=cats).sort_values(by=by, **kws)


## apply_agg
def agg_by_order(x, order):
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
    if len(x) == 1:
        #         print(x.values)
        return list(x.values)[0]
    for k in order:
        if k in x.values:
            return k


def agg_by_order_counts(x, order):
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
    ds = x.value_counts()
    ds = ds.add_prefix(f"{x.name}=")
    ds[x.name] = agg_by_order(x, order)
    return ds.to_frame("").T


@to_rd
def swap_paired_cols(df_, suffixes=["gene1", "gene2"]):
    """Swap suffixes of paired columns.

    Parameters:
        df_ (DataFrame): input dataframe.
        suffixes (list): suffixes.

    Returns:
        df (DataFrame): output dataframe.
    """
    rename = {
        c: c.replace(suffixes[0], suffixes[1])
        if (suffixes[0] in c)
        else c.replace(suffixes[1], suffixes[0])
        if (suffixes[1] in c)
        else c
        for c in df_
    }
    return df_.rename(columns=rename, errors="raise")


@to_rd
def sort_columns_by_values(
    df: pd.DataFrame,
    subset: list,
    suffixes: list = None,  # no spaces
    order: list = None,
    clean=False,
) -> pd.DataFrame:
    """Sort the values in columns in ascending order.

    Parameters:
        df (DataFrame): input dataframe.
        subset (list): columns.
        suffixes (list): suffixes.
        order (list): ordered list.

    Returns:
        df (DataFrame): output dataframe.

    Notes:
        In the output dataframe, `sorted` means values are sorted because gene1>gene2.
    """
    assert len(subset) == 2, subset
    if suffixes is None:
        from roux.lib.str import get_suffix

        suffixes = get_suffix(*subset, common=False, clean=True)
        assert set(suffixes) != set(subset), subset
        logging.info(f"suffixes inferred: {suffixes}")

    ## data checks
    df.rd.assert_no_na(subset=subset)

    if order is not None:
        ## ranks
        ranks = {s: i for i, s in enumerate(order)}
        df = df.assign(
            **{
                f"_rank {subset[0]}": lambda df: df[subset[0]].map(ranks),
                f"_rank {subset[1]}": lambda df: df[subset[1]].map(ranks),
            }
        )
        subset = [f"_rank {subset[0]}", f"_rank {subset[1]}"]

    suffixes = [s.replace(" ", "") for s in suffixes]
    dn2df = {}
    # keys: (equal, to be sorted)
    dn2df[(False, False)] = df.loc[(df[subset[0]] < df[subset[1]]), :]
    dn2df[(False, True)] = df.loc[(df[subset[0]] > df[subset[1]]), :]
    dn2df[(True, False)] = df.loc[(df[subset[0]] == df[subset[1]]), :]
    dn2df[(True, True)] = df.loc[(df[subset[0]] == df[subset[1]]), :]
    ## rename columns of of to be sorted
    ## TODO: use swap_paired_cols
    rename = {
        c: c.replace(suffixes[0], suffixes[1])
        if (suffixes[0] in c)
        else c.replace(suffixes[1], suffixes[0])
        if (suffixes[1] in c)
        else c
        for c in df
    }

    for k in [True, False]:
        dn2df[(k, True)] = dn2df[(k, True)].rename(columns=rename, errors="raise")

    df1 = pd.concat(dn2df, names=["equal", "sorted"]).reset_index([0, 1])
    logging.info(
        f"(equal, sorted) items: {df1.groupby(['equal','sorted']).size().to_dict()}"
    )
    if clean:
        df1 = df1.drop(
            ["equal", "sorted"] + (subset if order is not None else []),
            axis=1,
        )
    return df1

def _log_df_query(
    df1,
    expr,
):
    ## log relevant cols only
    from roux.lib.str import get_fills
    cols=get_fills(expr,marks='`')
    try:
        df1=(
            df1
                .loc[:,cols]
        )
    except:
        pass
    return (
        df1
            .query(expr=expr)
    )
        
@to_rd
def assert_expr(
    df1,
    expr,
    **kws
    ):
    assert df1.query(expr=expr,**kws).shape[0]==df1.shape[0], _log_df_query(df1,f"~({expr})") 
    return df1
    
## paired stats
from roux.viz import apply_plot

@to_rd
def check_corr(
    data,
    x,
    y,
    cols_id=None,
    method='spearman',
    resample=False,
    verbose=True,

    validate=None, 
    
    plot=False,    
    
    ## to be deprecated
    kws_plot={},
    kws_plot_set={},
    ax=None,

    out=False,
    **kws_get_corr,
    ):
    if validate in [False,'']:
        validate=None
    if cols_id is not None:
        logging.warning("cols_id is not implemented.")
    kws_stat={
        **dict(
            method=method,
            resample=resample,        
        ),
        **kws_get_corr,
    }
    if plot in [False, None]:
        from roux.stat.corr import get_corr
        res = get_corr(
            data[x],
            data[y],
            
            verbose=verbose,
            **kws_stat,
        )    
    else:    
        try:
            if not isinstance(plot,dict):
                plot={}
            
            ## to be removed part
            if len(kws_plot)>0 or len(kws_plot_set)>0 or ax is not None:
                logging.warning('kws_plot,kws_plot_set and ax  will be deprecated, provide them in the plot arg. instead')

                plot={
                    **dict(
                        set=kws_plot_set,
                        ax=ax,
                        )
                    ## override
                    **{
                        **kws_plot,
                        **plot,
                    }
                    }
                logging.warning(f"plot={plot}")

            from roux.viz.scatter import plot_scatter
            ax=apply_plot(
                plot_scatter,
                data,
                kws_plot={
                        **dict(
                            plot=dict(
                                x=x,
                                y=y,
                                stat_kws=kws_stat,
                            ),
                        ),
                        **plot
                    },
                )
            res=ax.stats
        except Exception as e:
            logging.exception(str(e))            
            ## return stats only
            return check_corr(
                data=data, #,
                x=x, #,
                y=y, #,
                method=method, #='spearman',
                resample=resample, #=False,
                verbose=verbose, #=True,

                validate=validate, #=None, 
                
                plot=False,    
                
                out=out, #=False,
                **kws_get_corr,
                )
    df1=pd.Series(res).to_frame().T
    if verbose:
        logging.info(f'{data.name if hasattr(data,"name") else ""}{x} - {y}\n'+df1.to_string(index=False))
    if validate is not None:
        assert_expr(df1,validate)
        # logging.info(df1)
    if out:
        return df1
    else:
        #pipe
        return data

@to_rd
def check_diff(
    data,
    x,
    y,
    cols_id,
    method=None, # mannwhitneyu

    order: list = None, 
    hue: str = None, ## subcategories compared
    hue_order: list = None, 

    out=False, # stats
    
    plot=False, 
    
    ## to be deprecated
    kws_plot={},
    kws_plot_set={},
    ax=None, 

    validate=None, 
    verbose=True,
    **kws_stats,
    ):
    
    kws_diff=dict(
        x=x,
        y=y,
        colindex=cols_id,

        order=order, 
        hue=hue, ## subcategories compared
        hue_order=hue_order, 
    )
    kws_stats=dict(
                func=method,
            )
    if validate in [False,'']:
        validate=None
    if plot in [False, None]:
        from roux.stat.diff import get_diff_inferred
        res,_=get_diff_inferred(
            data,
            kws_stats=kws_stats,
            **kws_diff,
        )
    else:
        try:
            if not isinstance(plot,dict):
                plot={}
            
            ## to be removed part
            if len(kws_plot)>0 or len(kws_plot_set)>0 or ax is not None:
                logging.warning('kws_plot,kws_plot_set and ax  will be deprecated, provide them in the plot arg. instead')

                plot={
                    **dict(
                        set=kws_plot_set,
                        ax=ax,
                        )
                    ## override
                    **{
                        **kws_plot,
                        **plot,
                    }
                    }
                logging.warning(f"plot={plot}")

            from roux.viz.dist import plot_dists
            ax=apply_plot(
                plot_dists,
                data,
                kws_plot={
                        **dict(
                            plot=kws_diff,
                        ),
                        **plot
                    },
                )
            res=ax.stats
        except Exception as e:
            logging.exception(str(e))            
            ## return stats only        
            return check_diff(
                data=data, #,
                x=x, #,
                y=y, #,
                cols_id=cols_id, #,
                method=method, #=None, # mannwhitneyu

                order=order, #: list = None, 
                hue=hue, #: str = None, ## subcategories compared
                hue_order=hue_order, #: list = None, 

                out=out, #=False, # stats
                
                plot=False, 
                
                validate=validate, #=None, 
                verbose=verbose, #=True,
                **kws_stats,
                )
        res=ax.stats
    # df1=pd.Series(res).to_frame().T
    df1=res
    if df1 is None:
        return None
    if verbose:
        logging.info(f'{data.name if hasattr(data,"name") else ""}{x} - {y}\n'+df1.to_string(index=False))
    if validate is not None:
        assert_expr(df1,validate)
        # logging.info(df1)
    if out:
        return df1
    else:
        #pipe
        return data

@to_rd
def check_sass(
    data,
    x,
    y,
    order_x=None,
    order_y=None,

    method=None, 
    cols_id=None, ## not implemented
    verbose=True,

    validate=None, 
    
    plot=False, 
    
    ## to be deprecated
    kws_plot={},
    kws_plot_set={},
    ax=None, 

    out=False,
    **kws_stats,
    ):

    if validate in [False,'']:
        validate=None
    if cols_id is not None:
        logging.warning("cols_id is not implemented.")
        
    kws_stats={
        **kws_stats,
        **dict(
            method=method,
            order_x=order_x,
            order_y=order_y,
        ),
    }
    
    if plot in [False, None]:
        from roux.stat.diff import compare_classes
        res = compare_classes(
            # data=data,
            x=data[x],
            y=data[y],
            
            **kws_stats
        )
    else:
        try:
            if not isinstance(plot,dict):
                plot={}
            
            ## to be removed part
            if len(kws_plot)>0 or len(kws_plot_set)>0 or ax is not None:
                logging.warning('kws_plot,kws_plot_set and ax  will be deprecated, provide them in the plot arg. instead')

                plot={
                    **dict(
                        set=kws_plot_set,
                        ax=ax,
                        )
                    ## override
                    **{
                        **kws_plot,
                        **plot,
                    }
                    }
                logging.warning(f"plot={plot}")

            from roux.viz.sets import plot_sets
            ax=apply_plot(
                plot_sets,
                data,
                kws_plot={
                        **dict(
                            plot=dict(
                                x=x,
                                y=y,
                                kws_stats=kws_stats,
                            ),
                        ),
                        **plot
                    },
                )
            res=ax.stats
        except Exception as e:
            logging.exception(str(e))
            return check_sass(
                data,
                x=x, #,
                y=y, #,

                order_x=order_x, #=None,
                order_y=order_y, #=None,

                method=method, #=None, 
                verbose=verbose, #=True,

                validate=validate, #=None, 
                
                plot=False, 
                
                out=out, #=False,
                **kws_stats,
                )

    df1=pd.Series(res).to_frame().T.drop(['ax','table'],axis=1,errors='ignore')
    if df1 is None:
        return None
    if verbose:
        logging.info(f'{data.name if hasattr(data,"name") else ""}{x} - {y}\n'+df1.to_string(index=False))
    if validate is not None:
        assert_expr(df1,validate)
        # logging.info(df1)
    if out:
        return df1
    else:
        #pipe
        return data

@to_rd
def check_links(
    data,
    x,
    y,
    cols_id,
    
    xbins=2,
    ybins=2,    

    ## specific
    kws_stats={},
    # method={}, # mannwhitneyu
    # validate={}, 
    plot=False, ## or dict with get_ax kws
    ## common
    verbose=True,
    out=False,    
    ):
    kws_common=dict(
        out=False,
        verbose=verbose,
    )
    
    df0=(
        data
            ## pre
            .rd.assert_dense(
                subset=cols_id
            )
            .log.dropna(
                subset=[x,y],
            )    
    )
    if isinstance(xbins,int):
        df0=df0.rd.get_qbins(
            x,
            xbins,
        )
        col_xbin=f'{x} bin'
        xbins=df0[col_xbin].sort_values().astype(str).unique().tolist()
    if isinstance(ybins,int):
        df0=df0.rd.get_qbins(
            y,
            ybins,
        )
        col_ybin=f'{y} bin'
        ybins=df0[col_ybin].sort_values().astype(str).unique().tolist()
    df0=(
        df0
        .astype(
            {
                col_xbin: str,
                col_ybin: str,
            }
        )
    )    

    if plot==True and not isinstance(plot,dict):
        plot={}
    kws_checks={}
    for k in ['corr','diffx','diffy','sass']:
        kws_checks[k]={
            **kws_common,
            **kws_stats.get(k[:4],{}),
            **dict(
                plot={
                    **dict(
                        # specific default kws_get_ax
                        ax=dict(
                            ax='gca',
                            cols_max=2,
                            figsize=[8,6],
                        )
                    ),
                    **(plot if isinstance(plot,dict) else {}),
                    },
            ),
        }
    
    df1=(
        df0
            # stats
            .pipe(
                check_corr,
                x=x,
                y=y,
                **kws_checks['corr'],
            )
            .pipe(
                check_diff,
                x=x,
                y=col_ybin,
                cols_id=cols_id,
                **kws_checks['diffx'],
            )
            .pipe(
                check_diff,
                x=col_xbin,
                y=y,
                cols_id=cols_id,
                **kws_checks['diffy'],
            )
            .pipe(
                check_sass,
                x=col_xbin,
                y=col_ybin,

                order_x=xbins,
                order_y=ybins,
                **kws_checks['sass'],
            )
    )

    if out:
        return df1
    else:
        #pipe
        return data

## tables io
def dict2df(
    d,
    colkey="key",
    colvalue="value",
    recursive=True,
    depth=None,
    ):
    """Dictionary to DataFrame.

    Parameters:
        d (dict): dictionary.
        colkey (str): name of column containing the keys.
        colvalue (str): name of column containing the values.

    Returns:
        df (DataFrame): output dataframe.
    """
    if len(d)==0:
        return
        
    if not isinstance(list(d.values())[0], list):
        df_=pd.DataFrame({colkey: d.keys(), colvalue: d.values()})
    else:
        df_=(
            pd.DataFrame(pd.concat({k: pd.Series(d[k]) for k in d}))
            .droplevel(1)
            .reset_index()
            .rename(
                columns={"index": colkey, 0: colvalue},
                errors="raise",
            )
        )

    if (not isinstance(df_[colvalue].values[0],dict)) or not recursive:
        return df_
        
    cols_groupby=[colkey]
    _depth=1
    while isinstance(df_[colvalue].values[0],dict):
        _cols_groupby=df_.columns.tolist()
        _len=len(df_)
        df_=(
            df_                        
            .groupby(cols_groupby,sort=False)
                .apply(
                    # recurse
                    lambda df: dict2df(                                                             
                        df[colvalue].values[0],
                        colkey=f'key{_depth}',
                        colvalue=colvalue,
                        recursive=False,
                    ),
                )
                .reset_index(
                    # drop=True,
                )
            .rd.dropby_patterns('level_')
        )
        logging.info(df_.columns)
        cols_groupby+=list(set(df_.columns.tolist()) - set(_cols_groupby))


        logging.info(f"{_len} -> {len(df_)}")
        if len(df_)>_len:
            df_=(
                df_
                .assign(
                    **{
                        f'list{_depth} id': lambda df: df.groupby(cols_groupby,sort=False)[colkey].transform(lambda x: range(len(x)))
                    }                                                                                            
                 )
            )
            cols_groupby+=[f'list{_depth} id']
        if _depth==depth:
            break
        _depth+=1

    return df_
    
## log
def _get_log_label(label):
    if label is None:
        label=''
    if len(label)>100:
        # logging.debug('set label=None because len(label)>100 ..')
        label=label[:100]+'.. (trimmed)'          
    return label
    
def log_shape_change(d1, fun="", label=None):
    """Report the changes in the shapes of a DataFrame.

    Parameters:
        d1 (dic): dictionary containing the shapes.
        fun (str): name of the function.
    """
    label=_get_log_label(label)
    if d1["from"] != d1["to"]:
        prefix = f"{fun} {label}: " if fun != "" else ""
        if d1["from"][0] == d1["to"][0]:
            logging.info(
                f"{prefix}shape changed: {d1['from']}->{d1['to']}, length constant"
            )
        elif d1["from"][1] == d1["to"][1]:
            logging.info(
                f"{prefix}shape changed: {d1['from']}->{d1['to']}, width constant"
            )
        else:
            logging.info(f"{prefix}shape changed: {d1['from']}->{d1['to']}")


def log_apply(
    df,
    fun,
    validate_equal_length=False,
    validate_equal_width=False,
    validate_equal_shape=False,
    validate_no_decrease_length=False,
    validate_no_decrease_width=False,
    validate_no_increase_length=False,
    validate_no_increase_width=False,
    label=None,
    *args,
    **kwargs,
):
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
    d1 = {}
    d1["from"] = df.shape
    if isinstance(fun, str):
        df = getattr(df, fun)(*args, **kwargs)
    else:
        df = fun(df, *args, **kwargs)
    d1["to"] = df.shape
    log_shape_change(d1, fun=fun,label=label)
    if validate_equal_length:
        assert d1["from"][0] == d1["to"][0], (d1["from"][0], d1["to"][0])
    if validate_equal_width:
        assert d1["from"][1] == d1["to"][1], (d1["from"][1], d1["to"][1])
    if validate_no_decrease_length:
        assert d1["from"][0] <= d1["to"][0], (d1["from"][0], d1["to"][0])
    if validate_no_decrease_width:
        assert d1["from"][1] <= d1["to"][1], (d1["from"][1], d1["to"][1])
    if validate_no_increase_length:
        assert d1["from"][0] >= d1["to"][0], (d1["from"][0], d1["to"][0])
    if validate_no_increase_width:
        assert d1["from"][1] >= d1["to"][1], (d1["from"][1], d1["to"][1])
    if validate_equal_shape:
        assert d1["from"] == d1["to"], (d1["from"], d1["to"])
    return df

def _get_preview_log_str(
    df,
    lin_if_cols_gt=10,
    cols_max=50,
    ):
    warn=''
    if df.shape[1]>cols_max:
        df=df.iloc[:,:cols_max] 
        warn=' (trimmed)'  
    df=df.T if (df.shape[1]>lin_if_cols_gt) else df
    return df.to_string().lstrip()+warn

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

    def __call__(
        self,
        subset=None,
        groupby=None,
        label="",
        suffix=None,  # to be deprecated in the future
        **kws_check_nunique,
    ):
        if suffix is not None:
            logging.warning("please use label= instead of suffix= in the future.")
            label = suffix
        if subset is not None:
            suffix_ = self._obj.rd.check_nunique(
                subset=subset,
                groupby=groupby,
                out=False,
                log=False,
                **kws_check_nunique,
            )
            label = f"{suffix_} {label}"
        elif groupby is not None:
            # value_counts in pipe
            ds_ = self._obj.groupby(groupby,observed=True).size().sort_values(ascending=False)
            suffix_ = f"by '{groupby}': {to_str(ds_)}"
            label = f"{suffix_} {label}"
        label=_get_log_label(label)    
        logging.info(f"shape = {self._obj.shape} {label}")
        return self._obj

    def dropna(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="dropna", **kws)

    def drop_duplicates(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="drop_duplicates", **kws)

    def drop(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="drop", **kws)

    def filter_(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="filter", **kws)

    def pivot(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="pivot", **kws)

    def pivot_table(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="pivot_table", **kws)

    def melt(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="melt", **kws)

    def stack(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="stack", **kws)

    def unstack(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="unstack", **kws)

    def explode(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="explode", **kws)

    def merge(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="merge", **kws)

    def join(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="join", **kws)

    def groupby(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun="groupby", **kws)

    def query(self, **kws):
        from roux.lib.df import log_apply
        if kws.get('expr') is None:
            return self._obj.log(label=kws.get('expr'))
        else:
            return log_apply(self._obj, fun="query", label=kws.get('expr'), **kws)

    ## rd
    def clean(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun=clean, **kws)

    def filter_rows(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun=filter_rows, **kws)

    def melt_paired(self, **kws):
        from roux.lib.df import log_apply

        return log_apply(self._obj, fun=melt_paired, **kws)
    
    def head(
        self,
        n=1, 
        lin_if_cols_gt=3, ## transpose if >lin_if_cols_gt
        # cols_max=100, ## trim if > cols
        ):  
        logging.info(
            f'head {n}/{len(self._obj)}:'+(
                _get_preview_log_str(
                    self._obj.head(n=n,),
                    lin_if_cols_gt=lin_if_cols_gt,
                    # cols_max=cols_max,
                )
            )
        )
        return self._obj
    def tail(self,
             n=1,
             lin_if_cols_gt=3, ## transpose if >lin_if_cols_gt
             # cols_max=100, ## trim if > cols
            ):
        logging.info(
            f'tail {n}/{len(self._obj)}:\n'+(
                _get_preview_log_str(
                    self._obj.tail(n=n,),
                    lin_if_cols_gt=lin_if_cols_gt,
                    # cols_max=cols_max,
                )
            )
        )
        return self._obj
    def describe(
        self,
        subset=None,
        **kws,
        ):
        if isinstance(subset,str):
            subset=[subset]
        if subset is None:
            subset=self._obj.columns.tolist()
            
        logging.info(
            'describe:\n'+(
                self._obj
                    .loc[:,subset]
                    .describe(
                        **kws,
                    ).T
                .to_string()
            )
            )
        return self._obj     
        
    ## .rd functions for logging-only, usage within pipes
    def check_na(self, **kws):
        # logging.info(f'na {kws}')
        logging.info(check_na(self._obj, **kws))
        return self._obj

    def check_dups(self, **kws):
        # logging.info(f'dups {kws}')
        logging.info(check_dups(self._obj, **kws))
        return self._obj
