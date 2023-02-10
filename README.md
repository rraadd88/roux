# `roux` : General-purpose helper functions in Python.
[![PyPI](https://img.shields.io/pypi/v/roux?style=flat-square&colorB=blue)![PyPI](https://img.shields.io/pypi/pyversions/roux?style=flat-square&colorB=blue)](https://pypi.org/project/roux)  
[![build](https://img.shields.io/github/workflow/status/rraadd88/roux/build?style=flat-square&colorB=blue)](https://github.com/rraadd88/roux/actions/workflows/build.yml)  
# Installation
    
```
pip install roux
```
# Examples
**[Tables/Dataframes.]('https://github.com/rraadd88/roux/blob/master/examples/roux_lib_df.ipynb')**  
**[General Input/Output.]('https://github.com/rraadd88/roux/blob/master/examples/roux_lib_io.ipynb')**  
**[Strings encoding/decoding.]('https://github.com/rraadd88/roux/blob/master/examples/roux_lib_str.ipynb')**  
**[File paths Input/Output.]('https://github.com/rraadd88/roux/blob/master/examples/roux_lib_sys.ipynb')**  
**[Clustering.]('https://github.com/rraadd88/roux/blob/master/examples/roux_stat_cluster.ipynb')**  
**[Annotating visualisations.]('https://github.com/rraadd88/roux/blob/master/examples/roux_viz_annot.ipynb')**  
**[Visualizations Input/Output.]('https://github.com/rraadd88/roux/blob/master/examples/roux_viz_io.ipynb')**  
**[Line plots.]('https://github.com/rraadd88/roux/blob/master/examples/roux_viz_line.ipynb')**  
**[Scatter plots.]('https://github.com/rraadd88/roux/blob/master/examples/roux_viz_scatter.ipynb')**  

# API
<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.global_imports`
For the use in jupyter notebook for example. 



<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.df`
For processing individual pandas DataFrames/Series 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_name`

```python
get_name(df1: DataFrame, cols: list = None, coff: float = 2, out=None)
```

Gets the name of the dataframe.  

Especially useful within `groupby`+`pandarellel` context. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`cols`</b> (list):  list groupby columns.  
 - <b>`coff`</b> (int):  cutoff of unique values to infer the name. 
 - <b>`out`</b> (str):  format of the output (list|not). 



**Returns:**
 
 - <b>`name`</b> (tuple|str|list):  name of the dataframe.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_groupby_columns`

```python
get_groupby_columns(df_)
```

Get the columns supplied to `groupby`. 



**Parameters:**
 
 - <b>`df_`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`columns`</b> (list):  list of columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_constants`

```python
get_constants(df1)
```

Get the columns with a single unique value. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`columns`</b> (list):  list of columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_unnamedcol`

```python
drop_unnamedcol(df)
```

Deletes the columns with "Unnamed" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_unnamedcol`

```python
drop_unnamedcol(df)
```

Deletes the columns with "Unnamed" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_levelcol`

```python
drop_levelcol(df)
```

Deletes the potentially temporary columns names with "level" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_constants`

```python
drop_constants(df)
```

Deletes columns with a single unique value. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dropby_patterns`

```python
dropby_patterns(df1, patterns=None, strict=False, test=False)
```

Deletes columns containing substrings i.e. patterns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`patterns`</b> (list):  list of substrings. 
 - <b>`test`</b> (bool):  verbose.  



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `flatten_columns`

```python
flatten_columns(df: DataFrame, sep: str = ' ', **kws) → DataFrame
```

Multi-index columns to single-level. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`sep`</b> (str):  separator within the joined tuples (' '). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 

Keyword Arguments: 
 - <b>`kws`</b> (dict):  parameters provided to `coltuples2str` function.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `lower_columns`

```python
lower_columns(df)
```

Column names of the dataframe to lower-case letters. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `renameby_replace`

```python
renameby_replace(
    df: DataFrame,
    replaces: dict,
    ignore: bool = True,
    **kws
) → DataFrame
```

Rename columns by replacing sub-strings. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`replaces`</b> (dict|list):  from->to format or list containing substrings to remove. 
 - <b>`ignore`</b> (bool):  if True, not validate the successful replacements. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 

Keyword Arguments: 
 - <b>`kws`</b> (dict):  parameters provided to `replacemany` function.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clean_columns`

```python
clean_columns(df: DataFrame) → DataFrame
```

Standardise columns. 

Steps:  1. Strip flanking white-spaces.  2. Lower-case letters. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clean`

```python
clean(
    df: DataFrame,
    cols: list = [],
    drop_constants: bool = False,
    drop_unnamed: bool = True,
    verb: bool = False
) → DataFrame
```

Deletes potentially temporary columns. 

Steps:  1. Strip flanking white-spaces.  2. Lower-case letters. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`drop_constants`</b> (bool):  whether to delete the columns with a single unique value.  
 - <b>`drop_unnamed`</b> (bool):  whether to delete the columns with 'Unnamed' prefix.  
 - <b>`verb`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compress`

```python
compress(df1, coff_categories=20, test=False)
```

Compress the dataframe by converting columns containing strings/objects to categorical. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`coff_categories`</b> (int):  if the number of unique values are less than cutoff the it will be converted to categories.  
 - <b>`test`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clean_compress`

```python
clean_compress(df, kws_compress={}, **kws_clean)
```

`clean` and `compress` the dataframe. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`kws_compress`</b> (int):  keyword arguments for the `compress` function.  
 - <b>`test`</b> (bool):  verbose. 

Keyword Arguments: 
 - <b>`kws_clean`</b> (dict):  parameters provided to `clean` function. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 

See Also: `clean` `compress` 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L310"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_na`

```python
check_na(df, subset=None, perc=False)
```

Number/percentage of missing values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L333"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_no_na`

```python
validate_no_na(df, subset=None)
```

Validate no missing values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L348"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_no_na`

```python
assert_no_na(df, subset=None)
```

Assert that no missing values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L364"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_nunique`

```python
check_nunique(
    df: DataFrame,
    subset: list = None,
    groupby: str = None,
    perc: bool = False,
    out=False
) → Series
```

Number/percentage of unique values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_inflation`

```python
check_inflation(df1, subset=None)
```

Occurances of values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L419"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_dups`

```python
check_dups(df, subset=None, perc=False)
```

Check duplicates. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_duplicated`

```python
check_duplicated(df, subset=None, perc=False)
```

Check duplicates (alias of `check_dups`)      




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_no_dups`

```python
validate_no_dups(df, subset=None)
```

Validate that no duplicates. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_no_duplicates`

```python
validate_no_duplicates(df, subset=None)
```

Validate that no duplicates (alias of `validate_no_dups`)  




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_no_dups`

```python
assert_no_dups(df, subset=None)
```

Assert that no duplicates  




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_dense`

```python
validate_dense(
    df01: DataFrame,
    subset: list = None,
    duplicates: bool = True,
    na: bool = True,
    message=None
) → DataFrame
```

Validate no missing values and no duplicates in the dataframe. 



**Parameters:**
 
 - <b>`df01`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`duplicates`</b> (bool):  whether to check duplicates. 
 - <b>`na`</b> (bool):  whether to check na. 
 - <b>`message`</b> (str):  error message 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L500"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_dense`

```python
assert_dense(
    df01: DataFrame,
    subset: list = None,
    duplicates: bool = True,
    na: bool = True,
    message=None
) → DataFrame
```

Alias of `validate_dense`. 



**Notes:**

> to be deprecated in future releases. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `classify_mappings`

```python
classify_mappings(
    df1: DataFrame,
    col1: str,
    col2: str,
    clean: bool = False
) → DataFrame
```

Classify mappings between items in two columns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`col1`<.py#1. 
 - <b>`col2`<.py#2. 
 - <b>`clean`</b> (str):  drop columns with the counts. 



**Returns:**
 
 - <b>`(pd.DataFrame)`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_mappings`

```python
check_mappings(
    df: DataFrame,
    subset: list = None,
    out: str = 'full'
) → DataFrame
```

Mapping between items in two columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (str):  format of the output. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_1_1_mappings`

```python
validate_1_1_mappings(df: DataFrame, subset: list = None) → DataFrame
```

Validate that the papping between items in two columns is 1:1. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (str):  format of the output. 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L590"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_mappings`

```python
get_mappings(
    df1: DataFrame,
    subset=None,
    keep='1:1',
    clean=False,
    cols=None
) → DataFrame
```

Classify the mapapping between items in two columns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`keep`</b> (str):  type of mapping (1:1|1:m|m:1). 
 - <b>`clean`</b> (bool):  whether remove temporary columns.  
 - <b>`cols`</b> (list):  alias of `subset`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L654"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_filter_fast`

```python
groupby_filter_fast(
    df1: DataFrame,
    col_groupby,
    fun_agg,
    expr,
    col_agg: str = 'temporary',
    **kws_query
) → DataFrame
```

Groupby and filter fast. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`by`</b> (str|list):  column name/s to groupby with. 
 - <b>`fun`</b> (object):  function to filter with. 
 - <b>`how`</b> (str):  greater or less than `coff` (>|<).   
 - <b>`coff`</b> (float):  cut-off.     



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 



**Todo:**
 Deprecation if `pandas.core.groupby.DataFrameGroupBy.filter` is faster. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_map_binary`

```python
to_map_binary(df: DataFrame, colgroupby=None, colvalue=None) → DataFrame
```

Convert linear mappings to a binary map 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`colgroupby`</b> (str):  name of the column for groupby. 
 - <b>`colvalue`</b> (str):  name of the column containing values. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L710"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_intersections`

```python
check_intersections(
    df: DataFrame,
    colindex=None,
    colgroupby=None,
    plot=False,
    **kws_plot
) → DataFrame
```

Check intersections. Linear dataframe to is converted to a binary map and then to a series using `groupby`. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`colindex`</b> (str):  name of the index column. 
 - <b>`colgroupby`</b> (str):  name of the groupby column. 
 - <b>`plot`</b> (bool):  plot or not. 



**Returns:**
 
 - <b>`ds1`</b> (Series):  output Series. 

Keyword Arguments: 
 - <b>`kws_plot`</b> (dict):  parameters provided to the plotting function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L766"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_totals`

```python
get_totals(ds1)
```

Get totals from the output of `check_intersections`. 



**Parameters:**
 
 - <b>`ds1`</b> (Series):  input Series. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L780"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `filter_rows`

```python
filter_rows(
    df,
    d,
    sign='==',
    logic='and',
    drop_constants=False,
    test=False,
    verbose=True
)
```

Filter rows using a dictionary. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`d`</b> (dict):  dictionary. 
 - <b>`sign`</b> (str):  condition within mappings ('=='). 
 - <b>`logic`</b> (str):  condition between mappings ('and'). 
 - <b>`drop_constants`</b> (bool):  to drop the columns with single unique value (False). 
 - <b>`test`</b> (bool):  testing (False). 
 - <b>`verbose`</b> (bool):  more verbose (True).  



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L850"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_bools`

```python
get_bools(df, cols, drop=False)
```

Columns to bools. One-hot-encoder (`get_dummies`). 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe.  
 - <b>`cols`</b> (list):  columns to encode. 
 - <b>`drop`</b> (bool):  drop the `cols` (False). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L873"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `agg_bools`

```python
agg_bools(df1, cols)
```

Bools to columns. Reverse of one-hot encoder (`get_dummies`).  



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`cols`</b> (list):  columns.  



**Returns:**
 
 - <b>`ds`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L892"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `melt_paired`

```python
melt_paired(
    df: DataFrame,
    cols_index: list = None,
    suffixes: list = None,
    cols_value: list = None,
    clean: bool = False
) → DataFrame
```

Melt a paired dataframe. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`cols_index`</b> (list):  paired index columns (None). 
 - <b>`suffixes`</b> (list):  paired suffixes (None). 
 - <b>`cols_value`</b> (list):  names of the columns containing the values (None). 



**Notes:**

> Partial melt melts selected columns `cols_value`. 
>

**Examples:**
 Paired parameters:  cols_value=['value1','value2'],  suffixes=['gene1','gene2'], 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L954"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_chunks`

```python
get_chunks(
    df1: DataFrame,
    colindex: str,
    colvalue: str,
    bins: int = None,
    value: str = 'right'
) → DataFrame
```

Get chunks of a dataframe. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`colindex`</b> (str):  name of the index column. 
 - <b>`colvalue`</b> (str):  name of the column containing values [0-100] 
 - <b>`bins`</b> (int):  number of bins. 
 - <b>`value`</b> (str):  value to use as the name of the chunk ('right'). 



**Returns:**
 
 - <b>`ds`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L997"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_group`

```python
get_group(groups, i: int = None, verbose: bool = True) → DataFrame
```

Get a dataframe for a group out of the `groupby` object. 



**Parameters:**
 
 - <b>`groups`</b> (object):  groupby object. 
 - <b>`i`</b> (int):  index of the group (None). 
 - <b>`verbose`</b> (bool):  verbose (True). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Notes:**

> Useful for testing `groupby`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `infer_index`

```python
infer_index(
    data: DataFrame,
    cols_drop=[],
    include=<class 'object'>,
    exclude=None
) → list
```

Infer the index (id) of the table. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1055"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_multiindex_columns`

```python
to_multiindex_columns(df, suffixes, test=False)
```

Single level columns to multiindex. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`suffixes`</b> (list):  list of suffixes. 
 - <b>`test`</b> (bool):  verbose (False). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1077"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_ranges`

```python
to_ranges(df1, colindex, colbool, sort=True)
```

Ranges from boolean columns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`colindex`</b> (str):  column containing index items. 
 - <b>`colbool`</b> (str):  column containing boolean values. 
 - <b>`sort`</b> (bool):  sort the dataframe (True). 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 



**TODO:**
 compare with io_sets.bools2intervals. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1099"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_boolean`

```python
to_boolean(df1)
```

Boolean from ranges. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output series. 



**TODO:**
 compare with io_sets.bools2intervals. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_cat`

```python
to_cat(ds1, cats, ordered=True)
```

To series containing categories. 



**Parameters:**
 
 - <b>`ds1`</b> (Series):  input series. 
 - <b>`cats`</b> (list):  categories. 
 - <b>`ordered`</b> (bool):  if the categories are ordered (True). 



**Returns:**
 
 - <b>`ds1`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_valuesby_list`

```python
sort_valuesby_list(df1, by, cats, **kws)
```

Sort dataframe by custom order of items in a column.  



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`by`</b> (str):  column. 
 - <b>`cats`</b> (list):  ordered list of items. 

Keyword parameters: 
 - <b>`kws`</b> (dict):  parameters provided to `sort_values`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `agg_by_order`

```python
agg_by_order(x, order)
```

Get first item in the order. 



**Parameters:**
 
 - <b>`x`</b> (list):  list. 
 - <b>`order`</b> (list):  desired order of the items. 



**Returns:**
 
 - <b>`k`</b>:  first item. 



**Notes:**

> Used for sorting strings. e.g. `damaging > other non-conserving > other conserving` 
>

**TODO:**
 Convert categories to numbers and take min 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `agg_by_order_counts`

```python
agg_by_order_counts(x, order)
```

Get the aggregated counts by order*. 



**Parameters:**
 
 - <b>`x`</b> (list):  list. 
 - <b>`order`</b> (list):  desired order of the items. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Examples:**
 df=pd.DataFrame({'a1':['a','b','c','a','b','c','d'], 'b1':['a1','a1','a1','b1','b1','b1','b1'],}) df.groupby('b1').apply(lambda df : agg_by_order_counts(x=df['a1'],  order=['b','c','a'],  )) 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_sort_values`

```python
groupby_sort_values(
    df,
    col_groupby,
    col_sortby,
    subset=None,
    col_subset=None,
    func='mean',
    ascending=True
)
```

Sort groups.  



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`col_groupby`</b> (str|list):  column/s to groupby with. 
 - <b>`col_sortby`</b> (str|list):  column/s to sort values with. 
 - <b>`subset`</b> (list):  columns (None). 
 - <b>`col_subset`</b> (str):  column containing the subset (None). 
 - <b>`func`</b> (str):  aggregate function, provided to numpy ('mean'). 
 - <b>`ascending`</b> (bool):  sort values ascending (True). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_sort_values`

```python
groupby_sort_values(
    df,
    col_groupby,
    col_sortby,
    subset=None,
    col_subset=None,
    func='mean',
    ascending=True
)
```

Sort groups.  



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`col_groupby`</b> (str|list):  column/s to groupby with. 
 - <b>`col_sortby`</b> (str|list):  column/s to sort values with. 
 - <b>`subset`</b> (list):  columns (None). 
 - <b>`col_subset`</b> (str):  column containing the subset (None). 
 - <b>`func`</b> (str):  aggregate function, provided to numpy ('mean'). 
 - <b>`ascending`</b> (bool):  sort values ascending (True). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `swap_paired_cols`

```python
swap_paired_cols(df_, suffixes=['gene1', 'gene2'])
```

Swap suffixes of paired columns. 



**Parameters:**
 
 - <b>`df_`</b> (DataFrame):  input dataframe. 
 - <b>`suffixes`</b> (list):  suffixes. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_columns_by_values`

```python
sort_columns_by_values(
    df: DataFrame,
    cols_sortby=['mutation gene1', 'mutation gene2'],
    suffixes=['gene1', 'gene2'],
    clean=False
) → DataFrame
```

Sort the values in columns in ascending order. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`cols_sortby`</b> (list):  (['mutation gene1','mutation gene2']) 
 - <b>`suffixes`</b> (list):  suffixes, without no spaces. (['gene1','gene2']) 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Notes:**

> In the output dataframe, `sorted` means values are sorted because gene1>gene2. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_ids`

```python
make_ids(df, cols, ids_have_equal_length, sep='--', sort=False)
```

Make ids by joining string ids in more than one columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`cols`</b> (list):  columns.  
 - <b>`ids_have_equal_length`</b> (bool):  ids have equal length, if True faster processing. 
 - <b>`sep`</b> (str):  separator between the ids ('--'). 
 - <b>`sort`</b> (bool):  sort the ids before joining (False). 



**Returns:**
 
 - <b>`ds`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_ids_sorted`

```python
make_ids_sorted(df, cols, ids_have_equal_length, sep='--')
```

Make sorted ids by joining string ids in more than one columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`cols`</b> (list):  columns.  
 - <b>`ids_have_equal_length`</b> (bool):  ids have equal length, if True faster processing. 
 - <b>`sep`</b> (str):  separator between the ids ('--'). 



**Returns:**
 
 - <b>`ds`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_alt_id`

```python
get_alt_id(s1='A--B', s2='A', sep='--')
```

Get alternate/partner id from a paired id. 



**Parameters:**
 
 - <b>`s1`</b> (str):  joined id. 
 - <b>`s2`</b> (str):  query id.  



**Returns:**
 
 - <b>`s`</b> (str):  partner id. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `split_ids`

```python
split_ids(df1, col, sep='--', prefix=None)
```

Split joined ids to individual ones. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`col`</b> (str):  column containing the joined ids. 
 - <b>`sep`</b> (str):  separator within the joined ids ('--'). 
 - <b>`prefix`</b> (str):  prefix of the individual ids (None). 

Return: 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dict2df`

```python
dict2df(d, colkey='key', colvalue='value')
```

Dictionary to DataFrame. 



**Parameters:**
 
 - <b>`d`</b> (dict):  dictionary. 
 - <b>`colkey`</b> (str):  name of column containing the keys. 
 - <b>`colvalue`</b> (str):  name of column containing the values. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1376"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_shape_change`

```python
log_shape_change(d1, fun='')
```

Report the changes in the shapes of a DataFrame. 



**Parameters:**
 
 - <b>`d1`</b> (dic):  dictionary containing the shapes. 
 - <b>`fun`</b> (str):  name of the function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1392"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_apply`

```python
log_apply(
    df,
    fun,
    validate_equal_length=False,
    validate_equal_width=False,
    validate_equal_shape=False,
    validate_no_decrease_length=False,
    validate_no_decrease_width=False,
    validate_no_increase_length=False,
    validate_no_increase_width=False,
    *args,
    **kwargs
)
```

Report (log) the changes in the shapes of the dataframe before and after an operation/s. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`fun`</b> (object):  function to apply on the dataframe. 
 - <b>`validate_equal_length`</b> (bool):  Validate that the number of rows i.e. length of the dataframe remains the same before and after the operation.  
 - <b>`validate_equal_width`</b> (bool):  Validate that the number of columns i.e. width of the dataframe remains the same before and after the operation.  
 - <b>`validate_equal_shape`</b> (bool):  Validate that the number of rows and columns i.e. shape of the dataframe remains the same before and after the operation.  

Keyword parameters: 
 - <b>`args`</b> (tuple):  provided to `fun`. 
 - <b>`kwargs`</b> (dict):  provided to `fun`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `log`
Report (log) the changes in the shapes of the dataframe before and after an operation/s.  



**TODO:**
  Create the attribures (`attr`) using strings e.g. setattr.  import inspect  fun=inspect.currentframe().f_code.co_name 

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1510"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `check_dups`

```python
check_dups(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1506"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `check_na`

```python
check_na(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1503"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `check_nunique`

```python
check_nunique(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1493"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `clean`

```python
clean(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `drop`

```python
drop(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1453"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `drop_duplicates`

```python
drop_duplicates(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `dropna`

```python
dropna(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `explode`

```python
explode(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1462"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `filter_`

```python
filter_(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `filter_rows`

```python
filter_rows(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1489"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `groupby`

```python
groupby(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1486"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `join`

```python
join(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `melt`

```python
melt(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `melt_paired`

```python
melt_paired(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `merge`

```python
merge(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1465"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `pivot`

```python
pivot(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `pivot_table`

```python
pivot_table(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `query`

```python
query(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1474"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `stack`

```python
stack(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/df.py#L1477"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `unstack`

```python
unstack(**kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.dfs`
For processing multiple pandas DataFrames/Series 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dfs.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `filter_dfs`

```python
filter_dfs(dfs, cols, how='inner')
```

Filter dataframes based items in the common columns. 



**Parameters:**
 
 - <b>`dfs`</b> (list):  list of dataframes. 
 - <b>`cols`</b> (list):  list of columns. 
 - <b>`how`</b> (str):  how to filter ('inner') 

Returns 
 - <b>`dfs`</b> (list):  list of dataframes.         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dfs.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_with_many_columns`

```python
merge_with_many_columns(
    df1: DataFrame,
    right: str,
    left_on: str,
    right_ons: list,
    right_id: str,
    how: str = 'inner',
    validate: str = '1:1',
    test: bool = False,
    verbose: bool = False,
    **kws_merge
) → DataFrame
```

Merge with many columns. For example, if ids in the left table can map to ids located in multiple columns of the right table. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  left table. 
 - <b>`right`</b> (pd.DataFrame):  right table. 
 - <b>`left_on`</b> (str):  column in the left table to merge on. 
 - <b>`right_ons`</b> (list):  columns in the right table to merge on. 
 - <b>`right_id`</b> (str):  column in the right dataframe with for example the ids to be merged. 

Keyword parameters: 
 - <b>`kws_merge`</b>:  to be supplied to `pandas.DataFrame.merge`. 



**Returns:**
 Merged table. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dfs.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_paired`

```python
merge_paired(
    df1,
    df2,
    left_ons,
    right_on,
    common=[],
    right_ons_common=[],
    how='inner',
    validates=['1:1', '1:1'],
    suffixes=None,
    test=False,
    verb=True,
    **kws
)
```

Merge uppaired dataframes to a paired dataframe.  



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  paired dataframe.   
 - <b>`df2`</b> (DataFrame):  unpaired dataframe. 
 - <b>`left_ons`</b> (list):  columns of the `df1` (suffixed). 
 - <b>`right_on`</b> (str|list):  column/s of the `df2` (to be suffixed). 
 - <b>`common`</b> (str|list):  common column/s between `df1` and `df2` (not suffixed). 
 - <b>`right_ons_common`</b> (str|list):  common column/s between `df2` to be used for merging (not to be suffixed). 
 - <b>`how`</b> (str):  method of merging ('inner'). 
 - <b>`validates`</b> (list):  validate mappings for the 1st mapping between `df1` and `df2` and 2nd one between `df1+df2` and `df2` (['1:1','1:1']). 
 - <b>`suffixes`</b> (list):  suffixes to be used (None). 
 - <b>`test`</b> (bool):  testing (False). 
 - <b>`verb`</b> (bool):  verbose (True). 

Keyword Parameters: 
 - <b>`kws`</b> (dict):  parameters provided to `merge`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Examples:**
 

**Parameters:**
  how='inner',  left_ons=['gene id gene1','gene id gene2'], # suffixed  common='sample id', # not suffixed  right_on='gene id', # to be suffixed  right_ons_common=[], # not to be suffixed     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dfs.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dfs`

```python
merge_dfs(dfs, **kws)
```

Merge dataframes from left to right.    



**Parameters:**
 
 - <b>`dfs`</b> (list):  list of dataframes. 

Keyword Parameters: 
 - <b>`kws`</b> (dict):  parameters provided to `merge`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Notes:**

> For example, reduce(lambda x, y: x.merge(y), [1, 2, 3, 4, 5]) merges ((((1.merge(2)).merge(3)).merge(4)).merge(5)). 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dfs.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_rows`

```python
compare_rows(df1, df2, test=False, **kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.dict`
For processing dictionaries. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `head_dict`

```python
head_dict(d, lines=5)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_dict`

```python
sort_dict(d1, by=1, ascending=True)
```

Sort dictionary by values. 



**Parameters:**
 
 - <b>`d1`</b> (dict):  input dictionary. 
 - <b>`by`</b> (int):  index of the value among the values. 
 - <b>`ascending`</b> (bool):  ascending order. 



**Returns:**
 
 - <b>`d1`</b> (dict):  output dictionary. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dicts`

```python
merge_dicts(l: list) → dict
```

Merge dictionaries. 



**Parameters:**
 
 - <b>`l`</b> (list):  list containing the dictionaries. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 

TODOs: 1. In python>=3.9, `merged = d1 | d2`? 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dicts_deep`

```python
merge_dicts_deep(left: dict, right: dict) → dict
```

Merge nested dictionaries. Overwrites left with right. 



**Parameters:**
 
 - <b>`left`<.py#1 
 - <b>`right`<.py#2 

TODOs: 1. In python>=3.9, `merged = d1 | d2`? 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dict_values`

```python
merge_dict_values(l, test=False)
```

Merge dictionary values.     

**Parameters:**
 
 - <b>`l`</b> (list):  list containing the dictionaries. 
 - <b>`test`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/dict.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `flip_dict`

```python
flip_dict(d)
```

switch values with keys and vice versa. 



**Parameters:**
 
 - <b>`d`</b> (dict):  input dictionary. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.google`
Processing files form google-cloud services. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object.  

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object.  

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list_files_in_folder`

```python
list_files_in_folder(service, folderid, filetype=None, fileext=None, test=False)
```

Lists files in a google drive folder. 

:param service: service object e.g. drive :param folderid: folder id from google drive :param filetype: specify file type :param fileext: specify file extension :param test: True if verbose else False ... :return: list of files in the folder      


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_file_id`

```python
get_file_id(p)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download_file`

```python
download_file(
    p=None,
    file_id=None,
    service=None,
    outd=None,
    outp=None,
    convert=False,
    force=False,
    test=False
)
```

Downloads a specified file. 

:param service: google service object :param file_id: file id as on google drive :param filetypes: specify file type :param outp: path to the ouput file :param test: True if verbose else False 

Ref: https://developers.google.com/drive/api/v3/ref-export-formats 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `upload_file`

```python
upload_file(service, filep, folder_id, test=False)
```

Uploads a local file onto google drive. 

:param service: google service object :param filep: path of the file :param folder_id: id of the folder on google drive where the file will be uploaded  :param test: True is verbose else False ... :return: id of the uploaded file  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `upload_files`

```python
upload_files(service, ps, folder_id, **kws)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download_drawings`

```python
download_drawings(folderid, outd, service=None, test=False)
```

Download specific files: drawings 

TODOs: 1. use download_file 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_comments`

```python
get_comments(
    fileid,
    fields='comments/quotedFileContent/value,comments/content,comments/id',
    service=None
)
```

Get comments. 

 fields: comments/  kind:  id:  createdTime:  modifiedTime:  author:  kind:  displayName:  photoLink:  me:  True  htmlContent:  content:  deleted:  quotedFileContent:  mimeType:  value:  anchor:  replies:  [] 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `search`

```python
search(query, results=1, service=None, **kws_search)
```

Google search. 

:param query: exact terms ... :return: dict 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_search_strings`

```python
get_search_strings(text, num=5, test=False)
```

Google search. 

:param text: string :param num: number of results :param test: True if verbose else False ... :return lines: list 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L376"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_metadata_of_paper`

```python
get_metadata_of_paper(
    file_id,
    service_drive,
    service_search,
    metadata=None,
    force=False,
    test=False
)
```

Get the metadata of a pdf document. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `share`

```python
share(
    drive_service,
    content_id,
    share=False,
    unshare=False,
    user_permission=None,
    permissionId='anyoneWithLink'
)
```

:params user_permission:      user_permission = {  'type': 'anyone',  'role': 'reader',  'email':'@' } Ref: https://developers.google.com/drive/api/v3/manage-sharing 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `slides`







---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `create_image`

```python
create_image(service, presentation_id, page_id, image_id)
```

image less than 1.5 Mb 

---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/google.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `get_page_ids`

```python
get_page_ids(service, presentation_id)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.io`
For input/output of data files. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_zip`

```python
read_zip(p: str, file_open: str = None, fun_read=None, test: bool = False)
```

Read the contents of a zip file. 



**Parameters:**
 
 - <b>`p`</b> (str):  path of the file. 
 - <b>`file_open`</b> (str):  path of file within the zip file to open. 
 - <b>`fun_read`</b> (object):  function to read the file. 



**Examples:**
 1. Setting `fun_read` parameter for reading tab-separated table from a zip file.  

 from io import StringIO  ...  fun_read=lambda x: pd.read_csv(io.StringIO(x.decode('utf-8')),sep=' ',header=None),  

 or   

 from io import BytesIO  ...  fun_read=lambda x: pd.read_table(BytesIO(x)), 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_zip`

```python
to_zip(p: str, outp: str = None, fmt: str = 'zip')
```

Compress a file/directory. 



**Parameters:**
 
 - <b>`p`</b> (str):  path to the file/directory. 
 - <b>`outp`</b> (str):  path to the output compressed file. 
 - <b>`fmt`</b> (str):  format of the compressed file. 



**Returns:**
 
 - <b>`outp`</b> (str):  path of the compressed file. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_version`

```python
get_version(suffix: str = '') → str
```

Get the time-based version string. 



**Parameters:**
 
 - <b>`suffix`</b> (string):  suffix. 



**Returns:**
 
 - <b>`version`</b> (string):  version. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `version`

```python
version(p: str, outd: str = None, **kws: dict) → str
```

Get the version of the file/directory. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 
 - <b>`outd`</b> (str):  output directory. 

Keyword parameters: 
 - <b>`kws`</b> (dict):  provided to `get_version`. 



**Returns:**
 
 - <b>`version`</b> (string):  version.         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `backup`

```python
backup(
    p: str,
    outd: str = None,
    versioned: bool = False,
    suffix: str = '',
    zipped: bool = False,
    move_only: bool = False,
    test: bool = True,
    verbose: bool = False,
    no_test: bool = False
)
```

Backup a directory 

Steps:  0. create version dir in outd  1. move ps to version (time) dir with common parents till the level of the version dir  2. zip or not  



**Parameters:**
 
 - <b>`p`</b> (str):  input path. 
 - <b>`outd`</b> (str):  output directory path. 
 - <b>`versioned`</b> (bool):  custom version for the backup (False). 
 - <b>`suffix`</b> (str):  custom suffix for the backup (''). 
 - <b>`zipped`</b> (bool):  whether to zip the backup (False). 
 - <b>`test`</b> (bool):  testing (True). 
 - <b>`no_test`</b> (bool):  no testing. Usage in command line (False).  



**TODO:**
 1. Chain to if exists and force. 2. Option to remove dirs  find and move/zip  "find -regex .*/_.*"  "find -regex .*/test.*" 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_url`

```python
read_url(url)
```

Read text from an URL. 



**Parameters:**
 
 - <b>`url`</b> (str):  URL link. 



**Returns:**
 
 - <b>`s`</b> (string):  text content of the URL. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download`

```python
download(
    url: str,
    outd: str,
    path: str = None,
    force: bool = False,
    verbose: bool = True
) → str
```

Download a file. 



**Parameters:**
 
 - <b>`url`</b> (str):  URL.  
 - <b>`path`</b> (str):  custom output path (None) 
 - <b>`outd`</b> (str):  output directory ('data/database'). 
 - <b>`force`</b> (bool):  overwrite output (False). 
 - <b>`verbose`</b> (bool):  verbose (True). 



**Returns:**
 
 - <b>`path`</b> (str):  output path (None) 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_text`

```python
read_text(p)
```

Read a file.  To be called by other functions   



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  contents. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_list`

```python
to_list(l1, p)
```

Save list. 



**Parameters:**
 
 - <b>`l1`</b> (list):  input list. 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`p`</b> (str):  path.         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_list`

```python
read_list(p)
```

Read the lines in the file. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_list`

```python
read_list(p)
```

Read the lines in the file. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_dict`

```python
is_dict(p)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_dict`

```python
read_dict(p, fmt: str = '', apply_on_keys=None, **kws) → dict
```

Read dictionary file. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 
 - <b>`fmt`</b> (str):  format of the file. 

Keyword Arguments: 
 - <b>`kws`</b> (d):  parameters provided to reader function. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_dict`

```python
to_dict(d, p, **kws)
```

Save dictionary file. 



**Parameters:**
 
 - <b>`d`</b> (dict):  input dictionary. 
 - <b>`p`</b> (str):  path.  

Keyword Arguments: 
 - <b>`kws`</b> (d):  parameters provided to export function. 



**Returns:**
 
 - <b>`p`</b> (str):  path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `post_read_table`

```python
post_read_table(
    df1: DataFrame,
    clean: bool,
    tables: list,
    verbose: bool = True,
    **kws_clean: dict
)
```

Post-reading a table. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`clean`</b> (bool):  whether to apply `clean` function.  tables () 
 - <b>`verbose`</b> (bool):  verbose. 

Keyword parameters: 
 - <b>`kws_clean`</b> (dict):  paramters provided to the `clean` function.  



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_table`

```python
read_table(
    p: str,
    ext: str = None,
    clean: bool = True,
    filterby_time=None,
    params: dict = {},
    kws_clean: dict = {},
    kws_cloud: dict = {},
    check_paths: bool = True,
    tables: int = 1,
    test: bool = False,
    verbose: bool = True,
    **kws_read_tables: dict
)
```

 Table/s reader.  



**Parameters:**
 
     - <b>`p`</b> (str):  path of the file. It could be an input for `read_ps`, which would include strings with wildcards, list etc.  
     - <b>`ext`</b> (str):  extension of the file (default: None meaning infered from the path). 
     - <b>`clean=(default`</b>: True). filterby_time=None). 
     - <b>`check_paths`</b> (bool):  read files in the path column (default:True).  
     - <b>`test`</b> (bool):  testing (default:False). 
     - <b>`params`</b>:  parameters provided to the 'pd.read_csv' (default:{}). For example 
     - <b>`params['columns']`</b>:  columns to read. 
     - <b>`kws_clean`</b>:  parameters provided to 'rd.clean' (default:{}). 
     - <b>`kws_cloud`</b>:  parameters for reading files from google-drive (default:{}). 
     - <b>`tables`</b>:  how many tables to be read (default:1). 
     - <b>`verbose`</b>:  verbose (default:True).  

Keyword parameters: 
     - <b>`kws_read_tables`</b> (dict):  parameters provided to `read_tables` function. For example: 
     - <b>`drop_index`</b> (bool):  whether to drop the index column e.g. `path` (default: True). 
     - <b>`replaces_index`</b> (object|dict|list|str):  for example, 'basenamenoext' if path to basename. 
     - <b>`colindex`</b> (str):  the name of the column containing the paths (default: 'path') 



**Returns:**
 
     - <b>`df`</b> (DataFrame):  output dataframe.  



**Examples:**
 1. For reading specific columns only set `params=dict(columns=list)`. 

2. While reading many files, convert paths to a column with corresponding values: 

 drop_index=False,  colindex='parameter',     replaces_index=lambda x: Path(x).parent 

3. Reading a vcf file.  p='*.vcf|vcf.gz'  read_table(p,  params_read_csv=dict(  #compression='gzip',  sep='        ',comment='#',header=None,  names=replace_many(get_header(path,comment='#',lineno=-1),['#',' '],'').split('  '))  ) 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_logp`

```python
get_logp(ps: list) → str
```

Infer the path of the log file. 



**Parameters:**
 
 - <b>`ps`</b> (list):  list of paths.      



**Returns:**
 
 - <b>`p`</b> (str):  path of the output file.      


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L603"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `apply_on_paths`

```python
apply_on_paths(
    ps: list,
    func,
    replaces_outp: str = None,
    replaces_index=None,
    drop_index: bool = True,
    colindex: str = 'path',
    filter_rows: dict = None,
    fast: bool = False,
    progress_bar: bool = True,
    params: dict = {},
    dbug: bool = False,
    test1: bool = False,
    verbose: bool = True,
    kws_read_table: dict = {},
    **kws: dict
)
```

Apply a function on list of files. 



**Parameters:**
 
 - <b>`ps`</b> (str|list):  paths or string to infer paths using `read_ps`. 
 - <b>`func`</b> (function):  function to be applied on each of the paths. 
 - <b>`replaces_outp`</b> (dict|function):  infer the output path (`outp`) by replacing substrings in the input paths (`p`). 
 - <b>`filter_rows`</b> (dict):  filter the rows based on dict, using `rd.filter_rows`. 
 - <b>`fast`</b> (bool):  parallel processing (default:False).  
 - <b>`progress_bar`</b> (bool):  show progress bar(default:True). 
 - <b>`params`</b> (dict):  parameters provided to the `pd.read_csv` function. 
 - <b>`dbug`</b> (bool):  debug mode on (default:False). 
 - <b>`test1`</b> (bool):  test on one path (default:False). 
 - <b>`kws_read_table`</b> (dict):  parameters provided to the `read_table` function (default:{}). 
 - <b>`replaces_index`</b> (object|dict|list|str):  for example, 'basenamenoext' if path to basename. 
 - <b>`drop_index`</b> (bool):  whether to drop the index column e.g. `path` (default: True). 
 - <b>`colindex`</b> (str):  the name of the column containing the paths (default: 'path') 

Keyword parameters: 
 - <b>`kws`</b> (dict):  parameters provided to the function. 



**Example:**
  1. Function:   def apply_(p,outd='data/data_analysed',force=False):  outp=f"{outd}/{basenamenoext(p)}.pqt'  if exists(outp) and not force:  return  df01=read_table(p)  apply_on_paths(  ps=glob("data/data_analysed/*"),  func=apply_,  outd="data/data_analysed/",  force=True,  fast=False,  read_path=True,  )  

TODOs: Move out of io.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_tables`

```python
read_tables(
    ps: list,
    fast: bool = False,
    filterby_time=None,
    to_dict: bool = False,
    params: dict = {},
    tables: int = None,
    **kws_apply_on_paths: dict
)
```

Read multiple tables. 



**Parameters:**
 
 - <b>`ps`</b> (list):  list of paths. 
 - <b>`fast`</b> (bool):  parallel processing (default:False) 
 - <b>`filterby_time`</b> (str):  filter by time (default:None) 
 - <b>`drop_index`</b> (bool):  drop index (default:True) 
 - <b>`to_dict`</b> (bool):  output dictionary (default:False) 
 - <b>`params`</b> (dict):  parameters provided to the `pd.read_csv` function (default:{}) 
 - <b>`tables`</b>:  number of tables (default:None). 

Keyword parameters: 
 - <b>`kws_apply_on_paths`</b> (dict):  parameters provided to `apply_on_paths`. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe.  

TODOs: Parameter to report the creation dates of the newest and the oldest files. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L802"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_table`

```python
to_table(
    df: DataFrame,
    p: str,
    colgroupby: str = None,
    test: bool = False,
    **kws
)
```

Save table. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  the input dataframe.  
 - <b>`p`</b> (str):  output path. 
 - <b>`colgroupby`</b> (str|list):  columns to groupby with to save the subsets of the data as separate files. 
 - <b>`test`</b> (bool):  testing on (default:False). 

Keyword parameters: 
 - <b>`kws`</b> (dict):  parameters provided to the `to_manytables` function. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L845"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_manytables`

```python
to_manytables(
    df: DataFrame,
    p: str,
    colgroupby: str,
    fmt: str = '',
    ignore: bool = False,
    **kws_get_chunks
)
```

Save many table. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  the input dataframe.  
 - <b>`p`</b> (str):  output path. 
 - <b>`colgroupby`</b> (str|list):  columns to groupby with to save the subsets of the data as separate files. 
 - <b>`fmt`</b> (str):  if '=' column names in the folder name e.g. col1=True. 
 - <b>`ignore`</b> (bool):  ignore the warnings (default:False). 

Keyword parameters: 
 - <b>`kws_get_chunks`</b> (dict):  parameters provided to the `get_chunks` function. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output.    

TODOs: 
 - <b>`1. Change in default parameter`</b>:  `fmt='='`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L903"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_table_pqt`

```python
to_table_pqt(
    df: DataFrame,
    p: str,
    engine: str = 'fastparquet',
    compression: str = 'gzip',
    **kws_pqt: dict
) → str
```

Save a parquet file. 



**Parameters:**
 
 - <b>`df`</b> (pd.DataFrame):  table. 
 - <b>`p`</b> (str):  path. 

Keyword parameters: Parameters provided to `pd.DataFrame.to_parquet`. 



**Returns:**
 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L929"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `tsv2pqt`

```python
tsv2pqt(p: str) → str
```

Convert tab-separated file to Apache parquet.  



**Parameters:**
 
 - <b>`p`</b> (str):  path of the input. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L942"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `pqt2tsv`

```python
pqt2tsv(p: str) → str
```

Convert Apache parquet file to tab-separated.  



**Parameters:**
 
 - <b>`p`</b> (str):  path of the input. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L956"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_excel`

```python
read_excel(
    p: str,
    sheet_name: str = None,
    kws_cloud: dict = {},
    test: bool = False,
    **kws
)
```

Read excel file 



**Parameters:**
 
 - <b>`p`</b> (str):  path of the file.  
 - <b>`sheet_name`</b> (str|None):  read 1st sheet if None (default:None) 
 - <b>`kws_cloud`</b> (dict):  parameters provided to read the file from the google drive (default:{}) 
 - <b>`test`</b> (bool):  if False and sheet_name not provided, return all sheets as a dictionary, else if True, print list of sheets. 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to the excel reader. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L996"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_excel_commented`

```python
to_excel_commented(p: str, comments: dict, outp: str = None, author: str = None)
```

Add comments to the columns of excel file and save. 



**Args:**
 
 - <b>`p`</b> (str):  input path of excel file. 
 - <b>`comments`</b> (dict):  map between column names and comment e.g. description of the column. 
 - <b>`outp`</b> (str):  output path of excel file. Defaults to None. 
 - <b>`author`</b> (str):  author of the comments. Defaults to 'Author'. 

TODOs: 1. Increase the limit on comments can be added to number of columns. Currently it is 26 i.e. upto Z1. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L1035"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_excel`

```python
to_excel(
    sheetname2df: dict,
    outp: str,
    comments: dict = None,
    save_input: bool = False,
    author: str = None,
    append: bool = False,
    **kws
)
```

Save excel file. 



**Parameters:**
 
 - <b>`sheetname2df`</b> (dict):  dictionary mapping the sheetname to the dataframe. 
 - <b>`outp`</b> (str):  output path.  
 - <b>`append`</b> (bool):  append the dataframes (default:False). 
 - <b>`comments`</b> (dict):  map between column names and comment e.g. description of the column. 
 - <b>`save_input`</b> (bool):  additionally save the input tables in text format. 

Keyword parameters:  
 - <b>`kws`</b>:  parameters provided to the excel writer. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/io.py#L1094"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_chunks`

```python
check_chunks(outd, col, plot=True)
```

Create chunks of the tables. 



**Parameters:**
 
 - <b>`outd`</b> (str):  output directory. 
 - <b>`col`</b> (str):  the column with values that are used for getting the chunks. 
 - <b>`plot`</b> (bool):  plot the chunk sizes (default:True). 



**Returns:**
 
 - <b>`df3`</b> (DataFrame):  output dataframe. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib`




**Global Variables**
---------------
- **df**
- **set**
- **str**
- **dict**
- **dfs**
- **sys**
- **text**
- **io**

---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L3"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_class`

```python
to_class(cls)
```

Get the decorator to attach functions.  



**Parameters:**
 
 - <b>`cls`</b> (class):  class object. 



**Returns:**
 
 - <b>`decorator`</b> (decorator):  decorator object. 

References: 
 - <b>`https`</b>: //gist.github.com/mgarod/09aa9c3d8a52a980bd4d738e52e5b97a 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decorator`

```python
decorator(func)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `rd`
`roux-dataframe` (`.rd`) extension.  



<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```









<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.seq`
For processing biological sequence data. 

**Global Variables**
---------------
- **bed_colns**

---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `reverse_complement`

```python
reverse_complement(s)
```

Reverse complement. 



**Args:**
 
 - <b>`s`</b> (str):  sequence 



**Returns:**
 
 - <b>`s`</b> (str):  reverse complemented sequence 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fa2df`

```python
fa2df(alignedfastap: str, ids2cols=False) → DataFrame
```

_summary_ 



**Args:**
 
 - <b>`alignedfastap`</b> (str):  path. 
 - <b>`ids2cols`</b> (bool, optional):  ids of the sequences to columns. Defaults to False. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_genomeocoords`

```python
to_genomeocoords(genomecoord: str) → tuple
```

String-formated genome co-ordinates to separated values. 



**Args:**
  genomecoord (str): 



**Raises:**
 
 - <b>`ValueError`</b>:  format of the genome co-ordinates. 



**Returns:**
 
 - <b>`tuple`</b>:  separated values i.e. chrom,start,end,strand 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_bed`

```python
to_bed(df, col_genomeocoord)
```

Genome co-ordinates to bed. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`col_genomeocoord`</b> (str):  column with the genome coordinates. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_fasta`

```python
read_fasta(fap: str, key_type: str = 'id', duplicates: bool = False) → dict
```

Read fasta 



**Args:**
 
 - <b>`fap`</b> (str):  path 
 - <b>`key_type`</b> (str, optional):  key type. Defaults to 'id'. 
 - <b>`duplicates`</b> (bool, optional):  duplicates present. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  data. 



**Notes:**

> 1. If `duplicates` key_type is set to `description` instead of `id`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/seq.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_fasta`

```python
to_fasta(
    sequences: dict,
    output_path: str,
    molecule_type: str,
    force: bool = True,
    **kws_SeqRecord
) → str
```

Save fasta file. 



**Args:**
 
 - <b>`sequences`</b> (dict):  dictionary mapping the sequence name to the sequence. 
 - <b>`output_path`</b> (str):  path of the fasta file. 
 - <b>`force`</b> (bool):  overwrite if file exists. 



**Returns:**
 
 - <b>`output_path`</b> (str):  path of the fasta file 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.set`
For processing list-like sets. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `union`

```python
union(l)
```

Union of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `union`

```python
union(l)
```

Union of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `intersection`

```python
intersection(l)
```

Intersections of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `intersection`

```python
intersection(l)
```

Intersections of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `nunion`

```python
nunion(l)
```

Count the items in union. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`i`</b> (int):  count.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `nintersection`

```python
nintersection(l)
```

Count the items in intersetion. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`i`</b> (int):  count.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dropna`

```python
dropna(x)
```

Drop `np.nan` items from a list. 



**Parameters:**
 
 - <b>`x`</b> (list):  list. 



**Returns:**
 
 - <b>`x`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `unique`

```python
unique(l)
```

Unique items in a list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 



**Notes:**

> The function can return list of lists if used in `pandas.core.groupby.DataFrameGroupBy.agg` context. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list2str`

```python
list2str(x, ignore=False)
```

Returns string if single item in a list. 



**Parameters:**
 
 - <b>`x`</b> (list):  list 



**Returns:**
 
 - <b>`s`</b> (str):  string.         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `unique_str`

```python
unique_str(l, **kws)
```

Unique single item from a list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `nunique`

```python
nunique(l, **kws)
```

Count unique items in a list 



**Parameters:**
 
 - <b>`l`</b> (list):  list 



**Returns:**
 
 - <b>`i`</b> (int):  count. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `flatten`

```python
flatten(l)
```

List of lists to list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  output list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_alt`

```python
get_alt(l1, s)
```

Get alternate item between two.  



**Parameters:**
 
 - <b>`l1`</b> (list):  list. 
 - <b>`s`</b> (str):  item. 



**Returns:**
 
 - <b>`s`</b> (str):  alternate item. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `jaccard_index`

```python
jaccard_index(l1, l2)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `intersections`

```python
intersections(dn2list, jaccard=False, count=True, fast=False, test=False)
```

Get intersections between lists. 



**Parameters:**
 
 - <b>`dn2list`</b> (dist):  dictionary mapping to lists. 
 - <b>`jaccard`</b> (bool):  return jaccard indices. 
 - <b>`count`</b> (bool):  return counts. 
 - <b>`fast`</b> (bool):  fast. 
 - <b>`test`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe.  

TODOs:  1. feed as an estimator to `df.corr()`. 2. faster processing by filling up the symetric half of the adjacency matrix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `range_overlap`

```python
range_overlap(l1, l2)
```

Overlap between ranges. 



**Parameters:**
 
 - <b>`l1`</b> (list):  start and end integers of one range. 
 - <b>`l2`</b> (list):  start and end integers of other range. 



**Returns:**
 
 - <b>`l`</b> (list):  overlapped range.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_windows`

```python
get_windows(
    a,
    size=None,
    overlap=None,
    windows=None,
    overlap_fraction=None,
    stretch_last=False,
    out_ranges=True
)
```

Windows/segments from a range.  



**Parameters:**
 
 - <b>`a`</b> (list):  range. 
 - <b>`size`</b> (int):  size of the windows.  
 - <b>`windows`</b> (int):  number of windows. 
 - <b>`overlap_fraction`</b> (float):  overlap fraction.  
 - <b>`overlap`</b> (int):  overlap length.  
 - <b>`stretch_last`</b> (bool):  stretch last window.  
 - <b>`out_ranges`</b> (bool):  whether to output ranges. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 



**Notes:**

> 1. For development, use of `int` provides `np.floor`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `bools2intervals`

```python
bools2intervals(v)
```

Convert bools to intervals. 



**Parameters:**
 
 - <b>`v`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`l`</b> (list):  intervals. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list2ranges`

```python
list2ranges(l)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/set.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_pairs`

```python
get_pairs(
    items: list,
    items_with: list = None,
    size: int = 2,
    with_self: bool = False
) → DataFrame
```

Creates a dataframe with the paired items. 



**Parameters:**
 
 - <b>`items`</b>:  the list of items to pair. 
 - <b>`items_with`</b>:  list of items to pair with. 
 - <b>`size`</b>:  size of the combinations. 
 - <b>`with_self`</b>:  pair with self or not. 



**Returns:**
 table with pairs of items. 



**Notes:**

> 1. the ids of the items are sorted e.g. 'a'-'b' not 'b'-'a'. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.str`
For processing strings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `substitution`

```python
substitution(s, i, replaceby)
```

Substitute character in a string. 



**Parameters:**
 
 - <b>`s`</b> (string):  string. 
 - <b>`i`</b> (int):  location. 
 - <b>`replaceby`</b> (string):  character to substitute with. 



**Returns:**
 
 - <b>`s`</b> (string):  output string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `substitution`

```python
substitution(s, i, replaceby)
```

Substitute character in a string. 



**Parameters:**
 
 - <b>`s`</b> (string):  string. 
 - <b>`i`</b> (int):  location. 
 - <b>`replaceby`</b> (string):  character to substitute with. 



**Returns:**
 
 - <b>`s`</b> (string):  output string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `replace_many`

```python
replace_many(
    s: str,
    replaces: dict,
    replacewith: str = '',
    ignore: bool = False
)
```

Rename by replacing sub-strings. 



**Parameters:**
 
 - <b>`s`</b> (str):  input string. 
 - <b>`replaces`</b> (dict|list):  from->to format or list containing substrings to remove. 
 - <b>`replacewith`</b> (str):  replace to in case `replaces` is a list. 
 - <b>`ignore`</b> (bool):  if True, not validate the successful replacements. 



**Returns:**
 
 - <b>`s`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `replace_many`

```python
replace_many(
    s: str,
    replaces: dict,
    replacewith: str = '',
    ignore: bool = False
)
```

Rename by replacing sub-strings. 



**Parameters:**
 
 - <b>`s`</b> (str):  input string. 
 - <b>`replaces`</b> (dict|list):  from->to format or list containing substrings to remove. 
 - <b>`replacewith`</b> (str):  replace to in case `replaces` is a list. 
 - <b>`ignore`</b> (bool):  if True, not validate the successful replacements. 



**Returns:**
 
 - <b>`s`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `tuple2str`

```python
tuple2str(tup, sep=' ')
```

Join tuple items. 



**Parameters:**
 
 - <b>`tup`</b> (tuple|list):  input tuple/list. 
 - <b>`sep`</b> (str):  separator between the items. 



**Returns:**
 
 - <b>`s`</b> (str):  output string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `linebreaker`

```python
linebreaker(text, width=None, break_pt=None, sep='\n', **kws)
```

Insert `newline`s within a string.  



**Parameters:**
 
 - <b>`text`</b> (str):  string. 
 - <b>`width`</b> (int):  insert `newline` at this interval.  
 - <b>`sep`</b> (string):  separator to split the sub-strings. 



**Returns:**
 
 - <b>`s`</b> (string):  output string. 

References: 
 - <b>`1. `textwrap``</b>:  https://docs.python.org/3/library/textwrap.html 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `findall`

```python
findall(s, ss, outends=False, outstrs=False, suffixlen=0)
```

Find the substrings or their locations in a string. 



**Parameters:**
 
 - <b>`s`</b> (string):  input string. 
 - <b>`ss`</b> (string):  substring. 
 - <b>`outends`</b> (bool):  output end positions. 
 - <b>`outstrs`</b> (bool):  output strings. 
 - <b>`suffixlen`</b> (int):  length of the suffix. 



**Returns:**
 
 - <b>`l`</b> (list):  output list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_marked_substrings`

```python
get_marked_substrings(
    s,
    leftmarker='{',
    rightmarker='}',
    leftoff=0,
    rightoff=0
) → list
```

Get the substrings flanked with markers from a string. 



**Parameters:**
 
 - <b>`s`</b> (str):  input string. 
 - <b>`leftmarker`</b> (str):  marker on the left. 
 - <b>`rightmarker`</b> (str):  marker on the right.  
 - <b>`leftoff`</b> (int):  offset on the left. 
 - <b>`rightoff`</b> (int):  offset on the right. 



**Returns:**
 
 - <b>`l`</b> (list):  list of substrings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_marked_substrings`

```python
get_marked_substrings(
    s,
    leftmarker='{',
    rightmarker='}',
    leftoff=0,
    rightoff=0
) → list
```

Get the substrings flanked with markers from a string. 



**Parameters:**
 
 - <b>`s`</b> (str):  input string. 
 - <b>`leftmarker`</b> (str):  marker on the left. 
 - <b>`rightmarker`</b> (str):  marker on the right.  
 - <b>`leftoff`</b> (int):  offset on the left. 
 - <b>`rightoff`</b> (int):  offset on the right. 



**Returns:**
 
 - <b>`l`</b> (list):  list of substrings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `mark_substrings`

```python
mark_substrings(s, ss, leftmarker='(', rightmarker=')') → str
```

Mark sub-string/s in a string. 



**Parameters:**
 
 - <b>`s`</b> (str):  input string. 
 - <b>`ss`</b> (str):  substring. 
 - <b>`leftmarker`</b> (str):  marker on the left. 
 - <b>`rightmarker`</b> (str):  marker on the right.  



**Returns:**
 
 - <b>`s`</b> (str):  string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_bracket`

```python
get_bracket(s, leftmarker='(', righttmarker=')') → str
```

Get bracketed substrings. 



**Parameters:**
 
 - <b>`s`</b> (string):  string. 
 - <b>`leftmarker`</b> (str):  marker on the left. 
 - <b>`rightmarker`</b> (str):  marker on the right.  



**Returns:**
 
 - <b>`s`</b> (str):  string. 

TODOs: 1. Use `get_marked_substrings`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `align`

```python
align(
    s1: str,
    s2: str,
    prefix: bool = False,
    suffix: bool = False,
    common: bool = True
) → list
```

Align strings.  



**Parameters:**
 
 - <b>`s1`<.py#1. 
 - <b>`s2`<.py#2. 
 - <b>`prefix`</b> (str):  prefix. 
 - <b>`suffix`</b> (str):  suffix. 
 - <b>`common`</b> (str):  common substring. 



**Returns:**
 
 - <b>`l`</b> (list):  output list. 



**Notes:**

> 1. Code to test: [ get_prefix(source,target,common=False), get_prefix(source,target,common=True), get_suffix(source,target,common=False), get_suffix(source,target,common=True),] 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_prefix`

```python
get_prefix(s1: str, s2: str, common: bool = True, clean: bool = True) → str
```

Get the prefix of the strings 



**Parameters:**
 
 - <b>`s1`</b> (str):  1st string. 
 - <b>`s2`</b> (str):  2nd string. 
 - <b>`common`</b> (bool):  get the common prefix (default:True). 
 - <b>`clean`</b> (bool):  clean the leading and trailing whitespaces (default:True). 



**Returns:**
 
 - <b>`s`</b> (str):  prefix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_suffix`

```python
get_suffix(s1: str, s2: str, common: bool = True, clean: bool = True) → str
```

Get the suffix of the strings 



**Parameters:**
 
 - <b>`s1`</b> (str):  1st string. 
 - <b>`s2`</b> (str):  2nd string. 
 - <b>`common`</b> (bool):  get the common prefix (default:True). 
 - <b>`clean`</b> (bool):  clean the leading and trailing whitespaces (default:True). 



**Returns:**
 
 - <b>`s`</b> (str):  suffix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_fix`

```python
get_fix(s1: str, s2: str, **kws: dict) → str
```

Infer common prefix or suffix. 



**Parameters:**
 
 - <b>`s1`</b> (str):  1st string. 
 - <b>`s2`</b> (str):  2nd string. 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to the `get_prefix` and `get_suffix` functions. 



**Returns:**
 
 - <b>`s`</b> (str):  prefix or suffix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `removesuffix`

```python
removesuffix(s1: str, suffix: str) → str
```

Remove suffix. 

Paramters:  s1 (str): input string.  suffix (str): suffix.  



**Returns:**
 
 - <b>`s1`</b> (str):  string without the suffix. 

TODOs:  1. Deprecate in py>39 use .removesuffix() instead. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `str2dict`

```python
str2dict(
    s: str,
    reversible: bool = True,
    sep: str = ';',
    sep_equal: str = '='
) → dict
```

String to dictionary. 



**Parameters:**
 
 - <b>`s`</b> (str):  string. 
 - <b>`sep`</b> (str):  separator between entries (default:';'). 
 - <b>`sep_equal`</b> (str):  separator between the keys and the values (default:'='). 



**Returns:**
 
 - <b>`d`</b> (dict):  dictionary. 

References: 
 - <b>`1. https`</b>: //stackoverflow.com/a/186873/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dict2str`

```python
dict2str(
    d1: dict,
    reversible: bool = True,
    sep: str = ';',
    sep_equal: str = '='
) → str
```

Dictionary to string. 



**Parameters:**
 
 - <b>`d`</b> (dict):  dictionary. 
 - <b>`sep`</b> (str):  separator between entries (default:';'). 
 - <b>`sep_equal`</b> (str):  separator between the keys and the values (default:'='). 
 - <b>`reversible`</b> (str):  use json 

**Returns:**
 
 - <b>`s`</b> (str):  string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L421"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `str2num`

```python
str2num(s: str) → float
```

String to number. 



**Parameters:**
 
 - <b>`s`</b> (str):  string. 



**Returns:**
 
 - <b>`i`</b> (int):  number. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `num2str`

```python
num2str(
    num: float,
    magnitude: bool = False,
    coff: float = 10000,
    decimals: int = 0
) → str
```

Number to string. 



**Parameters:**
 
 - <b>`num`</b> (int):  number. 
 - <b>`magnitude`</b> (bool):  use magnitudes (default:False). 
 - <b>`coff`</b> (int):  cutoff (default:10000). 
 - <b>`decimals`</b> (int):  decimal points (default:0). 



**Returns:**
 
 - <b>`s`</b> (str):  string. 

TODOs 1. ~ if magnitude else not 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L477"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `encode`

```python
encode(data, short: bool = False, method_short: str = 'sha256', **kws) → str
```

Encode the data as a string. 



**Parameters:**
 
 - <b>`data`</b> (str|dict|Series):  input data.  
 - <b>`short`</b> (bool):  Outputs short string, compatible with paths but non-reversible. Defaults to False. 
 - <b>`method_short`</b> (str):  method used for encoding when short=True. 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to encoding function. 



**Returns:**
 
 - <b>`s`</b> (string):  output string.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/str.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decode`

```python
decode(s, out=None, **kws)
```

Decode data from a string. 



**Parameters:**
 
 - <b>`s`</b> (string):  encoded string.  
 - <b>`out`</b> (str):  output format (dict|df). 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to `dict2df`. 



**Returns:**
 
 - <b>`d`</b> (dict|DataFrame):  output data.  


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.sys`
For processing file paths for example. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `basenamenoext`

```python
basenamenoext(p)
```

Basename without the extension. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `remove_exts`

```python
remove_exts(p: str, exts: tuple = None)
```

Filename without the extension. 



**Args:**
 
 - <b>`p`</b> (str):  path. 
 - <b>`exts`</b> (tuple):  extensions. 



**Returns:**
 
 - <b>`s`</b> (str):  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_ps`

```python
read_ps(ps, test=True) → list
```

Read a list of paths. 



**Parameters:**
 
 - <b>`ps`</b> (list|str):  list of paths or a string with wildcard/s. 
 - <b>`test`</b> (bool):  testing. 



**Returns:**
 
 - <b>`ps`</b> (list):  list of paths. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_path`

```python
to_path(s, replacewith='_', verbose=False, coff_len_escape_replacement=100)
```

Normalise a string to be used as a path of file. 



**Parameters:**
 
 - <b>`s`</b> (string):  input string. 
 - <b>`replacewith`</b> (str):  replace the whitespaces or incompatible characters with. 



**Returns:**
 
 - <b>`s`</b> (string):  output string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_path`

```python
to_path(s, replacewith='_', verbose=False, coff_len_escape_replacement=100)
```

Normalise a string to be used as a path of file. 



**Parameters:**
 
 - <b>`s`</b> (string):  input string. 
 - <b>`replacewith`</b> (str):  replace the whitespaces or incompatible characters with. 



**Returns:**
 
 - <b>`s`</b> (string):  output string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `makedirs`

```python
makedirs(p: str, exist_ok=True, **kws)
```

Make directories recursively. 



**Args:**
 
 - <b>`p`</b> (str):  path. 
 - <b>`exist_ok`</b> (bool, optional):  no error if the directory exists. Defaults to True. 



**Returns:**
 
 - <b>`p_`</b> (str):  the path of the directory. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_output_path`

```python
to_output_path(ps, outd=None, outp=None, suffix='')
```

Infer a single output path for a list of paths. 



**Parameters:**
 
 - <b>`ps`</b> (list):  list of paths. 
 - <b>`outd`</b> (str):  path of the output directory. 
 - <b>`outp`</b> (str):  path of the output file. 
 - <b>`suffix`</b> (str):  suffix of the filename. 



**Returns:**
 
 - <b>`outp`</b> (str):  path of the output file.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_output_paths`

```python
to_output_paths(
    input_paths: list = None,
    inputs: list = None,
    output_path: str = None,
    encode_short: bool = True,
    replaces_output_path=None,
    key_output_path: str = None,
    force: bool = False,
    verbose: bool = False
) → dict
```

Infer a output path for each of the paths or inputs. 



**Parameters:**
 
 - <b>`input_paths (list) `</b>:  list of input paths. Defaults to None. 
 - <b>`inputs (list) `</b>:  list of inputs e.g. dictionaries. Defaults to None. 
 - <b>`output_path (str) `</b>:  output path with a placeholder '{KEY}' to be replaced. Defaults to None. 
 - <b>`encode_short`</b>:  (bool) : short encoded string, else long encoded string (reversible) is used. Defaults to True. 
 - <b>`replaces_output_path `</b>:  list, dictionary or function to replace the input paths. Defaults to None. 
 - <b>`key_output_path (str) `</b>:  key to be used to incorporate output_path variable among the inputs. Defaults to None. 
 - <b>`force`</b> (bool):  overwrite the outputs. Defaults to False. 
 - <b>`verbose (bool) `</b>:  show verbose. Defaults to False. 



**Returns:**
 dictionary with the output path mapped to input paths or inputs. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_encoding`

```python
get_encoding(p)
```

Get encoding of a file. 



**Parameters:**
 
 - <b>`p`</b> (str):  file path 



**Returns:**
 
 - <b>`s`</b> (string):  encoding. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L235"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_all_subpaths`

```python
get_all_subpaths(d='.', include_directories=False)
```

Get all the subpaths. 



**Args:**
 
 - <b>`d`</b> (str, optional):  _description_. Defaults to '.'. 
 - <b>`include_directories`</b> (bool, optional):  to include the directories. Defaults to False. 



**Returns:**
 
 - <b>`paths`</b> (list):  sub-paths. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L260"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_env`

```python
get_env(env_name: str, return_path: bool = False)
```

Get the virtual environment as a dictionary. 



**Args:**
 
 - <b>`env_name`</b> (str):  name of the environment. 



**Returns:**
 
 - <b>`d`</b> (dict):  parameters of the virtual environment. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `runbash`

```python
runbash(s1, env=None, test=False, **kws)
```

Run a bash command.  



**Args:**
 
 - <b>`s1`</b> (str):  command. 
 - <b>`env`</b> (str):  environment name. 
 - <b>`test`</b> (bool, optional):  testing. Defaults to False. 



**Returns:**
 
 - <b>`output`</b>:  output of the `subprocess.call` function. 

TODOs: 1. logp 2. error ignoring 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `runbash_tmp`

```python
runbash_tmp(
    s1: str,
    env: str,
    df1=None,
    inp='INPUT',
    input_type='df',
    output_type='path',
    tmp_infn='in.txt',
    tmp_outfn='out.txt',
    outp=None,
    force=False,
    test=False,
    **kws
)
```

Run a bash command in `/tmp` directory. 



**Args:**
 
 - <b>`s1`</b> (str):  command. 
 - <b>`env`</b> (str):  environment name. 
 - <b>`df1`</b> (DataFrame, optional):  input dataframe. Defaults to None. 
 - <b>`inp`</b> (str, optional):  input path. Defaults to 'INPUT'. 
 - <b>`input_type`</b> (str, optional):  input type. Defaults to 'df'. 
 - <b>`output_type`</b> (str, optional):  output type. Defaults to 'path'. 
 - <b>`tmp_infn`</b> (str, optional):  temporary input file. Defaults to 'in.txt'. 
 - <b>`tmp_outfn`</b> (str, optional):  temporary output file.. Defaults to 'out.txt'. 
 - <b>`outp`</b> (_type_, optional):  output path. Defaults to None. 
 - <b>`force`</b> (bool, optional):  force. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`output`</b>:  output of the `subprocess.call` function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_symlink`

```python
create_symlink(p: str, outp: str, test=False)
```

Create symbolic links. 



**Args:**
 
 - <b>`p`</b> (str):  input path. 
 - <b>`outp`</b> (str):  output path. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`outp`</b> (str):  output path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L404"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `input_binary`

```python
input_binary(q: str)
```

Get input in binary format. 



**Args:**
 
 - <b>`q`</b> (str):  question. 



**Returns:**
 
 - <b>`b`</b> (bool):  response. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_interactive`

```python
is_interactive()
```

Check if the UI is interactive e.g. jupyter or command line.   




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_interactive_notebook`

```python
is_interactive_notebook()
```

Check if the UI is interactive e.g. jupyter or command line.      



**Notes:**

> 
>Reference: 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_excecution_location`

```python
get_excecution_location(depth=1)
```

Get the location of the function being executed. 



**Args:**
 
 - <b>`depth`</b> (int, optional):  Depth of the location. Defaults to 1. 



**Returns:**
 
 - <b>`tuple`</b> (tuple):  filename and line number. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_datetime`

```python
get_datetime(outstr: bool = True, fmt='%G%m%dT%H%M%S')
```

Get the date and time. 



**Args:**
 
 - <b>`outstr`</b> (bool, optional):  string output. Defaults to True. 
 - <b>`fmt`</b> (str):  format of the string. 



**Returns:**
 
 - <b>`s `</b>:  date and time. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L474"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `p2time`

```python
p2time(filename: str, time_type='m')
```

Get the creation/modification dates of files. 



**Args:**
 
 - <b>`filename`</b> (str):  filename. 
 - <b>`time_type`</b> (str, optional):  _description_. Defaults to 'm'. 



**Returns:**
 
 - <b>`time`</b> (str):  time. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L492"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `ps2time`

```python
ps2time(ps: list, **kws_p2time)
```

Get the times for a list of files.  



**Args:**
 
 - <b>`ps`</b> (list):  list of paths. 



**Returns:**
 
 - <b>`ds`</b> (Series):  paths mapped to corresponding times. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/sys.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_logger`

```python
get_logger(program='program', argv=None, level=None, dp=None)
```

Get the logging object. 



**Args:**
 
 - <b>`program`</b> (str, optional):  name of the program. Defaults to 'program'. 
 - <b>`argv`</b> (_type_, optional):  arguments. Defaults to None. 
 - <b>`level`</b> (_type_, optional):  level of logging. Defaults to None. 
 - <b>`dp`</b> (_type_, optional):  _description_. Defaults to None. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.text`
For processing text files. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/text.py#L2"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_header`

```python
get_header(path: str, comment='#', lineno=None)
```

Get the header of a file. 



**Args:**
 
 - <b>`path`</b> (str):  path. 
 - <b>`comment`</b> (str):  comment identifier. 
 - <b>`lineno`</b> (int):  line numbers upto. 



**Returns:**
 
 - <b>`lines`</b> (list):  header. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/lib/text.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `cat`

```python
cat(ps, outp)
```

Concatenate text files. 



**Args:**
 
 - <b>`ps`</b> (list):  list of paths. 
 - <b>`outp`</b> (str):  output path. 



**Returns:**
 
 - <b>`outp`</b> (str):  output path. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.run`
For access to a few functions from the terminal. 



<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.binary`
For processing binary data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_bools_jaccard`

```python
compare_bools_jaccard(x, y)
```

Compare bools in terms of the jaccard index. 



**Args:**
 
 - <b>`x`</b> (list):  list of bools. 
 - <b>`y`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`float`</b>:  jaccard index. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_bools_jaccard_df`

```python
compare_bools_jaccard_df(df: DataFrame) → DataFrame
```

Pairwise compare bools in terms of the jaccard index. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  dataframe with boolean columns. 



**Returns:**
 
 - <b>`DataFrame`</b>:  matrix with comparisons between the columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `classify_bools`

```python
classify_bools(l: list) → str
```

Classify bools. 



**Args:**
 
 - <b>`l`</b> (list):  list of bools 



**Returns:**
 
 - <b>`str`</b>:  classification. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `frac`

```python
frac(x: list) → float
```

Fraction. 



**Args:**
 
 - <b>`x`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`float`</b>:  fraction of True values. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `perc`

```python
perc(x: list) → float
```

Percentage. 



**Args:**
 
 - <b>`x`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`float`</b>:  Percentage of the True values 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats_confusion_matrix`

```python
get_stats_confusion_matrix(df_: DataFrame) → DataFrame
```

Get stats confusion matrix. 



**Args:**
 
 - <b>`df_`</b> (DataFrame):  Confusion matrix. 



**Returns:**
 
 - <b>`DataFrame`</b>:  stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/binary.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cutoff`

```python
get_cutoff(
    y_true,
    y_score,
    method,
    show_diagonal=True,
    show_area=True,
    kws_area: dict = {},
    show_cutoff=True,
    plot_pr=True,
    color='k',
    returns=['ax'],
    ax=None
)
```

Obtain threshold based on ROC or PR curve. 



**Returns:**
  Table: 
 - <b>`columns`</b>:  values 
 - <b>`method`</b>:  ROC, PR 
 - <b>`variable`</b>:  threshold (index), TPR, FPR, TP counts, precision, recall values:  Plots: AUC ROC, TPR vs TP counts PR Specificity vs TP counts Dictionary: Thresholds from AUC, PR 

TODOs:  1. Separate the plotting functions. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.classify`
For classification. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_low_complexity`

```python
drop_low_complexity(
    df1: DataFrame,
    min_nunique: int,
    max_inflation: int,
    max_nunique: int = None,
    cols: list = None,
    cols_keep: list = [],
    test: bool = False,
    verbose: bool = False
) → DataFrame
```

Remove low-complexity columns from the data.  



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`min_nunique`</b> (int):  minimum unique values. 
 - <b>`max_inflation`</b> (int):  maximum over-representation of the values. 
 - <b>`cols`</b> (list, optional):  columns. Defaults to None. 
 - <b>`cols_keep`</b> (list, optional):  columns to keep. Defaults to []. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_Xy_for_classification`

```python
get_Xy_for_classification(
    df1: DataFrame,
    coly: str,
    qcut: float = None,
    drop_xs_low_complexity: bool = False,
    min_nunique: int = 5,
    max_inflation: float = 0.5,
    **kws
) → dict
```

Get X matrix and y vector.  



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data, should be indexed. 
 - <b>`coly`</b> (str):  column with y values, bool if qcut is None else float/int 
 - <b>`qcut`</b> (float, optional):  quantile cut-off. Defaults to None. 
 - <b>`drop_xs_low_complexity`</b> (bool, optional):  to drop columns with <5 unique values. Defaults to False. 
 - <b>`min_nunique`</b> (int, optional):  minimum unique values in the column. Defaults to 5. 
 - <b>`max_inflation`</b> (float, optional):  maximum inflation. Defaults to 0.5. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `drop_low_complexity`. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cvsplits`

```python
get_cvsplits(
    X: <built-in function array>,
    y: <built-in function array>,
    cv: int = 5,
    random_state: int = None,
    outtest: bool = True
) → dict
```

Get cross-validation splits. 



**Args:**
 
 - <b>`X`</b> (np.array):  X matrix. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`cv`</b> (int, optional):  cross validations. Defaults to 5. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to None. 
 - <b>`outtest`</b> (bool, optional):  output testing. Defaults to True. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_grid_search`

```python
get_grid_search(
    modeln: str,
    X: <built-in function array>,
    y: <built-in function array>,
    param_grid: dict = {},
    cv: int = 5,
    n_jobs: int = 6,
    random_state: int = None,
    scoring: str = 'balanced_accuracy',
    **kws
) → object
```

Grid search. 



**Args:**
 
 - <b>`modeln`</b> (str):  name of the model. 
 - <b>`X`</b> (np.array):  X matrix. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`param_grid`</b> (dict, optional):  parameter grid. Defaults to {}. 
 - <b>`cv`</b> (int, optional):  cross-validations. Defaults to 5. 
 - <b>`n_jobs`</b> (int, optional):  number of cores. Defaults to 6. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to None. 
 - <b>`scoring`</b> (str, optional):  scoring system. Defaults to 'balanced_accuracy'. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to the `GridSearchCV` function. 



**Returns:**
 
 - <b>`object`</b>:  `grid_search`. 

References:  
 - <b>`1. https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
 - <b>`2. https`</b>: //scikit-learn.org/stable/modules/model_evaluation.html 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_estimatorn2grid_search`

```python
get_estimatorn2grid_search(
    estimatorn2param_grid: dict,
    X: DataFrame,
    y: Series,
    **kws
) → dict
```

Estimator-wise grid search. 



**Args:**
 
 - <b>`estimatorn2param_grid`</b> (dict):  estimator name to the grid search map. 
 - <b>`X`</b> (pd.DataFrame):  X matrix. 
 - <b>`y`</b> (pd.Series):  y vector. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_test_scores`

```python
get_test_scores(d1: dict) → DataFrame
```

Test scores. 



**Args:**
 
 - <b>`d1`</b> (dict):  dictionary with objects. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 

TODOs:  Get best param index. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_metrics`

```python
plot_metrics(outd: str, plot: bool = False) → DataFrame
```

Plot performance metrics. 



**Args:**
 
 - <b>`outd`</b> (str):  output directory. 
 - <b>`plot`</b> (bool, optional):  make plots. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_probability`

```python
get_probability(
    estimatorn2grid_search: dict,
    X: <built-in function array>,
    y: <built-in function array>,
    colindex: str,
    coff: float = 0.5,
    test: bool = False
)
```

Classification probability. 



**Args:**
 
 - <b>`estimatorn2grid_search`</b> (dict):  estimator to the grid search map. 
 - <b>`X`</b> (np.array):  X matrix. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`colindex`</b> (str):  index column.  
 - <b>`coff`</b> (float, optional):  cut-off. Defaults to 0.5. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L369"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_grid_search`

```python
run_grid_search(
    df: DataFrame,
    colindex: str,
    coly: str,
    n_estimators: int,
    qcut: float = None,
    evaluations: list = ['prediction', 'feature importances', 'partial dependence'],
    estimatorn2param_grid: dict = None,
    drop_xs_low_complexity: bool = False,
    min_nunique: int = 5,
    max_inflation: float = 0.5,
    cols_keep: list = [],
    outp: str = None,
    test: bool = False,
    **kws
) → dict
```

Run grid search. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input data. 
 - <b>`colindex`</b> (str):  column with the index. 
 - <b>`coly`</b> (str):  column with y values. Data type bool if qcut is None else float/int. 
 - <b>`n_estimators`</b> (int):  number of estimators. 
 - <b>`qcut`</b> (float, optional):  quantile cut-off. Defaults to None. 
 - <b>`evaluations`</b> (list, optional):  evaluations types. Defaults to ['prediction','feature importances', 'partial dependence', ]. 
 - <b>`estimatorn2param_grid`</b> (dict, optional):  estimator to the parameter grid map. Defaults to None. 
 - <b>`drop_xs_low_complexity`</b> (bool, optional):  drop the low complexity columns. Defaults to False. 
 - <b>`min_nunique`</b> (int, optional):  minimum unique values allowed. Defaults to 5. 
 - <b>`max_inflation`</b> (float, optional):  maximum inflation allowed. Defaults to 0.5. 
 - <b>`cols_keep`</b> (list, optional):  columns to keep. Defaults to []. 
 - <b>`outp`</b> (str, optional):  output path. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `get_estimatorn2grid_search`. 



**Returns:**
 
 - <b>`dict`</b>:  estimator to grid search map. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L488"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_feature_predictive_power`

```python
plot_feature_predictive_power(
    df3: DataFrame,
    ax: Axes = None,
    figsize: list = [3, 3],
    **kws
) → Axes
```

Plot feature-wise predictive power. 



**Args:**
 
 - <b>`df3`</b> (pd.DataFrame):  input data. 
 - <b>`ax`</b> (plt.Axes, optional):  axes object. Defaults to None. 
 - <b>`figsize`</b> (list, optional):  figure size. Defaults to [3,3]. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_feature_predictive_power`

```python
get_feature_predictive_power(
    d0: dict,
    df01: DataFrame,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = None,
    plot: bool = False,
    drop_na: bool = False,
    **kws
) → DataFrame
```

get_feature_predictive_power _summary_ 



**Notes:**

> x-values should be scale and sign agnostic. 
>

**Args:**
 
 - <b>`d0`</b> (dict):  input dictionary. 
 - <b>`df01`</b> (pd.DataFrame):  input data,  
 - <b>`n_splits`</b> (int, optional):  number of splits. Defaults to 5. 
 - <b>`n_repeats`</b> (int, optional):  number of repeats. Defaults to 10. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to None. 
 - <b>`plot`</b> (bool, optional):  plot. Defaults to False. 
 - <b>`drop_na`</b> (bool, optional):  drop missing values. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L578"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_feature_importances`

```python
get_feature_importances(
    estimatorn2grid_search: dict,
    X: DataFrame,
    y: Series,
    scoring: str = 'roc_auc',
    n_repeats: int = 20,
    n_jobs: int = 6,
    random_state: int = None,
    plot: bool = False,
    test: bool = False,
    **kws
) → DataFrame
```

Feature importances. 



**Args:**
 
 - <b>`estimatorn2grid_search`</b> (dict):  map between estimator name and grid search object.  
 - <b>`X`</b> (pd.DataFrame):  X matrix. 
 - <b>`y`</b> (pd.Series):  y vector. 
 - <b>`scoring`</b> (str, optional):  scoring type. Defaults to 'roc_auc'. 
 - <b>`n_repeats`</b> (int, optional):  number of repeats. Defaults to 20. 
 - <b>`n_jobs`</b> (int, optional):  number of cores. Defaults to 6. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to None. 
 - <b>`plot`</b> (bool, optional):  plot. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/classify.py#L655"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_partial_dependence`

```python
get_partial_dependence(
    estimatorn2grid_search: dict,
    X: DataFrame,
    y: Series
) → DataFrame
```

Partial dependence. 



**Args:**
 
 - <b>`estimatorn2grid_search`</b> (dict):  map between estimator name and grid search object. 
 - <b>`X`</b> (pd.DataFrame):  X matrix. 
 - <b>`y`</b> (pd.Series):  y vector. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output data. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.cluster`
For clustering data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_clusters`

```python
check_clusters(df: DataFrame)
```

Check clusters. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_clusters`

```python
get_clusters(
    X: <built-in function array>,
    n_clusters: int,
    random_state=88,
    params={},
    test=False
) → dict
```

Get clusters. 



**Args:**
 
 - <b>`X`</b> (np.array):  vector 
 - <b>`n_clusters`</b> (int):  int 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to 88. 
 - <b>`params`</b> (dict, optional):  parameters for the `MiniBatchKMeans` function. Defaults to {}. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 dict:  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_n_clusters_optimum`

```python
get_n_clusters_optimum(df5: DataFrame, test=False) → int
```

Get n clusters optimum. 



**Args:**
 
 - <b>`df5`</b> (DataFrame):  input dataframe 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`int`</b>:  knee point. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_silhouette`

```python
plot_silhouette(df: DataFrame, n_clusters_optimum=None, ax=None)
```

Plot silhouette 



**Args:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`n_clusters_optimum`</b> (int, optional):  number of clusters. Defaults to None:int. 
 - <b>`ax`</b> (axes, optional):  axes object. Defaults to None:axes. 



**Returns:**
 
 - <b>`ax`</b> (axes, optional):  axes object. Defaults to None:axes. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_clusters_optimum`

```python
get_clusters_optimum(
    X: <built-in function array>,
    n_clusters=range(2, 11),
    params_clustering={},
    test=False
) → dict
```

Get optimum clusters. 



**Args:**
 
 - <b>`X`</b> (np.array):  samples to cluster in indexed format. 
 - <b>`n_clusters`</b> (int, optional):  _description_. Defaults to range(2,11). 
 - <b>`params_clustering`</b> (dict, optional):  parameters provided to `get_clusters`. Defaults to {}. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  _description_ 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_gmm_params`

```python
get_gmm_params(g, x, n_clusters=2, test=False)
```

Intersection point of the two peak Gaussian mixture Models (GMMs). 



**Args:**
 
 - <b>`out`</b> (str):  `coff` only or `params` for all the parameters. 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_gmm_intersection`

```python
get_gmm_intersection(x, two_pdfs, means, weights, test=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `cluster_1d`

```python
cluster_1d(
    ds: Series,
    n_clusters: int,
    clf_type='gmm',
    random_state=1,
    test=False,
    returns=['coff'],
    **kws_clf
) → dict
```

Cluster 1D data. 



**Args:**
 
 - <b>`ds`</b> (Series):  series. 
 - <b>`n_clusters`</b> (int):  number of clusters. 
 - <b>`clf_type`</b> (str, optional):  type of classification. Defaults to 'gmm'. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to 88. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 
 - <b>`returns`</b> (list, optional):  return format. Defaults to ['df','coff','ax','model']. 
 - <b>`ax`</b> (axes, optional):  axes object. Defaults to None. 



**Raises:**
 
 - <b>`ValueError`</b>:  clf_type 



**Returns:**
 
 - <b>`dict`</b>:  _description_ 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/cluster.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_pos_umap`

```python
get_pos_umap(df1, spread=100, test=False, k='', **kws) → DataFrame
```

Get positions of the umap points. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe 
 - <b>`spread`</b> (int, optional):  spead extent. Defaults to 100. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 
 - <b>`k`</b> (str, optional):  number of clusters. Defaults to ''. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.compare`
For comparison related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/compare.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cols_x_for_comparison`

```python
get_cols_x_for_comparison(
    df1: DataFrame,
    cols_y: list,
    cols_index: list,
    cols_drop: list = [],
    cols_dropby_patterns: list = [],
    coff_rs: float = 0.7,
    min_nunique: int = 5,
    max_inflation: int = 50,
    verbose: bool = False,
    test: bool = False
) → dict
```

Identify X columns. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`cols_y`</b> (list):  y columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/compare.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_filteredby_samples`

```python
to_filteredby_samples(
    df1: DataFrame,
    colindex: str,
    colsample: str,
    coff_samples_min: int,
    colsubset: str,
    coff_subsets_min: int = 2
) → DataFrame
```

Filter table before calculating differences. (1) Retain minimum number of samples per item representing a subset and (2) Retain minimum number of subsets per item. 

Parameters:          df1 (pd.DataFrame): input table.  colindex (str): column containing items.  colsample (str): column containing samples.  coff_samples_min (int): minimum number of samples.  colsubset (str): column containing subsets.  coff_subsets_min (int): minimum number of subsets. Defaults to 2.  



**Returns:**
  pd.DataFrame  



**Examples:**
 

**Parameters:**
  colindex='genes id',  colsample='sample id',  coff_samples_min=3,  colsubset= 'pLOF or WT'   coff_subsets_min=2,   


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/compare.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_preprocessed_data`

```python
to_preprocessed_data(
    df1: DataFrame,
    columns: dict,
    fill_missing_desc_value: bool = False,
    fill_missing_cont_value: bool = False,
    normby_zscore: bool = False,
    verbose: bool = False,
    test: bool = False
) → DataFrame
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/compare.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_comparison`

```python
get_comparison(
    df1: DataFrame,
    d1: dict = None,
    coff_p: float = 0.05,
    between_ys: bool = False,
    verbose: bool = False,
    **kws
)
```

Compare the x and y columns. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`d1`</b> (dict):  columns dict, output of `get_cols_x_for_comparison`.   
 - <b>`between_ys`</b> (bool):  compare y's 



**Notes:**

> Column information: d1={'cols_index': ['id'], 'cols_x': {'cont': [], 'desc': []}, 'cols_y': {'cont': [], 'desc': []}} 
>Comparison types: 1. continuous vs continuous -> correlation 2. decrete vs continuous -> difference 3. decrete vs decrete -> FE or chi square 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/compare.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_strings`

```python
compare_strings(l0: list, l1: list, cutoff: float = 0.5) → DataFrame
```

Compare two lists of strings. 



**Parameters:**
 
 - <b>`l0`</b> (list):  list of strings.  
 - <b>`l1`</b> (list):  list of strings to compare with.  
 - <b>`cutoff`</b> (float):  threshold to filter the comparisons. 



**Returns:**
 table with the similarity scores. 

TODOs:  1. Add option for semantic similarity. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.corr`
For correlation stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_spearmanr`

```python
get_spearmanr(
    x: <built-in function array>,
    y: <built-in function array>
) → tuple
```

Get Spearman correlation coefficient. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 



**Returns:**
 
 - <b>`tuple`</b>:  rs, p-value 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_pearsonr`

```python
get_pearsonr(x: <built-in function array>, y: <built-in function array>) → tuple
```

Get Pearson correlation coefficient. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 



**Returns:**
 
 - <b>`tuple`</b>:  rs, p-value 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_corr_resampled`

```python
get_corr_resampled(
    x: <built-in function array>,
    y: <built-in function array>,
    method='spearman',
    ci_type='max',
    cv: int = 5,
    random_state=1,
    verbose=False
) → tuple
```

Get correlations after resampling. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`method`</b> (str, optional):  method name. Defaults to 'spearman'. 
 - <b>`ci_type`</b> (str, optional):  confidence interval type. Defaults to 'max'. 
 - <b>`cv`</b> (int, optional):  number of resamples. Defaults to 5. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to 1. 



**Returns:**
 
 - <b>`tuple`</b>:  mean correlation coefficient, confidence interval 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `corr_to_str`

```python
corr_to_str(
    method: str,
    r: float,
    p: float,
    show_n: bool = True,
    n: int = None,
    show_n_prefix: str = '',
    fmt='<',
    ci=None,
    ci_type=None,
    magnitide=True
) → str
```

Correlation to string 



**Args:**
 
 - <b>`method`</b> (str):  method name. 
 - <b>`r`</b> (float):  correlation coefficient. 
 - <b>`p`</b> (float):  p-value 
 - <b>`fmt`</b> (str, optional):  format of the p-value. Defaults to '<'. 
 - <b>`n`</b> (bool, optional):  sample size. Defaults to True. 
 - <b>`ci`</b> (_type_, optional):  confidence interval. Defaults to None. 
 - <b>`ci_type`</b> (_type_, optional):  confidence interval type. Defaults to None. 
 - <b>`magnitide`</b> (bool, optional):  show magnitude of the sample size. Defaults to True. 



**Returns:**
 
 - <b>`str`</b>:  string with the correation stats.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_corr`

```python
get_corr(
    x: <built-in function array>,
    y: <built-in function array>,
    method='spearman',
    resample=False,
    ci_type='max',
    sample_size_min=10,
    magnitide=True,
    outstr=False,
    kws_to_str={},
    verbose: bool = False,
    **kws_boots
)
```

Correlation between vectors (wrapper). 



**Args:**
 
 - <b>`x`</b> (np.array):  x. 
 - <b>`y`</b> (np.array):  y. 
 - <b>`method`</b> (str, optional):  method name. Defaults to 'spearman'. 
 - <b>`resample`</b> (bool, optional):  resampling. Defaults to False. 
 - <b>`ci_type`</b> (str, optional):  confidence interval type. Defaults to 'max'. 
 - <b>`magnitide`</b> (bool, optional):  show magnitude. Defaults to True. 
 - <b>`outstr`</b> (bool, optional):  output as string. Defaults to False. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `get_corr_resampled` function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_corrs`

```python
get_corrs(
    df1: DataFrame,
    method: str,
    cols: list = None,
    cols_with: list = None,
    pairs: list = None,
    coff_inflation_min: float = None,
    fast: bool = False,
    test: bool = False,
    verbose: bool = False,
    **kws
)
```

Correlate columns of a dataframes. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`method`</b> (str):  method of correlation `spearman` or `pearson`.         
 - <b>`cols`</b> (str):  columns. 
 - <b>`cols_with`</b> (str):  columns to correlate with i.e. variable2. 
 - <b>`pairs`</b> (list):  list of tuples of column (variable) pairs. 
 - <b>`fast`</b> (bool):  use parallel-processing if True. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `get_corr` function. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 0. Use `lib.set.get_pairs` to get the combinations. 1. Provide 2D array to `scipy.stats.spearmanr`? 2. Compare with `Pingouin`'s equivalent function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_partial_corrs`

```python
get_partial_corrs(
    df: DataFrame,
    xs: list,
    ys: list,
    method='spearman',
    splits=5
) → DataFrame
```

Get partial correlations. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`xs`</b> (list):  columns used as x variables. 
 - <b>`ys`</b> (list):  columns used as y variables. 
 - <b>`method`</b> (str, optional):  method name. Defaults to 'spearman'. 
 - <b>`splits`</b> (int, optional):  number of splits. Defaults to 5. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_collinearity`

```python
check_collinearity(
    df1: DataFrame,
    threshold: float = 0.7,
    colvalue: str = '$r_s$',
    cols_variable: list = ['variable1', 'variable2'],
    coff_pval: float = 0.05,
    method: str = 'spearman',
    coff_inflation_min: int = 50
) → Series
```

Check collinearity. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`threshold`</b> (float):  minimum threshold for the colinearity. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 1. Calculate variance inflation factor (VIF). 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/corr.py#L377"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `pairwise_chi2`

```python
pairwise_chi2(df1: DataFrame, cols_values: list) → DataFrame
```

Pairwise chi2 test. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  pd.DataFrame 
 - <b>`cols_values`</b> (list):  list of columns. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 0. use `lib.set.get_pairs` to get the combinations. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.diff`
For difference related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_demo_data`

```python
get_demo_data() → DataFrame
```

Demo data to test the differences. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_classes`

```python
compare_classes(x, y, method=None)
```

 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_classes_many`

```python
compare_classes_many(df1: DataFrame, cols_y: list, cols_x: list) → DataFrame
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_pval`

```python
get_pval(
    df: DataFrame,
    colvalue='value',
    colsubset='subset',
    colvalue_bool=False,
    colindex=None,
    subsets=None,
    test=False,
    fun=None
) → tuple
```

Get p-value. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`colvalue`</b> (str, optional):  column with values. Defaults to 'value'. 
 - <b>`colsubset`</b> (str, optional):  column with subsets. Defaults to 'subset'. 
 - <b>`colvalue_bool`</b> (bool, optional):  column with boolean values. Defaults to False. 
 - <b>`colindex`</b> (str, optional):  column with the index. Defaults to None. 
 - <b>`subsets`</b> (list, optional):  subset types. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 
 - <b>`fun`</b> (function, optional):  function. Defaults to None. 



**Raises:**
 
 - <b>`ArgumentError`</b>:  colvalue or colsubset not found in df. 
 - <b>`ValueError`</b>:  need only 2 subsets. 



**Returns:**
 
 - <b>`tuple`</b>:  stat,p-value 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stat`

```python
get_stat(
    df1: DataFrame,
    colsubset: str,
    colvalue: str,
    colindex: str,
    subsets=None,
    cols_subsets=['subset1', 'subset2'],
    df2=None,
    stats=[<function mean at 0x7f7c744d2e60>, <function median at 0x7f7c74119e60>, <function var at 0x7f7c744d5320>, <built-in function len>],
    coff_samples_min=None,
    verb=False,
    **kws
) → DataFrame
```

Get statistics. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`colvalue`</b> (str, optional):  column with values. Defaults to 'value'. 
 - <b>`colsubset`</b> (str, optional):  column with subsets. Defaults to 'subset'. 
 - <b>`colindex`</b> (str, optional):  column with the index. Defaults to None. 
 - <b>`subsets`</b> (list, optional):  subset types. Defaults to None. 
 - <b>`cols_subsets`</b> (list, optional):  columns with subsets. Defaults to ['subset1', 'subset2']. 
 - <b>`df2`</b> (DataFrame, optional):  second dataframe. Defaults to None. 
 - <b>`stats`</b> (list, optional):  summary statistics. Defaults to [np.mean,np.median,np.var]+[len]. 
 - <b>`coff_samples_min`</b> (int, optional):  minimum sample size required. Defaults to None. 
 - <b>`verb`</b> (bool, optional):  verbose. Defaults to False. 

Keyword Arguments: 
 - <b>`kws`</b>:  parameters provided to `get_pval` function. 



**Raises:**
 
 - <b>`ArgumentError`</b>:  colvalue or colsubset not found in df. 
 - <b>`ValueError`</b>:  len(subsets)<2 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 1. Rename to more specific `get_diff`, also other `get_stat*`/`get_pval*` functions. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats`

```python
get_stats(
    df1: DataFrame,
    colsubset: str,
    cols_value: list,
    colindex: str,
    subsets=None,
    df2=None,
    cols_subsets=['subset1', 'subset2'],
    stats=[<function mean at 0x7f7c744d2e60>, <function median at 0x7f7c74119e60>, <function var at 0x7f7c744d5320>, <built-in function len>],
    axis=0,
    test=False,
    **kws
) → DataFrame
```

Get statistics by iterating over columns wuth values. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`colsubset`</b> (str, optional):  column with subsets. 
 - <b>`cols_value`</b> (list):  list of columns with values. 
 - <b>`colindex`</b> (str, optional):  column with the index. 
 - <b>`subsets`</b> (list, optional):  subset types. Defaults to None. 
 - <b>`df2`</b> (DataFrame, optional):  second dataframe, e.g. `pd.DataFrame({"subset1":['test'],"subset2":['reference']})`. Defaults to None. 
 - <b>`cols_subsets`</b> (list, optional):  columns with subsets. Defaults to ['subset1', 'subset2']. 
 - <b>`stats`</b> (list, optional):  summary statistics. Defaults to [np.mean,np.median,np.var]+[len]. 
 - <b>`axis`</b> (int, optional):  1 if different tests else use 0. Defaults to 0. 

Keyword Arguments: 
 - <b>`kws`</b>:  parameters provided to `get_pval` function. 



**Raises:**
 
 - <b>`ArgumentError`</b>:  colvalue or colsubset not found in df. 
 - <b>`ValueError`</b>:  len(subsets)<2 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 1. No column prefix if `len(cols_value)==1`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_significant_changes`

```python
get_significant_changes(
    df1: DataFrame,
    coff_p=0.025,
    coff_q=0.1,
    alpha=None,
    changeby='mean',
    value_aggs=['mean', 'median']
) → DataFrame
```

Get significant changes. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`coff_p`</b> (float, optional):  cutoff on p-value. Defaults to 0.025. 
 - <b>`coff_q`</b> (float, optional):  cutoff on q-value. Defaults to 0.1. 
 - <b>`alpha`</b> (float, optional):  alias for `coff_p`. Defaults to None. 
 - <b>`changeby`</b> (str, optional):  "" if check for change by both mean and median. Defaults to "". 
 - <b>`value_aggs`</b> (list, optional):  values to aggregate. Defaults to ['mean','median']. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `apply_get_significant_changes`

```python
apply_get_significant_changes(
    df1: DataFrame,
    cols_value: list,
    cols_groupby: list,
    cols_grouped: list,
    fast=False,
    **kws
) → DataFrame
```

Apply on dataframe to get significant changes. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`cols_value`</b> (list):  columns with values. 
 - <b>`cols_groupby`</b> (list):  columns with groups. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats_groupby`

```python
get_stats_groupby(
    df1: DataFrame,
    cols_group: list,
    coff_p: float = 0.05,
    coff_q: float = 0.1,
    alpha=None,
    fast=False,
    **kws
) → DataFrame
```

Iterate over groups, to get the differences. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`cols_group`</b> (list):  columns to interate over. 
 - <b>`coff_p`</b> (float, optional):  cutoff on p-value. Defaults to 0.025. 
 - <b>`coff_q`</b> (float, optional):  cutoff on q-value. Defaults to 0.1. 
 - <b>`alpha`</b> (float, optional):  alias for `coff_p`. Defaults to None. 
 - <b>`fast`</b> (bool, optional):  parallel processing. Defaults to False. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_diff`

```python
get_diff(
    df1: DataFrame,
    cols_x: list,
    cols_y: list,
    cols_index: list,
    cols_group: list,
    coff_p: float = None,
    test: bool = False,
    **kws
) → DataFrame
```

Wrapper around the `get_stats_groupby` 

Keyword parameters:  cols=['variable x','variable y'],  coff_p=0.05,  coff_q=0.01,  colindex=['id'],     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/diff.py#L458"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `binby_pvalue_coffs`

```python
binby_pvalue_coffs(
    df1: DataFrame,
    coffs=[0.01, 0.05, 0.1],
    color=False,
    testn='MWU test, FDR corrected',
    colindex='genes id',
    colgroup='tissue',
    preffix='',
    colns=None,
    palette=None
) → tuple
```

Bin data by pvalue cutoffs. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`coffs`</b> (list, optional):  cut-offs. Defaults to [0.01,0.05,0.25]. 
 - <b>`color`</b> (bool, optional):  color asignment. Defaults to False. 
 - <b>`testn`</b> (str, optional):  test number. Defaults to 'MWU test, FDR corrected'. 
 - <b>`colindex`</b> (str, optional):  column with index. Defaults to 'genes id'. 
 - <b>`colgroup`</b> (str, optional):  column with the groups. Defaults to 'tissue'. 
 - <b>`preffix`</b> (str, optional):  prefix. Defaults to ''. 
 - <b>`colns`</b> (_type_, optional):  columns number. Defaults to None. 
 - <b>`notcountedpalette`</b> (_type_, optional):  _description_. Defaults to None. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 



**Notes:**

> 1. To be deprecated in the favor of the functions used for enrichment analysis for example. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.enrich`
For enrichment related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/enrich.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_enrichment`

```python
get_enrichment(
    df1: DataFrame,
    df2: DataFrame,
    background: int,
    colid: str = 'gene id',
    colref: str = 'gene set id',
    colrefname: str = 'gene set name',
    colreftype: str = 'gene set type',
    colrank: str = 'rank',
    outd: str = None,
    name: str = None,
    cutoff: float = 0.05,
    permutation_num: int = 1000,
    verbose: bool = False,
    no_plot: bool = True,
    **kws_prerank
)
```

Get enrichments between sets. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  test data. 
 - <b>`df2`</b> (pd.DataFrame):  reference set data. 
 - <b>`background`</b> (int):  background size. 
 - <b>`colid`</b> (str, optional):  column containing unique ids of the elements. Defaults to 'gene id'. 
 - <b>`colref`</b> (str, optional):  column containing the unique ids of the sets. Defaults to 'gene set id'. 
 - <b>`colrefname`</b> (str, optional):  column containing names of the sets. Defaults to 'gene set name'. 
 - <b>`colreftype`</b> (str, optional):  column containing the type/group name of the sets. Defaults to 'gene set type'. 
 - <b>`colrank`</b> (str, optional):  column containing the ranks. Defaults to 'rank'. 
 - <b>`outd`</b> (str, optional):  output directory path. Defaults to None. 
 - <b>`name`</b> (str, optional):  name of the result. Defaults to None. 
 - <b>`cutoff`</b> (float, optional):  p-value cutoff. Defaults to 0.05. 
 - <b>`verbose`</b> (bool, optional):  verbose. Defaults to False. 
 - <b>`no_plot`</b> (bool, optional):  do no plot. Defaults to True. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  if rank -> high rank first within the leading edge gene ids. 



**Notes:**

> 1. Unique ids are provided as inputs. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/enrich.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_enrichments`

```python
get_enrichments(
    df1: DataFrame,
    df2: DataFrame,
    background: int,
    coltest: str = 'subset',
    colid: str = 'gene id',
    colref: str = 'gene set id',
    colreftype: str = 'gene set type',
    fast: bool = False,
    **kws
) → DataFrame
```

Get enrichments between sets, iterate over types/groups of test elements e.g. upregulated and downregulated genes. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  test data. 
 - <b>`df2`</b> (pd.DataFrame):  reference set data. 
 - <b>`background`</b> (int):  background size. 
 - <b>`colid`</b> (str, optional):  column containing unique ids of the elements. Defaults to 'gene id'. 
 - <b>`colref`</b> (str, optional):  column containing the unique ids of the sets. Defaults to 'gene set id'. 
 - <b>`colrefname`</b> (str, optional):  column containing names of the sets. Defaults to 'gene set name'. 
 - <b>`colreftype`</b> (str, optional):  column containing the type/group name of the sets. Defaults to 'gene set type'. 
 - <b>`fast`</b> (bool, optional):  parallel processing. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.fit`
For fitting data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_curve_fit`

```python
fit_curve_fit(
    func,
    xdata: <built-in function array> = None,
    ydata: <built-in function array> = None,
    bounds: tuple = (-inf, inf),
    test=False,
    plot=False
) → tuple
```

Wrapper around `scipy`'s `curve_fit`. 



**Args:**
 
 - <b>`func`</b> (function):  fitting function. 
 - <b>`xdata`</b> (np.array, optional):  x data. Defaults to None. 
 - <b>`ydata`</b> (np.array, optional):  y data. Defaults to None. 
 - <b>`bounds`</b> (tuple, optional):  bounds. Defaults to (-np.inf, np.inf). 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 
 - <b>`plot`</b> (bool, optional):  plot. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_gauss_bimodal`

```python
fit_gauss_bimodal(
    data: <built-in function array>,
    bins: int = 50,
    expected: tuple = (1, 0.2, 250, 2, 0.2, 125),
    test=False
) → tuple
```

Fit bimodal gaussian distribution to the data in vector format. 



**Args:**
 
 - <b>`data`</b> (np.array):  vector. 
 - <b>`bins`</b> (int, optional):  bins. Defaults to 50. 
 - <b>`expected`</b> (tuple, optional):  expected parameters. Defaults to (1,.2,250,2,.2,125). 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  _description_ 



**Notes:**

> Observed better performance with `roux.stat.cluster.cluster_1d`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_grid`

```python
get_grid(
    x: <built-in function array>,
    y: <built-in function array>,
    z: <built-in function array> = None,
    off: int = 0,
    grids: int = 100,
    method='linear',
    test=False,
    **kws
) → tuple
```

2D grids from 1d data. 



**Args:**
 
 - <b>`x`</b> (np.array):  vector. 
 - <b>`y`</b> (np.array):  vector. 
 - <b>`z`</b> (np.array, optional):  vector. Defaults to None. 
 - <b>`off`</b> (int, optional):  offsets. Defaults to 0. 
 - <b>`grids`</b> (int, optional):  grids. Defaults to 100. 
 - <b>`method`</b> (str, optional):  method. Defaults to 'linear'. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_gaussian2d`

```python
fit_gaussian2d(
    x: <built-in function array>,
    y: <built-in function array>,
    z: <built-in function array>,
    grid=True,
    grids=20,
    method='linear',
    off=0,
    rescalez=True,
    test=False
) → tuple
```

Fit gaussian 2D. 



**Args:**
 
 - <b>`x`</b> (np.array):  vector. 
 - <b>`y`</b> (np.array):  vector. 
 - <b>`z`</b> (np.array):  vector. 
 - <b>`grid`</b> (bool, optional):  grid. Defaults to True. 
 - <b>`grids`</b> (int, optional):  grids. Defaults to 20. 
 - <b>`method`</b> (str, optional):  method. Defaults to 'linear'. 
 - <b>`off`</b> (int, optional):  offsets. Defaults to 0. 
 - <b>`rescalez`</b> (bool, optional):  rescalez. Defaults to True. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `fit_2d_distribution_kde`

```python
fit_2d_distribution_kde(
    x: <built-in function array>,
    y: <built-in function array>,
    bandwidth: float,
    xmin: float = None,
    xmax: float = None,
    xbins=100j,
    ymin: float = None,
    ymax: float = None,
    ybins=100j,
    test=False,
    **kwargs
) → tuple
```

2D kernel density estimate (KDE). 



**Notes:**

> Cut off outliers: quantile_coff=0.01 params_grid=merge_dicts([ df01.loc[:,var2col.values()].quantile(quantile_coff).rename(index=flip_dict({f"{k}min":var2col[k] for k in var2col})).to_dict(), df01.loc[:,var2col.values()].quantile(1-quantile_coff).rename(index=flip_dict({f"{k}max":var2col[k] for k in var2col})).to_dict(), ]) 
>

**Args:**
 
 - <b>`x`</b> (np.array):  vector. 
 - <b>`y`</b> (np.array):  vector. 
 - <b>`bandwidth`</b> (float):  bandwidth 
 - <b>`xmin`</b> (float, optional):  x minimum. Defaults to None. 
 - <b>`xmax`</b> (float, optional):  x maximum. Defaults to None. 
 - <b>`xbins`</b> (_type_, optional):  x bins. Defaults to 100j. 
 - <b>`ymin`</b> (float, optional):  y minimum. Defaults to None. 
 - <b>`ymax`</b> (float, optional):  y maximum. Defaults to None. 
 - <b>`ybins`</b> (_type_, optional):  y bins. Defaults to 100j. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_poly_fit`

```python
check_poly_fit(d: DataFrame, xcol: str, ycol: str, degmax: int = 5) → DataFrame
```

Check the fit of a polynomial equations. 



**Args:**
 
 - <b>`d`</b> (pd.DataFrame):  input dataframe. 
 - <b>`xcol`</b> (str):  column containing the x values. 
 - <b>`ycol`</b> (str):  column containing the y values. 
 - <b>`degmax`</b> (int, optional):  degree maximum. Defaults to 5. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  _description_ 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `mlr_2`

```python
mlr_2(df: DataFrame, coly: str, colxs: list) → tuple
```

Multiple linear regression between two variables. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input dataframe. 
 - <b>`coly`</b> (str):  column  containing y values. 
 - <b>`colxs`</b> (list):  columns containing x values. 



**Returns:**
 
 - <b>`tuple`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/fit.py#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_mlr_2_str`

```python
get_mlr_2_str(df: DataFrame, coly: str, colxs: list) → str
```

Get the result of the multiple linear regression between two variables as a string. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input dataframe. 
 - <b>`coly`</b> (str):  column  containing y values. 
 - <b>`colxs`</b> (list):  columns containing x values. 



**Returns:**
 
 - <b>`str`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.io`
For input/output of stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/io.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `perc_label`

```python
perc_label(a, b=None, bracket=True)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/io.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `pval2annot`

```python
pval2annot(
    pval: float,
    alternative: str = None,
    alpha: float = 0.05,
    fmt: str = '*',
    power: bool = True,
    linebreak: bool = False,
    replace_prefix: str = None
)
```

P/Q-value to annotation. 



**Parameters:**
 
 - <b>`fmt`</b> (str):  *|<|'num' 




<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat`




**Global Variables**
---------------
- **binary**
- **io**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.network`
For network related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/network.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_subgraphs`

```python
get_subgraphs(df1: DataFrame, source: str, target: str) → DataFrame
```

Subgraphs from the the edge list. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input dataframe containing edge-list. 
 - <b>`source`</b> (str):  source node. 
 - <b>`target`</b> (str):  taget node. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.norm`
For normalisation. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/norm.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `norm_by_quantile`

```python
norm_by_quantile(X: <built-in function array>) → <built-in function array>
```

Normalize the columns of X to each have the same distribution. 



**Notes:**

> Given an expression matrix (microarray data, read counts, etc) of M genes by N samples, quantile normalization ensures all samples have the same spread of data (by construction). 
>The data across each row are averaged to obtain an average column. Each column quantile is replaced with the corresponding quantile of the average column. 
>

**Parameters:**
 
 - <b>`X `</b>:  2D array of float, shape (M, N). The input data, with M rows (genes/features) and N columns (samples). 



**Returns:**
 
 - <b>`Xn `</b>:  2D array of float, shape (M, N). The normalized data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/norm.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `norm_by_gaussian_kde`

```python
norm_by_gaussian_kde(
    values: <built-in function array>
) → <built-in function array>
```

Normalise matrix by gaussian KDE. 



**Args:**
 
 - <b>`values`</b> (np.array):  input matrix. 



**Returns:**
 
 - <b>`np.array`</b>:  output matrix. 

References:  
 - <b>`https`</b>: //github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/norm.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `zscore`

```python
zscore(df: DataFrame, cols: list = None) → DataFrame
```

Z-score. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input table. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output table. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/norm.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `zscore_robust`

```python
zscore_robust(a: <built-in function array>) → <built-in function array>
```

Robust Z-score. 



**Args:**
 
 - <b>`a`</b> (np.array):  input data. 



**Returns:**
 
 - <b>`np.array`</b>:  output. 



**Example:**
 t = sc.stats.norm.rvs(size=100, scale=1, random_state=123456) plt.hist(t,bins=40) plt.hist(apply_zscore_robust(t),bins=40) print(np.median(t),np.median(apply_zscore_robust(t))) 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.paired`
For paired stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ratio_sorted`

```python
get_ratio_sorted(a: float, b: float, increase=True) → float
```

Get ratio sorted. 



**Args:**
 
 - <b>`a`<.py#1. 
 - <b>`b`<.py#2. 
 - <b>`increase`</b> (bool, optional):  check for increase. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `diff`

```python
diff(a: float, b: float, absolute=True) → float
```

Get difference 



**Args:**
 
 - <b>`a`<.py#1. 
 - <b>`b`<.py#2. 
 - <b>`absolute`</b> (bool, optional):  get absolute difference. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_diff_sorted`

```python
get_diff_sorted(a: float, b: float) → float
```

Difference sorted/absolute. 



**Args:**
 
 - <b>`a`<.py#1. 
 - <b>`b`<.py#2. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `balance`

```python
balance(a: float, b: float, absolute=True) → float
```

Balance. 



**Args:**
 
 - <b>`a`<.py#1. 
 - <b>`b`<.py#2. 
 - <b>`absolute`</b> (bool, optional):  absolute difference. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_paired_sets_stats`

```python
get_paired_sets_stats(l1: list, l2: list, test: bool = False) → list
```

Paired stats comparing two sets. 



**Args:**
 
 - <b>`l1`<.py#1. 
 - <b>`l2`<.py#2. 
 - <b>`test`</b> (bool):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  tuple (overlap, intersection, union, ratio). 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats_paired`

```python
get_stats_paired(
    df1: DataFrame,
    cols: list,
    input_logscale: bool,
    prefix: str = None,
    drop_cols: bool = False,
    unidirectional_stats: list = ['min', 'max'],
    fast: bool = False
) → DataFrame
```

Paired stats, row-wise. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`cols`</b> (list):  columns. 
 - <b>`input_logscale`</b> (bool):  if the input data is log-scaled. 
 - <b>`prefix`</b> (str, optional):  prefix of the output column/s. Defaults to None. 
 - <b>`drop_cols`</b> (bool, optional):  drop these columns. Defaults to False. 
 - <b>`unidirectional_stats`</b> (list, optional):  column-wise status. Defaults to ['min','max']. 
 - <b>`fast`</b> (bool, optional):  parallel processing. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats_paired_agg`

```python
get_stats_paired_agg(
    x: <built-in function array>,
    y: <built-in function array>,
    ignore: bool = False,
    verb: bool = True
) → Series
```

Paired stats aggregated, for example, to classify 2D distributions. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`ignore`</b> (bool, optional):  suppress warnings. Defaults to False. 
 - <b>`verb`</b> (bool, optional):  verbose. Defaults to True. 



**Returns:**
 
 - <b>`pd.Series`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/paired.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `classify_sharing`

```python
classify_sharing(
    df1: DataFrame,
    column_value: str,
    bins: list = [0, 25, 75, 100],
    labels: list = ['low', 'medium', 'high'],
    prefix: str = '',
    verbose: bool = False
) → DataFrame
```

Classify sharing % calculated from Jaccard index. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`column_value`</b> (str):  column with values. 
 - <b>`bins`</b> (list):  bins. Defaults to [0,25,75,100]. 
 - <b>`labels`</b> (list):  bin labels. Defaults to ['low','medium','high'], 
 - <b>`prefix`</b> (str):  prefix of the columns. 
 - <b>`verbose`</b> (bool):  verbose. Defaults to False. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.regress`
For regression. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_columns_renamed_for_regression`

```python
to_columns_renamed_for_regression(df1: DataFrame, columns: dict) → DataFrame
```

[UNDER DEVELOPMENT] 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_covariates`

```python
check_covariates(
    df1,
    covariates,
    colindex,
    plot: bool = False,
    **kws_drop_low_complexity
)
```

[UNDER DEVELOPMENT] Quality check covariates for redundancy. 

Todos:  Support continuous value covariates using `from roux.stat.compare import get_comparison`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_input_data_for_regression`

```python
to_input_data_for_regression(
    df1: DataFrame,
    cols_y: list,
    cols_index: list,
    desc_test_values: dict,
    verbose: bool = False,
    test: bool = False,
    **kws
) → tuple
```

Input data for the regression. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`cols_y`</b> (list):  y columns. 
 - <b>`cols_index`</b> (list):  index columns. 



**Returns:**
 Output table. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_formulas`

```python
to_formulas(
    formula: str,
    covariates: list,
    covariate_dtypes: dict = None
) → list
```

[UNDER DEVELOPMENT] Generate formulas. 



**Notes:**

> covariate_dtypes=data.dtypes.to_dict() 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_stats_regression`

```python
get_stats_regression(
    data: DataFrame,
    formulas: dict = {},
    variable: str = None,
    converged_only=False,
    out='df',
    verb=False,
    test=False,
    **kws_model
) → DataFrame
```

Get stats from regression models. 



**Args:**
 
 - <b>`data`</b> (DataFrame):  input dataframe. 
 - <b>`formulas`</b> (dict, optional):  base formula e.g. 'y ~ x' to model name map. Defaults to {}. 
 - <b>`variable`</b> (str, optional):  variable name e.g. 'C(variable)[T.True]', used to retrieve the stats for. Defaults to None. 
 - <b>`# covariates (list, optional)`</b>:  variables. Defaults to None. 
 - <b>`converged_only`</b> (bool, optional):  get the stats from the converged models only. Defaults to False. 
 - <b>`out`</b> (str, optional):  output format. Defaults to 'df'. 
 - <b>`verb`</b> (bool, optional):  verbose. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_filteredby_variable`

```python
to_filteredby_variable(
    df1: DataFrame,
    variable: str,
    colindex: str,
    coff_q: float = 0.1,
    coff_p_covariates: float = 0.05,
    plot: bool = False,
    test: bool = False
) → DataFrame
```

Filter regression statistics. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`variable`</b> (str):  variable name to filter by. 
 - <b>`colindex`</b> (str):  columns with index. 
 - <b>`coff_q`</b> (float, optional):  cut-off on the q-value. Defaults to 0.1. 
 - <b>`by_covariates`</b> (bool, optional):  filter by these covaliates. Defaults to True. 
 - <b>`coff_p_covariates`</b> (float, optional):  cut-off on the p-value for the covariates. Defaults to 0.05. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Raises:**
 
 - <b>`ValueError`</b>:  pval. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output. 



**Notes:**

> Filtering steps: 1. By variable of interest. 2. By statistical significance. 3. By statistical significance of co-variates. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L454"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_lr_test`

```python
run_lr_test(
    data: DataFrame,
    formula: str,
    covariate: str,
    col_group: str,
    params_model: dict = {'reml': False}
) → tuple
```

Run LR test. 



**Args:**
 
 - <b>`data`</b> (pd.DataFrame):  input data. 
 - <b>`formula`</b> (str):  formula. 
 - <b>`covariate`</b> (str):  covariate. 
 - <b>`col_group`</b> (str):  column with the group. 
 - <b>`params_model`</b> (dict, optional):  parameters of the model. Defaults to {'reml':False}. 



**Returns:**
 
 - <b>`tuple`</b>:  output tupe (stat, pval,dres). 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_residuals_versus_fitted`

```python
plot_residuals_versus_fitted(model: object) → Axes
```

plot Residuals Versus Fitted (RVF). 



**Args:**
 
 - <b>`model`</b> (object):  model. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L535"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_residuals_versus_groups`

```python
plot_residuals_versus_groups(model: object) → Axes
```

plot Residuals Versus groups. 



**Args:**
 
 - <b>`model`</b> (object):  model. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/regress.py#L556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_model_qcs`

```python
plot_model_qcs(model: object)
```

Plot Quality Checks. 



**Args:**
 
 - <b>`model`</b> (object):  model. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.set`
For set related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_intersection_stats`

```python
get_intersection_stats(df, coltest, colset, background_size=None)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_set_enrichment_stats`

```python
get_set_enrichment_stats(test, sets, background, fdr_correct=True)
```

test:  get_set_enrichment_stats(background=range(120),  test=range(100),  sets={f"set {i}":list(np.unique(np.random.randint(low=100,size=i+1))) for i in range(100)})  # background is int  get_set_enrichment_stats(background=110,  test=unique(range(100)),  sets={f"set {i}":unique(np.random.randint(low=140,size=i+1)) for i in range(0,140,10)})                         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `test_set_enrichment`

```python
test_set_enrichment(tests_set2elements, test2_set2elements, background_size)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_paired_sets_stats`

```python
get_paired_sets_stats(l1, l2)
```

overlap, intersection, union, ratio 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_enrichment`

```python
get_enrichment(
    df1,
    df2,
    background,
    colid='gene id',
    colref='gene set id',
    colrefname='gene set name',
    colreftype='gene set type',
    colrank='rank',
    outd=None,
    name=None,
    cutoff=0.05,
    permutation_num=1000,
    verbose=False,
    no_plot=True,
    **kws_prerank
)
```

:return leading edge gene ids: high rank first 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/set.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_enrichments`

```python
get_enrichments(
    df1,
    df2,
    background,
    coltest='subset',
    colid='gene id',
    colref='gene set id',
    colreftype='gene set type',
    fast=False,
    **kws
)
```

:param df1: test sets :param df2: reference sets 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.solve`
For solving equations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/solve.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_intersection_locations`

```python
get_intersection_locations(
    y1: <built-in function array>,
    y2: <built-in function array>,
    test: bool = False,
    x: <built-in function array> = None
) → list
```

Get co-ordinates of the intersection (x[idx]). 



**Args:**
 
 - <b>`y1`</b> (np.array):  vector. 
 - <b>`y2`</b> (np.array):  vector. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`x`</b> (np.array, optional):  vector. Defaults to None. 



**Returns:**
 
 - <b>`list`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.transform`
For transformations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plog`

```python
plog(x, p: float, base: int)
```

Psudo-log. 



**Args:**
 
 - <b>`x`</b> (float|np.array):  input. 
 - <b>`p`</b> (float):  pseudo-count. 
 - <b>`base`</b> (int):  base of the log. 



**Returns:**
 output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `anti_plog`

```python
anti_plog(x, p: float, base: int)
```

Anti-psudo-log. 



**Args:**
 
 - <b>`x`</b> (float|np.array):  input. 
 - <b>`p`</b> (float):  pseudo-count. 
 - <b>`base`</b> (int):  base of the log. 



**Returns:**
 output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_pval`

```python
log_pval(
    x,
    errors: str = 'raise',
    replace_zero_with: float = None,
    p_min: float = None
)
```

Transform p-values to Log10. 

Paramters:   x: input.  errors (str): Defaults to 'raise' else replace (in case of visualization only).  p_min (float): Replace zeros with this value. Note: to be used for visualization only.   



**Returns:**
  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_q`

```python
get_q(ds1: Series, col: str = None, verb: bool = True, test_coff: float = 0.1)
```

To FDR corrected P-value. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `glog`

```python
glog(x: float, l=2)
```

Generalised logarithm. 



**Args:**
 
 - <b>`x`</b> (float):  input. 
 - <b>`l`</b> (int, optional):  psudo-count. Defaults to 2. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rescale`

```python
rescale(
    a: <built-in function array>,
    range1: tuple = None,
    range2: tuple = [0, 1]
) → <built-in function array>
```

Rescale within a new range. 



**Args:**
 
 - <b>`a`</b> (np.array):  input vector. 
 - <b>`range1`</b> (tuple, optional):  existing range. Defaults to None. 
 - <b>`range2`</b> (tuple, optional):  new range. Defaults to [0,1]. 



**Returns:**
 
 - <b>`np.array`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/transform.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rescale_divergent`

```python
rescale_divergent(df1: DataFrame, col: str) → DataFrame
```

Rescale divergently i.e. two-sided. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  _description_ 
 - <b>`col`</b> (str):  column. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  column. 



**Notes:**

> Under development. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.variance`
For variance related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/variance.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `confidence_interval_95`

```python
confidence_interval_95(x: <built-in function array>) → float
```

95% confidence interval. 



**Args:**
 
 - <b>`x`</b> (np.array):  input vector. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/stat/variance.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ci`

```python
get_ci(rs, ci_type, outstr=False)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.annot`
For annotations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_label`

```python
set_label(
    s: str,
    ax: Axes,
    x: float = 0,
    y: float = 0,
    ha: str = 'left',
    va: str = 'top',
    loc=None,
    off_loc=0.01,
    title: bool = False,
    **kws
) → Axes
```

Set label on a plot. 



**Args:**
 
 - <b>`x`</b> (float):  x position. 
 - <b>`y`</b> (float):  y position. 
 - <b>`s`</b> (str):  label. 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`ha`</b> (str, optional):  horizontal alignment. Defaults to 'left'. 
 - <b>`va`</b> (str, optional):  vertical alignment. Defaults to 'top'. 
 - <b>`loc`</b> (int, optional):  location of the label. 1:'upper right', 2:'upper left', 3:'lower left':3, 4:'lower right' 
 - <b>`offs_loc`</b> (tuple,optional):  x and y location offsets. 
 - <b>`title`</b> (bool, optional):  set as title. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_suptitle`

```python
set_suptitle(axs, title, offy=0)
```

Combined title for a list of subplots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_side`

```python
annot_side(
    ax: Axes,
    df1: DataFrame,
    colx: str,
    coly: str,
    cols: str = None,
    hue: str = None,
    loc: str = 'right',
    scatter=False,
    lines=True,
    text=True,
    invert_xaxis: bool = False,
    offx3: float = 0.15,
    offymin: float = 0.1,
    offymax: float = 0.9,
    offx_text: float = 0,
    offy_text: float = 0,
    break_pt: int = 25,
    length_axhline: float = 3,
    va: str = 'bottom',
    zorder: int = 1,
    color: str = 'gray',
    kws_line: dict = {},
    kws_scatter: dict = {'zorder': 2, 'alpha': 0.75, 'marker': '|', 's': 100},
    **kws_text
) → Axes
```

Annot elements of the plots on the of the side plot. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data 
 - <b>`colx`</b> (str):  column with x values. 
 - <b>`coly`</b> (str):  column with y values. 
 - <b>`cols`</b> (str):  column with labels. 
 - <b>`hue`</b> (str):  column with colors of the labels. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`loc`</b> (str, optional):  location. Defaults to 'right'. 
 - <b>`invert_xaxis`</b> (bool, optional):  invert xaxis. Defaults to False. 
 - <b>`offx3`</b> (float, optional):  x-offset for bend position of the arrow. Defaults to 0.15. 
 - <b>`offymin`</b> (float, optional):  x-offset minimum. Defaults to 0.1. 
 - <b>`offymax`</b> (float, optional):  x-offset maximum. Defaults to 0.9. 
 - <b>`break_pt`</b> (int, optional):  break point of the labels. Defaults to 25. 
 - <b>`length_axhline`</b> (float, optional):  length of the horizontal line i.e. the "underline". Defaults to 3. 
 - <b>`zorder`</b> (int, optional):  z-order. Defaults to 1. 
 - <b>`color`</b> (str, optional):  color of the line. Defaults to 'gray'. 
 - <b>`kws_line`</b> (dict, optional):  parameters for formatting the line. Defaults to {}. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `ax.text` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_corners`

```python
annot_corners(
    ax: Axes,
    df1: DataFrame,
    colx: str,
    coly: str,
    coltext: str,
    off: float = 0.1,
    **kws
) → Axes
```

Annotate points above and below the diagonal. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `confidence_ellipse`

```python
confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs)
```

Create a plot of the covariance confidence ellipse of *x* and *y*. 



**Parameters:**
 
---------- x, y : array-like, shape (n, )  Input data. 

ax : matplotlib.axes.Axes  The axes object to draw the ellipse into. 

n_std : float  The number of standard deviations to determine the ellipse's radiuses. 

**kwargs  Forwarded to `~matplotlib.patches.Ellipse` 

Returns 
------- matplotlib.patches.Ellipse 

References 
---------- https://matplotlib.org/3.5.0/gallery/statistics/confidence_ellipse.html 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_box`

```python
show_box(
    ax: Axes,
    xy: tuple,
    width: float,
    height: float,
    fill: str = None,
    alpha: float = 1,
    lw: float = 1.1,
    ec: str = 'k',
    clip_on: bool = False,
    scale_width: float = 1,
    scale_height: float = 1,
    xoff: float = 0,
    yoff: float = 0,
    **kws
) → Axes
```

Highlight sections of a plot e.g. heatmap by drawing boxes. 



**Args:**
 
 - <b>`xy`</b> (tuple):  position of left, bottom corner of the box. 
 - <b>`width`</b> (float):  width. 
 - <b>`height`</b> (float):  height. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`fill`</b> (str, optional):  fill the box with color. Defaults to None. 
 - <b>`alpha`</b> (float, optional):  alpha of color. Defaults to 1. 
 - <b>`lw`</b> (float, optional):  line width. Defaults to 1.1. 
 - <b>`ec`</b> (str, optional):  edge color. Defaults to 'k'. 
 - <b>`clip_on`</b> (bool, optional):  clip the boxes by the axis limit. Defaults to False. 
 - <b>`scale_width`</b> (float, optional):  scale width. Defaults to 1. 
 - <b>`scale_height`</b> (float, optional):  scale height. Defaults to 1. 
 - <b>`xoff`</b> (float, optional):  x-offset. Defaults to 0. 
 - <b>`yoff`</b> (float, optional):  y-offset. Defaults to 0. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `Rectangle` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_confusion_matrix`

```python
annot_confusion_matrix(df_: DataFrame, ax: Axes = None, off: float = 0.5) → Axes
```

Annotate a confusion matrix. 



**Args:**
 
 - <b>`df_`</b> (pd.DataFrame):  input data. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`off`</b> (float, optional):  offset. Defaults to 0.5. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_logo_ax`

```python
get_logo_ax(
    ax: Axes,
    size: float = 0.5,
    bbox_to_anchor: list = None,
    loc: str = 1,
    axes_kwargs: dict = {'zorder': -1}
) → Axes
```

Get `plt.Axes` for placing the logo. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`size`</b> (float, optional):  size of the subplot. Defaults to 0.5. 
 - <b>`bbox_to_anchor`</b> (list, optional):  location. Defaults to None. 
 - <b>`loc`</b> (str, optional):  location. Defaults to 1. 
 - <b>`axes_kwargs`</b> (_type_, optional):  parameters provided to `inset_axes`. Defaults to {'zorder':-1}. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_logo`

```python
set_logo(
    imp: str,
    ax: Axes,
    size: float = 0.5,
    bbox_to_anchor: list = None,
    loc: str = 1,
    axes_kwargs: dict = {'zorder': -1},
    params_imshow: dict = {'aspect': 'auto', 'alpha': 1, 'interpolation': 'catrom'},
    test: bool = False,
    force: bool = False
) → Axes
```

Set logo. 



**Args:**
 
 - <b>`imp`</b> (str):  path to the logo file. 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`size`</b> (float, optional):  size of the subplot. Defaults to 0.5. 
 - <b>`bbox_to_anchor`</b> (list, optional):  location. Defaults to None. 
 - <b>`loc`</b> (str, optional):  location. Defaults to 1. 
 - <b>`axes_kwargs`</b> (_type_, optional):  parameters provided to `inset_axes`. Defaults to {'zorder':-1}. 
 - <b>`params_imshow`</b> (_type_, optional):  parameters provided to the `imshow` function. Defaults to {'aspect':'auto','alpha':1, 'interpolation':'catrom'}. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`force`</b> (bool, optional):  overwrite file. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `color_ax`

```python
color_ax(ax: Axes, c: str, linewidth: float = None) → Axes
```

Color border of `plt.Axes`. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`c`</b> (str):  color. 
 - <b>`linewidth`</b> (float, optional):  line width. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/annot.py#L531"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_n_legend`

```python
annot_n_legend(ax, df1: DataFrame, colid: str, colgroup: str, **kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.ax_`
For setting up subplots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_`

```python
set_(ax: Axes, test: bool = False, **kws) → Axes
```

Ser many axis parameters. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `ax.set` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_axes_minimal`

```python
set_axes_minimal(ax, xlabel=None, ylabel=None, off_axes_pad=0) → Axes
```

Set minimal axes labels, at the lower left corner. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_ylabel`

```python
set_ylabel(
    ax: Axes,
    s: str = None,
    x: float = -0.1,
    y: float = 1.02,
    xoff: float = 0,
    yoff: float = 0
) → Axes
```

Set ylabel horizontal. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`s`</b> (str, optional):  ylabel. Defaults to None. 
 - <b>`x`</b> (float, optional):  x position. Defaults to -0.1. 
 - <b>`y`</b> (float, optional):  y position. Defaults to 1.02. 
 - <b>`xoff`</b> (float, optional):  x offset. Defaults to 0. 
 - <b>`yoff`</b> (float, optional):  y offset. Defaults to 0. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rename_labels`

```python
rename_labels(ax, d1)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_labels`

```python
format_labels(ax, fmt='cap1', title_fontsize=15, test=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rename_ticklabels`

```python
rename_ticklabels(
    ax: Axes,
    axis: str,
    rename: dict = None,
    replace: dict = None,
    ignore: bool = False
) → Axes
```

Rename the ticklabels. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`axis`</b> (str):  axis (x|y). 
 - <b>`rename`</b> (dict, optional):  replace strings. Defaults to None. 
 - <b>`replace`</b> (dict, optional):  replace sub-strings. Defaults to None. 
 - <b>`ignore`</b> (bool, optional):  ignore warnings. Defaults to False. 



**Raises:**
 
 - <b>`ValueError`</b>:  either `rename` or `replace` should be provided. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ticklabel_position`

```python
get_ticklabel_position(ax: Axes, axis: str) → Axes
```

Get positions of the ticklabels. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`axis`</b> (str):  axis (x|y). 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ticklabel_position`

```python
get_ticklabel_position(ax: Axes, axis: str) → Axes
```

Get positions of the ticklabels. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`axis`</b> (str):  axis (x|y). 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_ticklabels_color`

```python
set_ticklabels_color(ax: Axes, ticklabel2color: dict, axis: str = 'y') → Axes
```

Set colors to ticklabels. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`ticklabel2color`</b> (dict):  colors of the ticklabels. 
 - <b>`axis`</b> (str):  axis (x|y). 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_ticklabels_color`

```python
set_ticklabels_color(ax: Axes, ticklabel2color: dict, axis: str = 'y') → Axes
```

Set colors to ticklabels. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`ticklabel2color`</b> (dict):  colors of the ticklabels. 
 - <b>`axis`</b> (str):  axis (x|y). 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_ticklabels`

```python
format_ticklabels(
    ax: Axes,
    axes: tuple = ['x', 'y'],
    n: int = None,
    fmt: str = None,
    font: str = None
) → Axes
```

format_ticklabels 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`axes`</b> (tuple, optional):  axes. Defaults to ['x','y']. 
 - <b>`n`</b> (int, optional):  number of ticks. Defaults to None. 
 - <b>`fmt`</b> (str, optional):  format. Defaults to None. 
 - <b>`font`</b> (str, optional):  font. Defaults to 'DejaVu Sans Mono'. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs:  1. include color_ticklabels 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_equallim`

```python
set_equallim(
    ax: Axes,
    diagonal: bool = False,
    difference: float = None,
    format_ticks: bool = True,
    **kws_format_ticklabels
) → Axes
```

Set equal axis limits. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`diagonal`</b> (bool, optional):  show diagonal. Defaults to False. 
 - <b>`difference`</b> (float, optional):  difference from . Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_axlims`

```python
get_axlims(ax: Axes) → Axes
```

Get axis limits. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_axlims`

```python
set_axlims(ax: Axes, off: float, axes: list = ['x', 'y']) → Axes
```

Set axis limits. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`off`</b> (float):  offset. 
 - <b>`axes`</b> (list, optional):  axis name/s. Defaults to ['x','y']. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_axlimsby_data`

```python
get_axlimsby_data(
    X: Series,
    Y: Series,
    off: float = 0.2,
    equal: bool = False
) → Axes
```

Infer axis limits from data. 



**Args:**
 
 - <b>`X`</b> (pd.Series):  x values. 
 - <b>`Y`</b> (pd.Series):  y values. 
 - <b>`off`</b> (float, optional):  offsets. Defaults to 0.2. 
 - <b>`equal`</b> (bool, optional):  equal limits. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `split_ticklabels`

```python
split_ticklabels(
    ax: Axes,
    axis='x',
    fmt: str = 'paired',
    group_x=0.01,
    group_prefix=None,
    group_loc='left',
    group_colors=None,
    group_alpha=0.2,
    show_group_line=True,
    show_group_span=True,
    sep: str = '-',
    pad_major=6,
    off: float = 0.2,
    **kws
) → Axes
```

Split ticklabels into major and minor. Two minor ticks are created per major tick.  



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`sep`</b> (str, optional):  separator within the tick labels. Defaults to ' '. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_grids`

```python
set_grids(ax: Axes, axis: str = None) → Axes
```

Show grids. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`axis`</b> (str, optional):  axis name. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L491"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rename_legends`

```python
rename_legends(ax: Axes, replaces: dict, **kws_legend) → Axes
```

Rename legends. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`replaces`</b> (dict):  _description_ 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `append_legends`

```python
append_legends(ax: Axes, labels: list, handles: list, **kws) → Axes
```

Append to legends. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`labels`</b> (list):  labels.  
 - <b>`handles`</b> (list):  handles. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L540"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_legends`

```python
sort_legends(ax: Axes, sort_order: list = None, **kws) → Axes
```

Sort or filter legends. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`sort_order`</b> (list, optional):  order of legends. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 



**Notes:**

> 1. Filter the legends by providing the indices of the legends to keep. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L570"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_duplicate_legend`

```python
drop_duplicate_legend(ax, **kws)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `reset_legend_colors`

```python
reset_legend_colors(ax)
```

Reset legend colors. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L587"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_legends_merged`

```python
set_legends_merged(axs)
```

Reset legend colors. 



**Args:**
 
 - <b>`axs`</b> (list):  list of `plt.Axes` objects. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  first `plt.Axes` object in the list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_legend_custom`

```python
set_legend_custom(
    ax: Axes,
    legend2param: dict,
    param: str = 'color',
    lw: float = 1,
    marker: str = 'o',
    markerfacecolor: bool = True,
    size: float = 10,
    color: str = 'k',
    linestyle: str = '',
    title_ha: str = 'center',
    frameon: bool = True,
    **kws
) → Axes
```

Set custom legends. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`legend2param`</b> (dict):  legend name to parameter to change e.g. name of the color.  
 - <b>`param`</b> (str, optional):  parameter to change. Defaults to 'color'. 
 - <b>`lw`</b> (float, optional):  line width. Defaults to 1. 
 - <b>`marker`</b> (str, optional):  marker type. Defaults to 'o'. 
 - <b>`markerfacecolor`</b> (bool, optional):  marker face color. Defaults to True. 
 - <b>`size`</b> (float, optional):  size of the markers. Defaults to 10. 
 - <b>`color`</b> (str, optional):  color of the markers. Defaults to 'k'. 
 - <b>`linestyle`</b> (str, optional):  line style. Defaults to ''. 
 - <b>`title_ha`</b> (str, optional):  title horizontal alignment. Defaults to 'center'. 
 - <b>`frameon`</b> (bool, optional):  show frame. Defaults to True. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: 1. differnet number of points for eachh entry 

 from matplotlib.legend_handler import HandlerTuple  l1, = plt.plot(-1, -1, lw=0, marker="o",  markerfacecolor='k', markeredgecolor='k')  l2, = plt.plot(-0.5, -1, lw=0, marker="o",  markerfacecolor="none", markeredgecolor='k')  plt.legend([(l1,), (l1, l2)], ["test 1", "test 2"],  
 - <b>`handler_map={tuple`</b>:  HandlerTuple(2)} ) 

References:  
 - <b>`https`</b>: //matplotlib.org/stable/api/markers_api.html 
 - <b>`http`</b>: //www.cis.jhu.edu/~shanest/mpt/js/mathjax/mathjax-dev/fonts/Tables/STIX/STIX/All/All.html 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_line_cap_length`

```python
get_line_cap_length(ax: Axes, linewidth: float) → Axes
```

Get the line cap length. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object 
 - <b>`linewidth`</b> (float):  width of the line. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L692"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_subplot_dimentions`

```python
get_subplot_dimentions(ax=None)
```

Calculate the aspect ratio of `plt.Axes`. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object 

References:  
 - <b>`https`</b>: //github.com/matplotlib/matplotlib/issues.py#issuecomment-285472404     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_colorbar`

```python
set_colorbar(
    fig: object,
    ax: Axes,
    ax_pc: Axes,
    label: str,
    bbox_to_anchor: tuple = (0.05, 0.5, 1, 0.45),
    orientation: str = 'vertical'
)
```

Set colorbar. 



**Args:**
 
 - <b>`fig`</b> (object):  figure object. 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`ax_pc`</b> (plt.Axes):  `plt.Axes` object for the colorbar. 
 - <b>`label`</b> (str):  label 
 - <b>`bbox_to_anchor`</b> (tuple, optional):  location. Defaults to (0.05, 0.5, 1, 0.45). 
 - <b>`orientation`</b> (str, optional):  orientation. Defaults to "vertical". 



**Returns:**
 figure object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/ax_.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_colorbar_label`

```python
set_colorbar_label(ax: Axes, label: str) → Axes
```

Find colorbar and set label for it. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`label`</b> (str):  label. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.bar`
For bar plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_barh`

```python
plot_barh(
    df1: DataFrame,
    colx: str,
    coly: str,
    colannnotside: str = None,
    x1: float = None,
    offx: float = 0,
    ax: Axes = None,
    **kws
) → Axes
```

Plot horizontal bar plot with text on them. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`colx`</b> (str):  x column. 
 - <b>`coly`</b> (str):  y column. 
 - <b>`colannnotside`</b> (str):  column with annotations to show on the right side of the plot. 
 - <b>`x1`</b> (float):  x position of the text. 
 - <b>`offx`</b> (float):  x-offset of x1, multiplier. 
 - <b>`color`</b> (str):  color of the bars. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `barh` function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_value_counts`

```python
plot_value_counts(
    df: DataFrame,
    col: str,
    logx: bool = False,
    kws_hist: dict = {'bins': 10},
    kws_bar: dict = {},
    grid: bool = False,
    axes: list = None,
    fig: object = None,
    hist: bool = True
)
```

Plot pandas's `value_counts`.  



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input data `value_counts`. 
 - <b>`col`</b> (str):  column with counts. 
 - <b>`logx`</b> (bool, optional):  x-axis on log-scale. Defaults to False. 
 - <b>`kws_hist`</b> (_type_, optional):  parameters provided to the `hist` function. Defaults to {'bins':10}. 
 - <b>`kws_bar`</b> (dict, optional):  parameters provided to the `bar` function. Defaults to {}. 
 - <b>`grid`</b> (bool, optional):  show grids or not. Defaults to False. 
 - <b>`axes`</b> (list, optional):  list of `plt.axes`. Defaults to None. 
 - <b>`fig`</b> (object, optional):  figure object. Defaults to None. 
 - <b>`hist`</b> (bool, optional):  show histgram. Defaults to True. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_barh_stacked_percentage`

```python
plot_barh_stacked_percentage(
    df1: DataFrame,
    coly: str,
    colannot: str,
    color: str = None,
    yoff: float = 0,
    ax: Axes = None
) → Axes
```

Plot horizontal stacked bar plot with percentages. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. values in rows sum to 100%. 
 - <b>`coly`</b> (str):  y column. yticklabels, e.g. retained and dropped. 
 - <b>`colannot`</b> (str):  column with annotations. 
 - <b>`color`</b> (str, optional):  color. Defaults to None. 
 - <b>`yoff`</b> (float, optional):  y-offset. Defaults to 0. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_bar_serial`

```python
plot_bar_serial(
    d1: dict,
    polygon: bool = False,
    polygon_x2i: float = 0,
    labelis: list = [],
    y: float = 0,
    ylabel: str = None,
    off_arrowy: float = 0.15,
    kws_rectangle={'height': 0.5, 'linewidth': 1},
    ax: Axes = None
) → Axes
```

Barplots with serial increase in resolution. 



**Args:**
 
 - <b>`d1`</b> (dict):  dictionary with the data. 
 - <b>`polygon`</b> (bool, optional):  show polygon. Defaults to False. 
 - <b>`polygon_x2i`</b> (float, optional):  connect polygon to this subset. Defaults to 0. 
 - <b>`labelis`</b> (list, optional):  label these subsets. Defaults to []. 
 - <b>`y`</b> (float, optional):  y position. Defaults to 0. 
 - <b>`ylabel`</b> (str, optional):  y label. Defaults to None. 
 - <b>`off_arrowy`</b> (float, optional):  offset for the arrow. Defaults to 0.15. 
 - <b>`kws_rectangle`</b> (_type_, optional):  parameters provided to the `rectangle` function. Defaults to dict(height=0.5,linewidth=1). 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_barh_stacked_percentage_intersections`

```python
plot_barh_stacked_percentage_intersections(
    df0: DataFrame,
    colxbool: str,
    colybool: str,
    colvalue: str,
    colid: str,
    colalt: str,
    colgroupby: str,
    coffgroup: float = 0.95,
    ax: Axes = None
) → Axes
```

Plot horizontal stacked bar plot with percentages and intesections. 



**Args:**
 
 - <b>`df0`</b> (pd.DataFrame):  input data. 
 - <b>`colxbool`</b> (str):  x column. 
 - <b>`colybool`</b> (str):  y column. 
 - <b>`colvalue`</b> (str):  column with the values. 
 - <b>`colid`</b> (str):  column with ids. 
 - <b>`colalt`</b> (str):  column with the alternative subset. 
 - <b>`colgroupby`</b> (str):  column with groups. 
 - <b>`coffgroup`</b> (float, optional):  cut-off between the groups. Defaults to 0.95. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 



**Examples:**
 

**Parameters:**
  colxbool='paralog',  colybool='essential',  colvalue='value',  colid='gene id',  colalt='singleton',  coffgroup=0.95,  colgroupby='tissue', 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_input_data_sankey`

```python
to_input_data_sankey(
    df0,
    colid,
    cols_groupby=None,
    colall='all',
    remove_all=False
)
```

 




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/bar.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_sankey`

```python
plot_sankey(
    df1,
    cols_groupby=None,
    hues=None,
    node_color=None,
    link_color=None,
    info=None,
    x=None,
    y=None,
    colors=None,
    hovertemplate=None,
    text_width=20,
    convert=True,
    width=400,
    height=400,
    outp=None,
    validate=True,
    test=False,
    **kws
)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.colors`
For setting up colors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rgbfloat2int`

```python
rgbfloat2int(rgb_float)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_colors_default`

```python
get_colors_default() → list
```

get default colors. 



**Returns:**
 
 - <b>`list`</b>:  colors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ncolors`

```python
get_ncolors(
    n: int,
    cmap: str = 'Spectral',
    ceil: bool = False,
    test: bool = False,
    N: int = 20,
    out: str = 'hex',
    **kws_get_cmap_section
) → list
```

Get colors. 



**Args:**
 
 - <b>`n`</b> (int):  number of colors to get. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Spectral'. 
 - <b>`ceil`</b> (bool, optional):  ceil. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`N`</b> (int, optional):  number of colors in the colormap. Defaults to 20. 
 - <b>`out`</b> (str, optional):  output. Defaults to 'hex'. 



**Returns:**
 
 - <b>`list`</b>:  colors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_val2color`

```python
get_val2color(
    ds: Series,
    vmin: float = None,
    vmax: float = None,
    cmap: str = 'Reds'
) → dict
```

Get color for a value. 



**Args:**
 
 - <b>`ds`</b> (pd.Series):  values. 
 - <b>`vmin`</b> (float, optional):  minimum value. Defaults to None. 
 - <b>`vmax`</b> (float, optional):  maximum value. Defaults to None. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `saturate_color`

```python
saturate_color(color, alpha: float) → object
```

Saturate a color. 



**Args:**
  color (_type_): 
 - <b>`alpha`</b> (float):  alpha level. 



**Returns:**
 
 - <b>`object`</b>:  output. 

References:  
 - <b>`https`</b>: //stackoverflow.com/a/60562502/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `mix_colors`

```python
mix_colors(d: dict) → str
```

Mix colors. 



**Args:**
 
 - <b>`d`</b> (dict):  colors to alpha map. 



**Returns:**
 
 - <b>`str`</b>:  hex color. 

References:  
 - <b>`https`</b>: //stackoverflow.com/a/61488997/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_cmap`

```python
make_cmap(cs: list, N: int = 20, **kws)
```

Create a colormap. 



**Args:**
 
 - <b>`cs`</b> (list):  colors 
 - <b>`N`</b> (int, optional):  resolution i.e. number of colors. Defaults to 20. 



**Returns:**
 cmap. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cmap_section`

```python
get_cmap_section(
    cmap,
    vmin: float = 0.0,
    vmax: float = 1.0,
    n: int = 100
) → object
```

Get section of a colormap. 



**Args:**
 
 - <b>`cmap`</b> (object| str):  colormap. 
 - <b>`vmin`</b> (float, optional):  minimum value. Defaults to 0.0. 
 - <b>`vmax`</b> (float, optional):  maximum value. Defaults to 1.0. 
 - <b>`n`</b> (int, optional):  resolution i.e. number of colors. Defaults to 100. 



**Returns:**
 
 - <b>`object`</b>:  cmap. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/colors.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `append_cmap`

```python
append_cmap(
    cmap: str = 'Reds',
    color: str = '#D3DDDC',
    cmap_min: float = 0.2,
    cmap_max: float = 0.8,
    ncolors: int = 100,
    ncolors_min: int = 1,
    ncolors_max: int = 0
)
```

Append a color to colormap. 



**Args:**
 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 
 - <b>`color`<.py#D3DDDC'. 
 - <b>`cmap_min`</b> (float, optional):  cmap_min. Defaults to 0.2. 
 - <b>`cmap_max`</b> (float, optional):  cmap_max. Defaults to 0.8. 
 - <b>`ncolors`</b> (int, optional):  number of colors. Defaults to 100. 
 - <b>`ncolors_min`</b> (int, optional):  number of colors minimum. Defaults to 1. 
 - <b>`ncolors_max`</b> (int, optional):  number of colors maximum. Defaults to 0. 



**Returns:**
 cmap. 

References:  
 - <b>`https`</b>: //matplotlib.org/stable/tutorials/colors/colormap-manipulation.html 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.compare`
For comparative plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/compare.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_comparisons`

```python
plot_comparisons(
    plot_data,
    x,
    ax=None,
    output_dir_path=None,
    force=False,
    return_path=False
)
```



**Parameters:**
 
 - <b>`plot_data`</b>:  output of `.stat.compare.get_comparison` 

**Notes:**

> `sample type`: different sample of the same data. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.dist`
For distribution plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `hist_annot`

```python
hist_annot(
    dplot: DataFrame,
    colx: str,
    colssubsets: list = [],
    bins: int = 100,
    subset_unclassified: bool = True,
    cmap: str = 'hsv',
    ymin=None,
    ymax=None,
    ylimoff: float = 1,
    ywithinoff: float = 1.2,
    annotaslegend: bool = True,
    annotn: bool = True,
    params_scatter: dict = {'zorder': 2, 'alpha': 0.1, 'marker': '|'},
    xlim: tuple = None,
    ax: Axes = None,
    **kws
) → Axes
```

Annoted histogram. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input dataframe. 
 - <b>`colx`</b> (str):  x column. 
 - <b>`colssubsets`</b> (list, optional):  columns indicating subsets. Defaults to []. 
 - <b>`bins`</b> (int, optional):  bins. Defaults to 100. 
 - <b>`subset_unclassified`</b> (bool, optional):  call non-annotated subset as 'unclassified'. Defaults to True. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds_r'. 
 - <b>`ylimoff`</b> (float, optional):  y-offset for y-axis limit . Defaults to 1.2. 
 - <b>`ywithinoff`</b> (float, optional):  y-offset for the distance within labels. Defaults to 1.2. 
 - <b>`annotaslegend`</b> (bool, optional):  convert labels to legends. Defaults to True. 
 - <b>`annotn`</b> (bool, optional):  annotate sample sizes. Defaults to True. 
 - <b>`params_scatter`</b> (_type_, optional):  parameters of the scatter plot. Defaults to {'zorder':2,'alpha':0.1,'marker':'|'}. 
 - <b>`xlim`</b> (tuple, optional):  x-axis limits. Defaults to None. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `hist` function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: For scatter, use `annot_side` with `loc='top'`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_gmm`

```python
plot_gmm(
    x: Series,
    coff: float = None,
    mix_pdf: object = None,
    two_pdfs: tuple = None,
    weights: tuple = None,
    n_clusters: int = 2,
    bins: int = 20,
    show_cutoff: bool = True,
    show_cutoff_line: bool = True,
    colors: list = ['gray', 'gray', 'lightgray'],
    out_coff: bool = False,
    hist: bool = True,
    test: bool = False,
    ax: Axes = None,
    kws_axvline={'color': 'k'},
    **kws
) → Axes
```

Plot Gaussian mixture Models (GMMs). 



**Args:**
 
 - <b>`x`</b> (pd.Series):  input vector. 
 - <b>`coff`</b> (float, optional):  intersection between two fitted distributions. Defaults to None. 
 - <b>`mix_pdf`</b> (object, optional):  Probability density function of the mixed distribution. Defaults to None. 
 - <b>`two_pdfs`</b> (tuple, optional):  Probability density functions of the separate distributions. Defaults to None. 
 - <b>`weights`</b> (tuple, optional):  weights of the individual distributions. Defaults to None. 
 - <b>`n_clusters`</b> (int, optional):  number of distributions. Defaults to 2. 
 - <b>`bins`</b> (int, optional):  bins. Defaults to 50. 
 - <b>`colors`</b> (list, optional):  colors of the invividual distributions and of the mixed one. Defaults to ['gray','gray','lightgray']. 'gray' 
 - <b>`out_coff`</b> (bool,False):  return the cutoff. Defaults to False. 
 - <b>`hist`</b> (bool, optional):  show histogram. Defaults to True. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `hist` function. 
 - <b>`kws_axvline`</b>:  parameters provided to the `axvline` function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_normal`

```python
plot_normal(x: Series, ax: Axes = None) → Axes
```

Plot normal distribution. 



**Args:**
 
 - <b>`x`</b> (pd.Series):  input vector. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_jitter_positions`

```python
get_jitter_positions(ax, df1, order, column_category, column_position)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_dists`

```python
plot_dists(
    df1: DataFrame,
    x: str,
    y: str,
    colindex: str,
    hue: str = None,
    order: list = None,
    hue_order: list = None,
    kind: str = 'box',
    show_p: bool = True,
    show_n: bool = True,
    show_n_prefix: str = '',
    show_n_ha=None,
    show_n_ticklabels: bool = True,
    show_outlines: bool = False,
    kws_outlines: dict = {},
    alternative: str = 'two-sided',
    offx_n: float = 0,
    axis_cont_lim: tuple = None,
    axis_cont_scale: str = 'linear',
    offx_pval: float = 0.05,
    offy_pval: float = None,
    alpha: float = 0.5,
    ax: Axes = None,
    test: bool = False,
    kws_stats: dict = {},
    **kws
) → Axes
```

Plot distributions. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`x`</b> (str):  x column. 
 - <b>`y`</b> (str):  y column. 
 - <b>`colindex`</b> (str):  index column. 
 - <b>`hue`</b> (str, optional):  column with values to be encoded as hues. Defaults to None. 
 - <b>`order`</b> (list, optional):  order of categorical values. Defaults to None. 
 - <b>`hue_order`</b> (list, optional):  order of values to be encoded as hues. Defaults to None. 
 - <b>`kind`</b> (str, optional):  kind of distribution. Defaults to 'box'. 
 - <b>`show_p`</b> (bool, optional):  show p-values. Defaults to True. 
 - <b>`show_n`</b> (bool, optional):  show sample sizes. Defaults to True. 
 - <b>`show_n_prefix`</b> (str, optional):  show prefix of sample size label i.e. `n=`. Defaults to ''. 
 - <b>`offx_n`</b> (float, optional):  x-offset for the sample size label. Defaults to 0. 
 - <b>`axis_cont_lim`</b> (tuple, optional):  x-axis limits. Defaults to None. 
 - <b>`offx_pval`</b> (float, optional):  x-offset for the p-value labels. Defaults to 0.05. 
 - <b>`offy_pval`</b> (float, optional):  y-offset for the p-value labels. Defaults to None. 
 - <b>`# saturate_color_alpha (float, optional)`</b>:  saturation of the color. Defaults to 1.5. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`kws_stats`</b> (dict, optional):  parameters provided to the stat function. Defaults to {}. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `seaborn` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: 1. Sort categories. 2. Change alpha of the boxplot rather than changing saturation of the swarmplot.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/dist.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `pointplot_groupbyedgecolor`

```python
pointplot_groupbyedgecolor(data: DataFrame, ax: Axes = None, **kws) → Axes
```

Plot seaborn's `pointplot` grouped by edgecolor of points. 



**Args:**
 
 - <b>`data`</b> (pd.DataFrame):  input data. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `seaborn`'s `pointplot` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.figure`
For setting up figures. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/figure.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_subplots`

```python
get_subplots(nrows: int, ncols: int, total: int = None) → list
```

Get subplots. 



**Args:**
 
 - <b>`nrows`</b> (int):  number of rows. 
 - <b>`ncols`</b> (int):  number of columns. 
 - <b>`total`</b> (int, optional):  total subplots. Defaults to None. 



**Returns:**
 
 - <b>`list`</b>:  list of `plt.Axes` objects. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/figure.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `labelplots`

```python
labelplots(
    fig,
    axes: list = None,
    labels: list = None,
    xoff: float = 0,
    yoff: float = 0,
    custom_positions: dict = {},
    size: float = 20,
    va: str = 'bottom',
    ha: str = 'right',
    test: bool = False,
    **kws_text
)
```

Label (sub)plots. 



**Args:**
 
 - <b>`fig `</b>:  `plt.figure` object.  
 - <b>`axes`</b> (_type_):  list of `plt.Axes` objects. 
 - <b>`xoff`</b> (int, optional):  x offset. Defaults to 0. 
 - <b>`yoff`</b> (int, optional):  y offset. Defaults to 0. 
 - <b>`params_alignment`</b> (dict, optional):  alignment parameters. Defaults to {}. 
 - <b>`params_text`</b> (dict, optional):  parameters provided to `plt.text`. Defaults to {'size':20,'va':'bottom', 'ha':'right' }. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.heatmap`
For heatmaps. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/heatmap.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_table`

```python
plot_table(
    df1: DataFrame,
    xlabel: str = None,
    ylabel: str = None,
    annot: bool = True,
    cbar: bool = False,
    linecolor: str = 'k',
    linewidths: float = 1,
    cmap: str = None,
    sorty: bool = False,
    linebreaky: bool = False,
    scales: tuple = [1, 1],
    ax: Axes = None,
    **kws
) → Axes
```

Plot to show a table. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`xlabel`</b> (str, optional):  x label. Defaults to None. 
 - <b>`ylabel`</b> (str, optional):  y label. Defaults to None. 
 - <b>`annot`</b> (bool, optional):  show numbers. Defaults to True. 
 - <b>`cbar`</b> (bool, optional):  show colorbar. Defaults to False. 
 - <b>`linecolor`</b> (str, optional):  line color. Defaults to 'k'. 
 - <b>`linewidths`</b> (float, optional):  line widths. Defaults to 1. 
 - <b>`cmap`</b> (str, optional):  color map. Defaults to None. 
 - <b>`sorty`</b> (bool, optional):  sort rows. Defaults to False. 
 - <b>`linebreaky`</b> (bool, optional):  linebreak for y labels. Defaults to False. 
 - <b>`scales`</b> (tuple, optional):  scale of the table. Defaults to [1,1]. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `sns.heatmap` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/heatmap.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_crosstab`

```python
plot_crosstab(
    df1: DataFrame,
    cols: list = None,
    alpha: float = 0.05,
    method: str = None,
    confusion: bool = False,
    rename_cols: bool = False,
    sort_cols: tuple = [True, True],
    order_x: list = None,
    order_y: list = None,
    annot_pval: str = 'bottom',
    cmap: str = 'Reds',
    ax: Axes = None,
    **kws
) → Axes
```

Plot crosstab table. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data 
 - <b>`cols`</b> (list, optional):  columns. Defaults to None. 
 - <b>`alpha`</b> (float, optional):  alpha for the stats. Defaults to 0.05. 
 - <b>`method`</b> (str, optional):  method to check the association ['chi2','FE']. Defaults to None. 
 - <b>`rename_cols`</b> (bool, optional):  rename the columns. Defaults to True. 
 - <b>`annot_pval`</b> (str, optional):  annotate p-values. Defaults to 'bottom'. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Raises:**
 
 - <b>`ValueError`</b>:  `annot_pval` position should be the allowed one. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: 1. Use `compare_classes` to get the stats. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.image`
For visualization of images. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/image.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_image`

```python
plot_image(
    imp: str,
    ax: Axes = None,
    force=False,
    margin=0,
    axes=False,
    test=False,
    **kwarg
) → Axes
```

Plot image e.g. schematic. 



**Args:**
 
 - <b>`imp`</b> (str):  path of the image. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 
 - <b>`margin`</b> (int, optional):  margins. Defaults to 0. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

:param kwarg: cairosvg: {'dpi':500,'scale':2}; imagemagick: {'trim':False,'alpha':False} 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.io`
For input/output of plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_plotp`

```python
to_plotp(
    ax: Axes = None,
    prefix: str = 'plot/plot_',
    suffix: str = '',
    fmts: list = ['png']
) → str
```

Infer output path for a plot. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`prefix`</b> (str, optional):  prefix with directory path for the plot. Defaults to 'plot/plot_'. 
 - <b>`suffix`</b> (str, optional):  suffix of the filename. Defaults to ''. 
 - <b>`fmts`</b> (list, optional):  formats of the images. Defaults to ['png']. 



**Returns:**
 
 - <b>`str`</b>:  output path for the plot. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `savefig`

```python
savefig(
    plotp: str,
    tight_layout: bool = True,
    bbox_inches: list = None,
    fmts: list = ['png'],
    savepdf: bool = False,
    normalise_path: bool = True,
    replaces_plotp: dict = None,
    dpi: int = 500,
    force: bool = True,
    kws_replace_many: dict = {},
    kws_savefig: dict = {},
    **kws
) → str
```

Wrapper around `plt.savefig`. 



**Args:**
 
 - <b>`plotp`</b> (str):  output path or `plt.Axes` object. 
 - <b>`tight_layout`</b> (bool, optional):  tight_layout. Defaults to True. 
 - <b>`bbox_inches`</b> (list, optional):  bbox_inches. Defaults to None. 
 - <b>`savepdf`</b> (bool, optional):  savepdf. Defaults to False. 
 - <b>`normalise_path`</b> (bool, optional):  normalise_path. Defaults to True. 
 - <b>`replaces_plotp`</b> (dict, optional):  replaces_plotp. Defaults to None. 
 - <b>`dpi`</b> (int, optional):  dpi. Defaults to 500. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to True. 
 - <b>`kws_replace_many`</b> (dict, optional):  parameters provided to the `replace_many` function. Defaults to {}. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to `to_plotp` function. 
 - <b>`kws_savefig`</b>:  parameters provided to `to_savefig` function. 
 - <b>`kws_replace_many`</b>:  parameters provided to `replace_many` function. 



**Returns:**
 
 - <b>`str`</b>:  output path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `savelegend`

```python
savelegend(
    plotp: str,
    legend: object,
    expand: list = [-5, -5, 5, 5],
    **kws_savefig
) → str
```

Save only the legend of the plot/figure. 



**Args:**
 
 - <b>`plotp`</b> (str):  output path. 
 - <b>`legend`</b> (object):  legend object. 
 - <b>`expand`</b> (list, optional):  expand. Defaults to [-5,-5,5,5]. 



**Returns:**
 
 - <b>`str`</b>:  output path. 

References:  
 - <b>`1. https`</b>: //stackoverflow.com/a/47749903/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `update_kws_plot`

```python
update_kws_plot(kws_plot: dict, kws_plotp: dict, test: bool = False) → dict
```

Update the input parameters. 



**Args:**
 
 - <b>`kws_plot`</b> (dict):  input parameters. 
 - <b>`kws_plotp`</b> (dict):  saved parameters. 
 - <b>`test`</b> (bool, optional):  _description_. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  updated parameters. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_plot_inputs`

```python
get_plot_inputs(
    plotp: str,
    df1: DataFrame = None,
    kws_plot: dict = {},
    outd: str = None
) → tuple
```

Get plot inputs. 



**Args:**
 
 - <b>`plotp`</b> (str):  path of the plot. 
 - <b>`df1`</b> (pd.DataFrame):  data for the plot. 
 - <b>`kws_plot`</b> (dict):  parameters of the plot. 
 - <b>`outd`</b> (str):  output directory. 



**Returns:**
 
 - <b>`tuple`</b>:  (path,dataframe,dict) 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code.  




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code.  




---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_lines`

```python
get_lines(
    logp: str = 'log_notebook.log',
    sep: str = 'begin_plot()',
    test: bool = False
) → list
```

Get lines from the log. 



**Args:**
 
 - <b>`logp`</b> (str, optional):  path to the log file. Defaults to 'log_notebook.log'. 
 - <b>`sep`</b> (str, optional):  label marking the start of code of the plot. Defaults to 'begin_plot()'. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  lines of code.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_script`

```python
to_script(
    srcp: str,
    plotp: str,
    defn: str = 'plot_',
    s4: str = '    ',
    test: bool = False,
    **kws
) → str
```

Save the script with the code for the plot. 



**Args:**
 
 - <b>`srcp`</b> (str):  path of the script. 
 - <b>`plotp`</b> (str):  path of the plot. 
 - <b>`defn`</b> (str, optional):  prefix of the function. Defaults to "plot_". 
 - <b>`s4`</b> (str, optional):  a tab. Defaults to '    '. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  path of the script. 

TODOs: 1. Compatible with names of the input dataframes other that `df1`.  1. Get the variable name of the dataframe  

 def get_df_name(df):  name =[x for x in globals() if globals()[x] is df and not x.startswith('-')][0]  return name  

 2. Replace `df1` with the variable name of the dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_plot`

```python
to_plot(
    plotp: str,
    data: DataFrame = None,
    df1: DataFrame = None,
    kws_plot: dict = {},
    logp: str = 'log_notebook.log',
    sep: str = 'begin_plot()',
    validate: bool = False,
    show_path: bool = False,
    show_path_offy: float = -0.2,
    force: bool = True,
    test: bool = False,
    quiet: bool = True,
    **kws
) → str
```

Save a plot. 



**Args:**
 
 - <b>`plotp`</b> (str):  output path. 
 - <b>`df1`</b> (pd.DataFrame, optional):  dataframe with plotting data. Defaults to None. 
 - <b>`data`</b> (pd.DataFrame, optional):  dataframe with plotting data. Defaults to None. 
 - <b>`kws_plot`</b> (dict, optional):  parameters for plotting. Defaults to dict(). 
 - <b>`logp`</b> (str, optional):  path to the log. Defaults to 'log_notebook.log'. 
 - <b>`sep`</b> (str, optional):  separator marking the start of the plotting code in jupyter notebook. Defaults to 'begin_plot()'. 
 - <b>`validate`</b> (bool, optional):  validate the "readability" using `read_plot` function. Defaults to False. 
 - <b>`show_path`</b> (bool, optional):  show path on the plot. Defaults to False. 
 - <b>`show_path_offy`</b> (float, optional):  y-offset for the path label. Defaults to 0. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to True. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`quiet`</b> (bool, optional):  quiet mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  output path. 



**Notes:**

> Requirement: 1. Start logging in the jupyter notebook. 
>from IPython import get_ipython log_notebookp=f'log_notebook.log';open(log_notebookp, 'w').close();get_ipython().run_line_magic('logstart','{log_notebookp} over') 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_plot`

```python
read_plot(p: str, safe: bool = False, test: bool = False, **kws) → Axes
```

Generate the plot from data, parameters and a script. 



**Args:**
 
 - <b>`p`</b> (str):  path of the plot saved using `to_plot` function. 
 - <b>`safe`</b> (bool, optional):  read as an image. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_concat`

```python
to_concat(
    ps: list,
    how: str = 'h',
    use_imagemagick: bool = False,
    use_conda_env: bool = False,
    test: bool = False,
    **kws_outp
) → str
```

Concat images. 



**Args:**
 
 - <b>`ps`</b> (list):  list of paths. 
 - <b>`how`</b> (str, optional):  horizontal (`h`) or vertical `v`. Defaults to 'h'. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  path of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L492"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_montage`

```python
to_montage(
    ps: list,
    layout: str,
    source_path: str = None,
    env_name: str = None,
    hspace: float = 0,
    vspace: float = 0,
    output_path: str = None,
    test: bool = False,
    **kws_outp
) → str
```

To montage. 



**Args:**
 
 - <b>`ps`</b> (_type_):  list of paths. 
 - <b>`layout`</b> (_type_):  layout of the images. 
 - <b>`hspace`</b> (int, optional):  horizontal space. Defaults to 0. 
 - <b>`vspace`</b> (int, optional):  vertical space. Defaults to 0. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  path of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_gif`

```python
to_gif(
    ps: list,
    outp: str,
    duration: int = 200,
    loop: int = 0,
    optimize: bool = True
) → str
```

Convert to GIF. 



**Args:**
 
 - <b>`ps`</b> (list):  list of paths. 
 - <b>`outp`</b> (str):  output path. 
 - <b>`duration`</b> (int, optional):  duration. Defaults to 200. 
 - <b>`loop`</b> (int, optional):  loop or not. Defaults to 0. 
 - <b>`optimize`</b> (bool, optional):  optimize the size. Defaults to True. 



**Returns:**
 
 - <b>`str`</b>:  output path. 

References: 
 - <b>`1. https`</b>: //pillow.readthedocs.io/en/stable/handbook.py#gif 
 - <b>`2. https`</b>: //stackoverflow.com/a/57751793/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L564"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_data`

```python
to_data(path: str) → str
```

Convert to base64 string. 



**Args:**
 
 - <b>`path`</b> (str):  path of the input. 



**Returns:**
 base64 string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L585"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_convert`

```python
to_convert(filep: str, outd: str = None, fmt: str = 'JPEG') → str
```

Convert format of image using `PIL`. 



**Args:**
 
 - <b>`filep`</b> (str):  input path. 
 - <b>`outd`</b> (str, optional):  output directory. Defaults to None. 
 - <b>`fmt`</b> (str, optional):  format of the output. Defaults to "JPEG". 



**Returns:**
 
 - <b>`str`</b>:  output path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L605"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_raster`

```python
to_raster(
    plotp: str,
    dpi: int = 500,
    alpha: bool = False,
    trim: bool = False,
    force: bool = False,
    test: bool = False
) → str
```

to_raster _summary_ 



**Args:**
 
 - <b>`plotp`</b> (str):  input path. 
 - <b>`dpi`</b> (int, optional):  DPI. Defaults to 500. 
 - <b>`alpha`</b> (bool, optional):  transparency. Defaults to False. 
 - <b>`trim`</b> (bool, optional):  trim margins. Defaults to False. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  _description_ 



**Notes:**

> 1. Runs a bash command: `convert -density 300 -trim`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/io.py#L644"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_rasters`

```python
to_rasters(plotd, ext='svg')
```

Convert many images to raster. Uses inkscape. 



**Args:**
 
 - <b>`plotd`</b> (str):  directory. 
 - <b>`ext`</b> (str, optional):  extension of the output. Defaults to 'svg'. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.line`
For line plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/line.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_range`

```python
plot_range(
    df00: DataFrame,
    colvalue: str,
    colindex: str,
    k: str,
    headsize: int = 15,
    headcolor: str = 'lightgray',
    ax: Axes = None
) → Axes
```

Plot range/intervals e.g. genome coordinates as lines. 



**Args:**
 
 - <b>`df00`</b> (pd.DataFrame):  input data. 
 - <b>`colvalue`</b> (str):  column with values. 
 - <b>`colindex`</b> (str):  column with ids. 
 - <b>`k`</b> (str):  subset name. 
 - <b>`headsize`</b> (int, optional):  margin at top. Defaults to 15. 
 - <b>`headcolor`</b> (str, optional):  color of the margin. Defaults to 'lightgray'. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/line.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_connections`

```python
plot_connections(
    dplot: DataFrame,
    label2xy: dict,
    colval: str = '$r_{s}$',
    line_scale: int = 40,
    legend_title: str = 'similarity',
    label2rename: dict = None,
    element2color: dict = None,
    xoff: float = 0,
    yoff: float = 0,
    rectangle: dict = {'width': 0.2, 'height': 0.32},
    params_text: dict = {'ha': 'center', 'va': 'center'},
    params_legend: dict = {'bbox_to_anchor': (1.1, 0.5), 'ncol': 1, 'frameon': False},
    legend_elements: list = [],
    params_line: dict = {'alpha': 1},
    ax: Axes = None,
    test: bool = False
) → Axes
```

Plot connections between points with annotations. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input data. 
 - <b>`label2xy`</b> (dict):  label to position. 
 - <b>`colval`</b> (str, optional):  column with values. Defaults to '{s}$'. 
 - <b>`line_scale`</b> (int, optional):  line_scale. Defaults to 40. 
 - <b>`legend_title`</b> (str, optional):  legend_title. Defaults to 'similarity'. 
 - <b>`label2rename`</b> (dict, optional):  label2rename. Defaults to None. 
 - <b>`element2color`</b> (dict, optional):  element2color. Defaults to None. 
 - <b>`xoff`</b> (float, optional):  xoff. Defaults to 0. 
 - <b>`yoff`</b> (float, optional):  yoff. Defaults to 0. 
 - <b>`rectangle`</b> (_type_, optional):  rectangle. Defaults to {'width':0.2,'height':0.32}. 
 - <b>`params_text`</b> (_type_, optional):  params_text. Defaults to {'ha':'center','va':'center'}. 
 - <b>`params_legend`</b> (_type_, optional):  params_legend. Defaults to {'bbox_to_anchor':(1.1, 0.5), 'ncol':1, 'frameon':False}. 
 - <b>`legend_elements`</b> (list, optional):  legend_elements. Defaults to []. 
 - <b>`params_line`</b> (_type_, optional):  params_line. Defaults to {'alpha':1}. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/line.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_kinetics`

```python
plot_kinetics(
    df1: DataFrame,
    x: str,
    y: str,
    hue: str,
    cmap: str = 'Reds_r',
    ax: Axes = None,
    test: bool = False,
    kws_legend: dict = {},
    **kws_set
) → Axes
```

Plot time-dependent kinetic data. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`x`</b> (str):  x column. 
 - <b>`y`</b> (str):  y column. 
 - <b>`hue`</b> (str):  hue column. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds_r'. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`kws_legend`</b> (dict, optional):  legend parameters. Defaults to {}. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/line.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_steps`

```python
plot_steps(
    df1: DataFrame,
    col_step_name: str,
    col_step_size: str,
    ax: Axes = None,
    test: bool = False
) → Axes
```

Plot step-wise changes in numbers, e.g. for a filtering process. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`col_step_size`</b> (str):  column containing the numbers. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object.     


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz`




**Global Variables**
---------------
- **colors**
- **figure**
- **io**
- **ax_**
- **annot**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.scatter`
For scatter plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_outlines`

```python
annot_outlines(
    data: DataFrame,
    colx: str,
    coly: str,
    column_outlines: str,
    outline_colors: dict,
    style=None,
    legend: bool = True,
    kws_legend: dict = {},
    zorder: int = 3,
    ax: Axes = None
) → Axes
```

Outline points on the scatter plot by categories. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_trendline`

```python
plot_trendline(
    dplot: DataFrame,
    colx: str,
    coly: str,
    poly: bool = False,
    lowess: bool = True,
    linestyle: str = ':',
    params_poly: dict = {'deg': 1},
    params_lowess: dict = {'frac': 0.7, 'it': 5},
    lw: float = 2,
    color: str = 'k',
    ax: Axes = None,
    **kws
) → Axes
```

Plot a trendline. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input dataframe. 
 - <b>`colx`</b> (str):  x column. 
 - <b>`coly`</b> (str):  y column. 
 - <b>`params_plot`</b> (dict, optional):  parameters provided to the plot. Defaults to {'color':'r','linestyle':'solid','lw':2}. 
 - <b>`poly`</b> (bool, optional):  apply polynomial function. Defaults to False. 
 - <b>`lowess`</b> (bool, optional):  apply lowess function. Defaults to True. 
 - <b>`params_poly`</b> (_type_, optional):  parameters provided to the polynomial function. Defaults to {'deg':1}. 
 - <b>`params_lowess`</b> (_type_, optional):  parameters provided to the lowess function.. Defaults to {'frac':0.7,'it':5}. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `plot` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs:  1. Label with goodness of fit, r (y_hat vs y) 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_scatter`

```python
plot_scatter(
    dplot: DataFrame,
    colx: str,
    coly: str,
    colz: str = None,
    kind: str = 'scatter',
    trendline_method: str = 'poly',
    stat_method: str = 'spearman',
    resample: bool = False,
    cmap: str = 'Reds',
    label_colorbar: str = None,
    gridsize: int = 25,
    bbox_to_anchor: list = [1, 1],
    loc: str = 'upper left',
    title: str = None,
    show_n: bool = True,
    show_n_prefix: str = '',
    params_plot_trendline: dict = {},
    params_set_label: dict = {},
    verbose: bool = False,
    ax: Axes = None,
    **kws
) → Axes
```

Plot scatter. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input dataframe. 
 - <b>`colx`</b> (str):  x column. 
 - <b>`coly`</b> (str):  y column. 
 - <b>`colz`</b> (str, optional):  z column. Defaults to None. 
 - <b>`kind`</b> (str, optional):  kind of scatter. Defaults to 'hexbin'. 
 - <b>`trendline_method`</b> (str, optional):  trendline method ['poly','lowess']. Defaults to 'poly'. 
 - <b>`stat_method`</b> (str, optional):  method of annoted stats ['mlr',"spearman"]. Defaults to "spearman". 
 - <b>`resample`</b> (bool, optional):  resample data. Defaults to False. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 
 - <b>`label_colorbar`</b> (str, optional):  label of the colorbar. Defaults to None. 
 - <b>`gridsize`</b> (int, optional):  number of grids in the hexbin. Defaults to 25. 
 - <b>`bbox_to_anchor`</b> (list, optional):  location of the legend. Defaults to [1,1]. 
 - <b>`loc`</b> (str, optional):  location of the legend. Defaults to 'upper left'. 
 - <b>`title`</b> (str, optional):  title of the plot. Defaults to None. 
 - <b>`#params_plot (dict, optional)`</b>:  parameters provided to the `plot` function. Defaults to {}. 
 - <b>`params_plot_trendline`</b> (dict, optional):  parameters provided to the `plot_trendline` function. Defaults to {}. 
 - <b>`params_set_label`</b> (dict, optional):  parameters provided to the `set_label` function. Defaults to dict(x=0,y=1). 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `plot` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 



**Notes:**

> For a rasterized scatter plot set `scatter_kws={'rasterized': True}` 
>TODOs: 1. Access the function as an attribute of roux-data i.e. `rd`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_qq`

```python
plot_qq(x: Series) → Axes
```

plot QQ. 



**Args:**
 
 - <b>`x`</b> (pd.Series):  input vector. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_ranks`

```python
plot_ranks(
    df1: DataFrame,
    colid: str,
    colx: str,
    coly: str = 'rank',
    ascending: bool = True,
    ax=None,
    **kws
) → Axes
```

Plot rankings. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input data. 
 - <b>`colx`</b> (str):  x column. 
 - <b>`coly`</b> (str):  y column. 
 - <b>`colid`</b> (str):  column with unique ids. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `seaborn.scatterplot` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/scatter.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_volcano`

```python
plot_volcano(
    data: DataFrame,
    colx: str,
    coly: str,
    colindex: str,
    hue: str = 'x',
    style: str = 'P=0',
    show_labels: int = None,
    show_outlines: int = None,
    outline_colors: list = ['k'],
    collabel: str = None,
    show_line=True,
    line_pvalue=0.1,
    line_x=0.0,
    show_text: bool = True,
    text_increase: str = None,
    text_decrease: str = None,
    text_diff: str = None,
    legend: bool = False,
    verbose: bool = False,
    p_min: float = None,
    ax: Axes = None,
    outmore: bool = False,
    kws_legend: dict = {},
    **kws_scatterplot
) → Axes
```

[UNDER DEVELOPMENT]Volcano plot. 



**Parameters:**
 

Keyword parameters: 



**Returns:**
  plt.Axes 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.sets`
For plotting sets. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/sets.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_venn`

```python
plot_venn(
    ds1: Series,
    ax: Axes = None,
    figsize: tuple = [2.5, 2.5],
    show_n: bool = True,
    outmore=False,
    **kws
) → Axes
```

Plot Venn diagram. 



**Args:**
 
 - <b>`ds1`</b> (pd.Series):  input pandas.Series or dictionary. Subsets in the index levels, mapped to counts.  
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`figsize`</b> (tuple, optional):  figure size. Defaults to [2.5,2.5]. 
 - <b>`show_n`</b> (bool, optional):  show sample sizes. Defaults to True. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object.         


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/sets.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_intersections`

```python
plot_intersections(
    ds1: Series,
    item_name: str = None,
    figsize: tuple = [4, 4],
    text_width: float = 2,
    yorder: list = None,
    sort_by: str = 'cardinality',
    sort_categories_by: str = None,
    element_size: int = 40,
    facecolor: str = 'gray',
    bari_annot: int = None,
    totals_bar: bool = False,
    totals_text: bool = True,
    intersections_ylabel: float = None,
    intersections_min: float = None,
    test: bool = False,
    annot_text: bool = False,
    set_ylabelx: float = -0.25,
    set_ylabely: float = 0.5,
    **kws
) → Axes
```

Plot upset plot. 



**Args:**
 
 - <b>`ds1`</b> (pd.Series):  input vector. 
 - <b>`item_name`</b> (str, optional):  name of items. Defaults to None. 
 - <b>`figsize`</b> (tuple, optional):  figure size. Defaults to [4,4]. 
 - <b>`text_width`</b> (float, optional):  max. width of the text. Defaults to 2. 
 - <b>`yorder`</b> (list, optional):  order of y elements. Defaults to None. 
 - <b>`sort_by`</b> (str, optional):  sorting method. Defaults to 'cardinality'. 
 - <b>`sort_categories_by`</b> (str, optional):  sorting method. Defaults to None. 
 - <b>`element_size`</b> (int, optional):  size of elements. Defaults to 40. 
 - <b>`facecolor`</b> (str, optional):  facecolor. Defaults to 'gray'. 
 - <b>`bari_annot`</b> (int, optional):  annotate nth bar. Defaults to None. 
 - <b>`totals_text`</b> (bool, optional):  show totals. Defaults to True. 
 - <b>`intersections_ylabel`</b> (float, optional):  y-label of the intersections. Defaults to None. 
 - <b>`intersections_min`</b> (float, optional):  intersection minimum to show. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`annot_text`</b> (bool, optional):  annotate text. Defaults to False. 
 - <b>`set_ylabelx`</b> (float, optional):  x position of the ylabel. Defaults to -0.25. 
 - <b>`set_ylabely`</b> (float, optional):  y position of the ylabel. Defaults to 0.5. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `upset.plot` function.  



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 



**Notes:**

> sort_by:{‘cardinality’, ‘degree’} If ‘cardinality’, subset are listed from largest to smallest. If ‘degree’, they are listed in order of the number of categories intersected. sort_categories_by:{‘cardinality’, None} Whether to sort the categories by total cardinality, or leave them in the provided order. 
>References: https://upsetplot.readthedocs.io/en/stable/api.html 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/viz/sets.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_enrichment`

```python
plot_enrichment(
    data: DataFrame,
    x: str,
    y: str,
    s: str,
    hue='Q',
    xlabel=None,
    ylabel='significance\n(-log10(Q))',
    size: int = None,
    color: str = None,
    annots_side: int = 5,
    annots_side_labels=None,
    coff_fdr: float = None,
    xlim: tuple = None,
    xlim_off: float = 0.2,
    ylim: tuple = None,
    ax: Axes = None,
    break_pt: int = 25,
    annot_coff_fdr: bool = False,
    kws_annot: dict = {'loc': 'right', 'offx3': 0.15},
    returns='ax',
    **kwargs
) → Axes
```

Plot enrichment stats. 



**Args:**
 
     - <b>`data`</b> (pd.DataFrame):  input data. 
     - <b>`x`</b> (str):  x column. 
     - <b>`y`</b> (str):  y column. 
     - <b>`s`</b> (str):  size column. 
     - <b>`size`</b> (int, optional):  size of the points. Defaults to None. 
     - <b>`color`</b> (str, optional):  color of the points. Defaults to None. 
     - <b>`annots_side`</b> (int, optional):  how many labels to show on side. Defaults to 5. 
     - <b>`coff_fdr`</b> (float, optional):  FDR cutoff. Defaults to None. 
     - <b>`xlim`</b> (tuple, optional):  x-axis limits. Defaults to None. 
     - <b>`xlim_off`</b> (float, optional):  x-offset on limits. Defaults to 0.2. 
     - <b>`ylim`</b> (tuple, optional):  y-axis limits. Defaults to None. 
     - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
     - <b>`break_pt`</b> (int, optional):  break point (' ') for the labels. Defaults to 25. 
     - <b>`annot_coff_fdr`</b> (bool, optional):  show FDR cutoff. Defaults to False. 
     - <b>`kws_annot`</b> (dict, optional):  parameters provided to the `annot_side` function. Defaults to dict( loc='right', annot_count_max=5, offx3=0.15, ). 

Keyword Args: 
     - <b>`kwargs`</b>:  parameters provided to the `sns.scatterplot` function.  



**Returns:**
 
     - <b>`plt.Axes`</b>:  `plt.Axes` object. 




<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/vizi"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.vizi`






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/vizi/scatter"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.vizi.scatter`






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.df`
For management of tables. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/df.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `exclude_items`

```python
exclude_items(df1: DataFrame, metadata: dict) → DataFrame
```

Exclude items from the table with the workflow info. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`metadata`</b> (dict):  metadata of the repository. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.function`
For function management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_quoted_path`

```python
get_quoted_path(s1: str) → str
```

Quoted paths. 



**Args:**
 
 - <b>`s1`</b> (str):  path. 



**Returns:**
 
 - <b>`str`</b>:  quoted path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_path`

```python
get_path(
    s: str,
    validate: bool,
    prefixes=['data/', 'metadata/', 'plot/'],
    test=False
) → str
```

Extract pathsfrom a line of code. 



**Args:**
 
 - <b>`s`</b> (str):  line of code. 
 - <b>`validate`</b> (bool):  validate the output. 
 - <b>`prefixes`</b> (list, optional):  allowed prefixes. Defaults to ['data/','metadata/','plot/']. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  path. 

TODOs: 1. Use wildcards i.e. *'s. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `remove_dirs_from_outputs`

```python
remove_dirs_from_outputs(outputs: list, test: bool = False) → list
```

Remove directories from the output paths. 



**Args:**
 
 - <b>`outputs`</b> (list):  output paths. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  paths. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ios`

```python
get_ios(l: list, test=False) → tuple
```

Get input and output (IO) paths. 



**Args:**
 
 - <b>`l`</b> (list):  list of lines of code. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  paths of inputs and outputs. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_name`

```python
get_name(s: str, i: int, sep_step: str = '## step') → str
```

Get name of the function. 



**Args:**
 
 - <b>`s`</b> (str):  lines in markdown format. 
 - <b>`sep_step`<.py# step".         
 - <b>`i`</b> (int):  index of the step. 



**Returns:**
 
 - <b>`str`</b>:  name of the function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_step`

```python
get_step(
    l: list,
    name: str,
    sep_step: str = '## step',
    sep_step_end: str = '## tests',
    test=False,
    tab='    '
) → dict
```

Get code for a step. 



**Args:**
 
 - <b>`l`</b> (list):  list of lines of code 
 - <b>`name`</b> (str):  name of the function. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`tab`</b> (str, optional):  tab format. Defaults to '    '. 



**Returns:**
 
 - <b>`dict`</b>:  step name to code map.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/function.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_task`

```python
to_task(
    notebookp,
    task=None,
    sep_step: str = '## step',
    sep_step_end: str = '## tests',
    notebook_suffix: str = '_v',
    force=False,
    validate=False,
    path_prefix=None,
    verbose=True,
    test=False
) → str
```

Get the lines of code for a task (script to be saved as an individual `.py` file). 



**Args:**
 
 - <b>`notebookp`</b> (_type_):  path of the notebook. 
 - <b>`sep_step`<.py# step".         
 - <b>`sep_step_end`<.py# tests".         
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 
 - <b>`validate`</b> (bool, optional):  validate output. Defaults to False. 
 - <b>`path_prefix`</b> (_type_, optional):  prefix to the path. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  show verbose. Defaults to True. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  lines of the code. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.io`
For input/output of workflow. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clear_variables`

```python
clear_variables(dtype=None, variables=None)
```

Clear dataframes from the workspace. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clear_dataframes`

```python
clear_dataframes()
```






---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_lines`

```python
get_lines(p: str, keep_comments: bool = True) → list
```

Get lines of code from notebook. 



**Args:**
 
 - <b>`p`</b> (str):  path to notebook. 
 - <b>`keep_comments`</b> (bool, optional):  keep comments. Defaults to True. 



**Returns:**
 
 - <b>`list`</b>:  lines. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_py`

```python
to_py(
    notebookp: str,
    pyp: str = None,
    force: bool = False,
    **kws_get_lines
) → str
```

To python script (.py). 



**Args:**
 
 - <b>`notebookp`</b> (str):  path to the notebook path. 
 - <b>`pyp`</b> (str, optional):  path to the python file. Defaults to None. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  path of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `import_from_file`

```python
import_from_file(pyp: str)
```

Import functions from python (`.py`) file. 



**Args:**
 
 - <b>`pyp`</b> (str):  python file (`.py`). 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_parameters`

```python
to_parameters(f: object, test: bool = False) → dict
```

Get function to parameters map. 



**Args:**
 
 - <b>`f`</b> (object):  function. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_nb_md`

```python
read_nb_md(p: str) → list
```

Read notebook's documentation in the markdown cells. 



**Args:**
 
 - <b>`p`</b> (str):  path of the notebook. 



**Returns:**
 
 - <b>`list`</b>:  lines of the strings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_config`

```python
read_config(p: str, config_base=None, convert_dtype: bool = True)
```

Read configuration. 



**Parameters:**
 
 - <b>`p`</b> (str):  input path.  


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_metadata`

```python
read_metadata(
    p: str = './metadata.yaml',
    ind: str = None,
    max_paths: int = 30,
    config_paths: list = [],
    config_paths_auto=False,
    verbose: bool = False,
    **kws_read_config
) → dict
```

Read metadata. 



**Args:**
 
 - <b>`p`</b> (str, optional):  file containing metadata. Defaults to './metadata.yaml'. 
 - <b>`ind`</b> (str, optional):  directory containing specific setings and other data to be incorporated into metadata. Defaults to './metadata/'. 



**Returns:**
 
 - <b>`dict`</b>:  output. 

TODOs: 1. Metadata files include colors.yaml, database.yaml, constants.yaml etc. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_info`

```python
to_info(p: str = '*_*_v*.ipynb', outp: str = 'README.md') → str
```

Save README.md file. 



**Args:**
 
 - <b>`p`</b> (str, optional):  path of the notebook files that would be converted to "tasks". Defaults to '*_*_v*.ipynb'. 
 - <b>`outp`</b> (str, optional):  path of the output file. Defaults to 'README.md'. 



**Returns:**
 
 - <b>`str`</b>:  path of the output file. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_symlinks`

```python
make_symlinks(
    d1: dict,
    d2: dict,
    project_path: str,
    data: bool = True,
    notebook_suffix: str = '_v',
    test: bool = False
) → list
```

Make symbolic links. 



**Args:**
 
 - <b>`d1`</b> (dict):  `project name` to `repo name`. 
 - <b>`d2`</b> (dict):  `task name` to tuple containing `from project name` `to project name`. 
 - <b>`project_path`</b> (str):  path of the repository. 
 - <b>`data`</b> (bool, optional):  make links for the data. Defaults to True. 
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  list of commands. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L300"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_workflow`

```python
to_workflow(df2: DataFrame, workflowp: str, tab: str = '    ') → str
```

Save workflow file. 



**Args:**
 
 - <b>`df2`</b> (pd.DataFrame):  input table. 
 - <b>`workflowp`</b> (str):  path of the workflow file. 
 - <b>`tab`</b> (str, optional):  tab format. Defaults to '    '. 



**Returns:**
 
 - <b>`str`</b>:  path of the workflow file. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_workflow_report`

```python
create_workflow_report(workflowp: str, env: str) → int
```

Create report for the workflow run. 



**Parameters:**
 
 - <b>`workflowp`</b> (str):  path of the workflow file (`snakemake`). 
 - <b>`env`</b> (str):  name of the conda virtual environment where required the workflow dependency is available i.e. `snakemake`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/io.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_diff_notebooks`

```python
to_diff_notebooks(
    notebook_paths,
    url_prefix='https://localhost:8888/nbdime/difftool?',
    remove_prefix='file://',
    verbose=True
) → list
```

"Diff" notebooks using `nbdiff` (https://nbdime.readthedocs.io/en/latest/) 

Todos:  1. Deprecate if functionality added to `nbdiff-web`. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.knit`
For workflow set up. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/knit.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `nb_to_py`

```python
nb_to_py(
    notebookp: str,
    test: bool = False,
    validate: bool = True,
    sep_step: str = '## step',
    notebook_suffix: str = '_v'
)
```

notebook to script. 



**Args:**
 
 - <b>`notebookp`</b> (str):  path to the notebook. 
 - <b>`sep_step`<.py# step". 
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`validate`</b> (bool, optional):  validate. Defaults to True. 

TODOs:  1. Add `check_outputs` parameter to only filter out non-executable code (i.e. tests) if False else edit the code. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/knit.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_stepns`

```python
sort_stepns(l: list) → list
```

Sort steps (functions) of a task (script). 



**Args:**
 
 - <b>`l`</b> (list):  list of steps. 



**Returns:**
 
 - <b>`list`</b>:  sorted list of steps. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow`




**Global Variables**
---------------
- **io**
- **df**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.monitor`
For workflow monitors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/monitor.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_workflow_log`

```python
plot_workflow_log(dplot: DataFrame) → Axes
```

Plot workflow log. 



**Args:**
 
 - <b>`dplot`</b> (pd.DataFrame):  input data (dparam). 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 

TODOs: 1. use the statistics tagged as `## stats`.  


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.task`
For task management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/task.py#L5"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_experiment`

```python
run_experiment(
    parameters: dict,
    input_notebook_path: str,
    kernel: str,
    output_notebook_path: str = None,
    test=False,
    verbose=False,
    **kws_papermill
)
```

[UNDER DEVELOPMENT] Execute a single notebook.     


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/task.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_experiments`

```python
run_experiments(
    input_notebook_path: str,
    kernel: str,
    inputs: list = None,
    output_path: str = None,
    parameters_list: list = None,
    fast: bool = False,
    test1: bool = False,
    force: bool = False,
    test: bool = False,
    verbose: bool = False,
    **kws_papermill
)
```

[UNDER DEVELOPMENT] Execute a list of notebooks. 

TODOs:   1. Integrate with apply_on_paths for parallel processing etc.  2. Reporting by quarto? 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.version`
For version control. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/version.py#L3"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `git_commit`

```python
git_commit(repop: str, suffix_message: str = '')
```

Version control. 



**Args:**
 
 - <b>`repop`</b> (str):  path to the repository. 
 - <b>`suffix_message`</b> (str, optional):  add suffix to the version (commit) message. Defaults to ''. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.workflow`
For workflow management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/workflow.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_scripts`

```python
get_scripts(
    ps: list,
    notebook_prefix: str = '\\d{2}',
    notebook_suffix: str = '_v\\d{2}',
    test: bool = False,
    fast: bool = True,
    cores: int = 6,
    force: bool = False,
    tab: str = '    ',
    **kws
) → DataFrame
```

Get scripts. 



**Args:**
 
 - <b>`ps`</b> (list):  paths. 
 - <b>`notebook_prefix`</b> (str, optional):  prefix of the notebook file to be considered as a "task". 
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`fast`</b> (bool, optional):  parallel processing. Defaults to True. 
 - <b>`cores`</b> (int, optional):  cores to use. Defaults to 6. 
 - <b>`force`</b> (bool, optional):  overwrite the outputs. Defaults to False. 
 - <b>`tab`</b> (str, optional):  tab in spaces. Defaults to '    '. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output table. 


---

<a href="https://github.com/rraadd88/roux/blob/master/roux/workflow/workflow.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_scripts`

```python
to_scripts(
    packagep: str,
    notebooksdp: str,
    validate: bool = False,
    ps: list = None,
    notebook_prefix: str = '\\d{2}',
    notebook_suffix: str = '_v\\d{2}',
    scripts: bool = True,
    workflow: bool = True,
    sep_step: str = '## step',
    todos: bool = False,
    git: bool = True,
    clean: bool = False,
    test: bool = False,
    force: bool = True,
    tab: str = '    ',
    **kws
)
```

To scripts. 



**Args:**
 
 - <b>`# packagen (str)`</b>:  package name. 
 - <b>`packagep`</b> (str):  path to the package. 
 - <b>`notebooksdp`</b> (str, optional):  path to the notebooks. Defaults to None. 
 - <b>`validate`</b> (bool, optional):  validate if functions are formatted correctly. Defaults to False. 
 - <b>`ps`</b> (list, optional):  paths. Defaults to None. 
 - <b>`notebook_prefix`</b> (str, optional):  prefix of the notebook file to be considered as a "task". 
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`scripts`</b> (bool, optional):  make scripts. Defaults to True. 
 - <b>`workflow`</b> (bool, optional):  make workflow file. Defaults to True. 
 - <b>`sep_step`<.py# step". 
 - <b>`todos`</b> (bool, optional):  show todos. Defaults to False. 
 - <b>`git`</b> (bool, optional):  save version. Defaults to True. 
 - <b>`clean`</b> (bool, optional):  clean temporary files. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`force`</b> (bool, optional):  overwrite outputs. Defaults to True. 
 - <b>`tab`</b> (str, optional):  tab size. Defaults to '    '. 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to the `get_script` function,   including `sep_step` and `sep_step_end` 

TODOs:  
 - <b>`1. For version control, use https`</b>: //github.com/jupyterlab/jupyterlab-git. 


