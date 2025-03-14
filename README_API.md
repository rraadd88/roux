# API
<!-- separated from main README bcz it is slow to render on GH -->
<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.compare`
For comparative plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/compare.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.cluster`
For clustering data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_clusters`

```python
check_clusters(df: DataFrame)
```

Check clusters. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_gmm_params`

```python
get_gmm_params(g, x, n_clusters=2, test=False)
```

Intersection point of the two peak Gaussian mixture Models (GMMs). 



**Args:**
 
 - <b>`out`</b> (str):  `coff` only or `params` for all the parameters. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_gmm_intersection`

```python
get_gmm_intersection(x, two_pdfs, means, weights, test=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/cluster.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.version`
For version control. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/version.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `git_commit`

```python
git_commit(repop: str, suffix_message: str = '', force=False)
```

Version control. 



**Args:**
 
 - <b>`repop`</b> (str):  path to the repository. 
 - <b>`suffix_message`</b> (str, optional):  add suffix to the version (commit) message. Defaults to ''. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.log`





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/log.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `print_parameters`

```python
print_parameters(d: dict)
```

Print a directory with parameters as lines of code 



**Parameters:**
 
 - <b>`d`</b> (dict):  directory with parameters 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/log.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `test_params`

```python
test_params(params, i=0)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.io`
For input/output of workflow. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clear_variables`

```python
clear_variables(dtype=None, variables=None)
```

Clear dataframes from the workspace. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clear_dataframes`

```python
clear_dataframes()
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_nb_cells`

```python
to_nb_cells(notebook, outp, new_cells, validate_diff=None)
```

Replace notebook cells. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `import_from_file`

```python
import_from_file(pyp: str)
```

Import functions from python (`.py`) file. 



**Args:**
 
 - <b>`pyp`</b> (str):  python file (`.py`). 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `infer_parameters`

```python
infer_parameters(input_value, default_value)
```

Infer the input values and post warning messages. 



**Parameters:**
 
 - <b>`input_value`</b>:  the primary value. 
 - <b>`default_value`</b>:  the default/alternative/inferred value. 



**Returns:**
 Inferred value. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_config`

```python
read_config(
    p: str,
    config_base=None,
    inputs=None,
    append_to_key=None,
    convert_dtype: bool = True,
    verbose: bool = True
)
```

Read configuration. 



**Parameters:**
 
 - <b>`p`</b> (str):  input path. 
 - <b>`config_base`</b>:  base config with the inputs for the interpolations 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_metadata`

```python
read_metadata(
    p: str,
    ind: str = None,
    max_paths: int = 30,
    config_path_key: str = 'config_path',
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


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_workflow_report`

```python
create_workflow_report(workflowp: str, env: str) → int
```

Create report for the workflow run. 



**Parameters:**
 
 - <b>`workflowp`</b> (str):  path of the workflow file (`snakemake`). 
 - <b>`env`</b> (str):  name of the conda virtual environment where required the workflow dependency is available i.e. `snakemake`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L400"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `replacestar`

```python
replacestar(
    input_path,
    output_path=None,
    replace_from='from roux.global_imports import *',
    in_place: bool = False,
    attributes={'pandarallel': ['parallel_apply'], 'rd': ['.rd.', '.log.']},
    verbose: bool = False,
    test: bool = False,
    **kws_fix_code
)
```

Post-development, replace wildcard (global) import from roux i.e. 'from roux.global_imports import *' with individual imports with accompanying documentation. 

Usage:  For notebooks developed using roux.global_imports. 

Parameters  input_path (str): path to the .py or .ipynb file.  output_path (str): path to the output.  py_path (str): path to the intermediate .py file.  in_place (bool): whether to carry out the modification in place.  return_replacements (bool): return dict with strings to be replaced.  attributes (dict): attribute names mapped to their keywords for searching.  verbose (bool): verbose toggle.  test (bool): test-mode if output file not provided and in-place modification not allowed. 



**Returns:**
 
 - <b>`output_path`</b> (str):  path to the modified notebook. 



**Examples:**
 roux replacestar -i notebook.ipynb roux replacestar -i notebooks/*.ipynb 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `replacestar_ruff`

```python
replacestar_ruff(
    p: str,
    outp: str,
    replace: str = 'from roux.global_imports import *',
    verbose=True
) → str
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L652"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `post_code`

```python
post_code(p: str, lint: bool, format: bool, verbose: bool = True)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/io.py#L674"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_clean_nb`

```python
to_clean_nb(
    p,
    outp: str = None,
    in_place: bool = False,
    temp_outp: str = None,
    clear_outputs=False,
    drop_code_lines_containing=['.*%run .*', '^#\\s*.*=.*', '^#\\s*".*', "^#\\s*'.*", '^#\\s*f".*', "^#\\s*f'.*", '^#\\s*df.*', '^#\\s*.*kws_.*', '^\\s*#\\s*$', '^\\s*#\\s*break\\s*$', '\\[X', '\\[old ', '#old', '# old', '\\[not used', '# not used', '#tmp', '# tmp', '#temp', '# temp', 'check ', 'checking', '# check', '\\[SKIP', 'DEBUG '],
    drop_headers_containing=['check', '[check', 'old', '[old', 'tmp', '[tmp'],
    lint=False,
    format=False,
    **kws_fix_code
) → str
```

Wraper around the notebook post-processing functions. 

Usage:  For notebooks developed using roux.global_imports. 

 On command line: 

 ## single input  roux to-clean-nb in.ipynb out.ipynb -c -l -f 

 ## multiple inputs  roux to-clean-nb "in*.ipynb" -i -c -l -f 



**Parameters:**
 
 - <b>`temp_outp`</b> (str):  path to the intermediate output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.image`
For visualization of images. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/image.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/image.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_images`

```python
plot_images(image_paths, ncols=3, title_func=None, size=3)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.sys`
For processing file paths for example. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `remove_exts`

```python
remove_exts(p: str)
```

Filename without the extension. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_ps`

```python
read_ps(ps, test: bool = True, verbose: bool = True) → list
```

Read a list of paths. 



**Parameters:**
 
 - <b>`ps`</b> (list|str):  list of paths or a string with wildcard/s. 
 - <b>`test`</b> (bool):  testing. 
 - <b>`verbose`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`ps`</b> (list):  list of paths. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_output_paths`

```python
to_output_paths(
    input_paths: list = None,
    inputs: list = None,
    output_path_base: str = None,
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
 - <b>`output_path_base (str) `</b>:  output path with a placeholder '{KEY}' to be replaced. Defaults to None. 
 - <b>`encode_short`</b>:  (bool) : short encoded string, else long encoded string (reversible) is used. Defaults to True. 
 - <b>`replaces_output_path `</b>:  list, dictionary or function to replace the input paths. Defaults to None. 
 - <b>`key_output_path (str) `</b>:  key to be used to incorporate output_path variable among the inputs. Defaults to None. 
 - <b>`force`</b> (bool):  overwrite the outputs. Defaults to False. 
 - <b>`verbose (bool) `</b>:  show verbose. Defaults to False. 



**Returns:**
 dictionary with the output path mapped to input paths or inputs. 

TODOs: 1. Placeholders other than {KEY}. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_com`

```python
run_com(com: str, env=None, test: bool = False, **kws)
```

Run a bash command. 



**Args:**
 
 - <b>`com`</b> (str):  command. 
 - <b>`env`</b> (str):  environment name. 
 - <b>`test`</b> (bool, optional):  testing. Defaults to False. 



**Returns:**
 
 - <b>`output`</b>:  output of the `subprocess.call` function. 

TODOs: 1. logp 2. error ignoring 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_com`

```python
run_com(com: str, env=None, test: bool = False, **kws)
```

Run a bash command. 



**Args:**
 
 - <b>`com`</b> (str):  command. 
 - <b>`env`</b> (str):  environment name. 
 - <b>`test`</b> (bool, optional):  testing. Defaults to False. 



**Returns:**
 
 - <b>`output`</b>:  output of the `subprocess.call` function. 

TODOs: 1. logp 2. error ignoring 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `create_symlink`

```python
create_symlink(p: str, outp: str, test=False, force=False)
```

Create symbolic links. 



**Args:**
 
 - <b>`p`</b> (str):  input path. 
 - <b>`outp`</b> (str):  output path. 
 - <b>`test`</b> (bool, optional):  test. Defaults to False. 



**Returns:**
 
 - <b>`outp`</b> (str):  output path. 

TODOs: 
 - <b>`Use `pathlib``</b>:  `Path(p).symlink_to(Path(outp))` 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L527"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_interactive`

```python
is_interactive()
```

Check if the UI is interactive e.g. jupyter or command line. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_interactive_notebook`

```python
is_interactive_notebook()
```

Check if the UI is interactive e.g. jupyter or command line. 



**Notes:**

> 
>Reference: 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L563"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L580"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L624"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L648"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L694"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `tree`

```python
tree(folder_path: str, log=True)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/sys.py#L710"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `grep`

```python
grep(
    p: str,
    checks: list,
    exclude: list = [],
    exclude_str: list = [],
    verbose: bool = True
) → list
```

To get the output of grep as a list of strings. 



**Parameters:**
 
 - <b>`p`</b> (str):  input path 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.transform`
For transformations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Paramters:  x: input.  errors (str): Defaults to 'raise' else replace (in case of visualization only).  p_min (float): Replace zeros with this value. Note: to be used for visualization only. 



**Returns:**
  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_q`

```python
get_q(ds1: Series, col: str = None, verb: bool = True, test_coff: float = 0.1)
```

To FDR corrected P-value. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/transform.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rescale_divergent`

```python
rescale_divergent(df1: DataFrame, col: str, col_sign: str = None) → DataFrame
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.ds`
For processing pandas Series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/ds.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_near_quantile`

```python
get_near_quantile(x: Series, q: float)
```

Retrieve the nearest value to a quantile. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.dist`
For distribution plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_jitter_positions`

```python
get_jitter_positions(ax, df1, order, column_category, column_position)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    offs_pval: dict = None,
    fmt_pval: str = '<',
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
 - <b>`offs_pval`</b> (float, optional):  x and y offsets for the p-value labels. 
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/dist.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.theme`
Theming. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/theme.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_theme`

```python
set_theme(
    font: str = 'Myriad Pro',
    fontsize: int = 12,
    pad: int = 2,
    palette: list = ['#50AADC', '#D3DDDC', '#F1D929', '#f55f5f', '#046C9A', '#00A08A', '#F2AD00', '#F98400', '#5BBCD6', '#ECCBAE', '#D69C4E', '#ABDDDE', '#000000']
)
```

Set the theme. 



**Parameters:**
 
 - <b>`font`</b> (str):  font name. 
 - <b>`fontsize`</b> (int):  font size. 
 - <b>`pad`</b> (int):  padding. 

TODOs: Addition of `palette` options. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.workflow`
For workflow management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/workflow.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/workflow.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`sep_step`</b> (str, optional):  separator marking the start of a step. Defaults to "## step". 
 - <b>`todos`</b> (bool, optional):  show todos. Defaults to False. 
 - <b>`git`</b> (bool, optional):  save version. Defaults to True. 
 - <b>`clean`</b> (bool, optional):  clean temporary files. Defaults to False. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`force`</b> (bool, optional):  overwrite outputs. Defaults to True. 
 - <b>`tab`</b> (str, optional):  tab size. Defaults to '    '. 

Keyword parameters: 
 - <b>`kws`</b>:  parameters provided to the `get_script` function,  including `sep_step` and `sep_step_end` 

TODOs: 
 - <b>`1. For version control, use https`</b>: //github.com/jupyterlab/jupyterlab-git. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat`




**Global Variables**
---------------
- **binary**
- **io**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.io`
For input/output of data files. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_zip_dir`

```python
to_zip_dir(source, destination=None, fmt='zip')
```

Zip a folder. Ref: https://stackoverflow.com/a/50381250/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_zip`

```python
to_zip(
    p: str,
    outp: str = None,
    func_rename=None,
    fmt: str = 'zip',
    test: bool = False
)
```

Compress a file/directory. 



**Parameters:**
 
 - <b>`p`</b> (str):  path to the file/directory. 
 - <b>`outp`</b> (str):  path to the output compressed file. 
 - <b>`fmt`</b> (str):  format of the compressed file. 



**Returns:**
 
 - <b>`outp`</b> (str):  path of the compressed file. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_dir`

```python
to_dir(
    paths: dict,
    output_dir_path: str,
    rename_basename=None,
    force=False,
    test=False
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_version`

```python
to_version(
    p: str,
    outd: str = None,
    test: bool = False,
    label: str = '',
    **kws: dict
) → str
```

Rename a file/directory to a version. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 
 - <b>`outd`</b> (str):  output directory. 

Keyword parameters: 
 - <b>`kws`</b> (dict):  provided to `get_version`. 



**Returns:**
 
 - <b>`version`</b> (string):  version. 

TODOs: 1. Use `to_dir`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

TODOs: 1. Use `to_dir`. 2. Option to remove dirs  find and move/zip  "find -regex .*/_.*"  "find -regex .*/test.*" 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L291"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download`

```python
download(
    url: str,
    path: str = None,
    outd: str = None,
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_text`

```python
read_text(p)
```

Read a file. To be called by other functions 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  contents. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `is_dict`

```python
is_dict(p)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L477"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L522"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`clean`</b> (bool):  whether to apply `clean` function. tables () 
 - <b>`verbose`</b> (bool):  verbose. 

Keyword parameters: 
 - <b>`kws_clean`</b> (dict):  paramters provided to the `clean` function. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    use_paths: bool = False,
    tables: int = 1,
    test: bool = False,
    verbose: bool = True,
    engine: str = 'pyarrow',
    **kws_read_tables: dict
)
```

 Table/s reader. 



**Parameters:**
 
     - <b>`p`</b> (str):  path of the file. It could be an input for `read_ps`, which would include strings with wildcards, list etc. 
     - <b>`ext`</b> (str):  extension of the file (default: None meaning infered from the path). 
     - <b>`clean=(default`</b>: True). filterby_time=None). 
     - <b>`check_paths`</b> (bool):  read files in the path column (default:True). 
     - <b>`use_paths`</b> (bool):  forced read files in the path column (default:False). 
     - <b>`test`</b> (bool):  testing (default:False). 
     - <b>`params`</b>:  parameters provided to the 'pd.read_csv' (default:{}). For example 
     - <b>`params['columns']`</b>:  columns to read. 
     - <b>`kws_clean`</b>:  parameters provided to 'rd.clean' (default:{}). 
     - <b>`kws_cloud`</b>:  parameters for reading files from google-drive (default:{}). 
     - <b>`tables`</b>:  how many tables to be read (default:1). 
     - <b>`verbose`</b>:  verbose (default:True). 

Keyword parameters: 
     - <b>`kws_read_tables`</b> (dict):  parameters provided to `read_tables` function. For example: 
     - <b>`to_col={colindex`</b>:  replaces_index} 



**Returns:**
 
     - <b>`df`</b> (DataFrame):  output dataframe. 



**Examples:**
 1. For reading specific columns only set `params=dict(columns=list)`. 

2. For reading many files, convert paths to a column with corresponding values: 

 to_col={colindex: replaces_index} 

3. Reading a vcf file.  p='*.vcf|vcf.gz'  read_table(p,  params_read_csv=dict(  #compression='gzip',  sep='        ',comment='#',header=None,  names=replace_many(get_header(path,comment='#',lineno=-1),['#',' '],'').split('  '))  ) 




---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L741"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L760"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `apply_on_paths`

```python
apply_on_paths(
    ps: list,
    func,
    replaces_outp: str = None,
    to_col: dict = None,
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
 - <b>`to_col`</b> (dict):  convert the paths to a column e.g. {colindex: replaces_index} 
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
  1. Function:  def apply_(p,outd='data/data_analysed',force=False):  outp=f"{outd}/{basenamenoext(p)}.pqt'  if exists(outp) and not force:  return  df01=read_table(p)  apply_on_paths(  ps=glob("data/data_analysed/*"),  func=apply_,  outd="data/data_analysed/",  force=True,  fast=False,  read_path=True,  ) 

TODOs: Move out of io. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L954"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1006"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1051"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_manytables`

```python
to_manytables(
    df: DataFrame,
    p: str,
    colgroupby: str,
    fmt: str = '',
    ignore: bool = False,
    kws_get_chunks={},
    **kws_to_table
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_table_pqt`

```python
to_table_pqt(
    df: DataFrame,
    p: str,
    engine: str = 'pyarrow',
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_excel`

```python
to_excel(
    sheetname2df: dict,
    outp: str,
    comments: dict = None,
    save_input: bool = False,
    author: str = None,
    append: bool = False,
    adjust_column_width: bool = True,
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/io.py#L1373"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib`




**Global Variables**
---------------
- **set**
- **str**
- **sys**
- **df**
- **dfs**
- **text**
- **io**

---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L3"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decorator`

```python
decorator(func)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decorator`

```python
decorator(func)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `rd`
`roux-dataframe` (`.rd`) extension. 

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```









---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `rs`
`roux-series` (`.rs`) extension. 

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```









<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.figure`
For setting up figures. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/figure.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_children`

```python
get_children(fig)
```

Get all the individual objects included in the figure. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/figure.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_child_text`

```python
get_child_text(search_name, all_children=None, fig=None)
```

Get text object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/figure.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `align_texts`

```python
align_texts(fig, texts: list, align: str, test=False)
```

Align text objects. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/figure.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `labelplots`

```python
labelplots(
    axes: list = None,
    fig=None,
    labels: list = None,
    xoff: float = 0,
    yoff: float = 0,
    auto: bool = False,
    xoffs: dict = {},
    yoffs: dict = {},
    va: str = 'center',
    ha: str = 'left',
    verbose: bool = True,
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

Todos: 1. Get the x coordinate of the ylabel. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/figure.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_axs`

```python
annot_axs(data, ax1, ax2, cols, **kws_line)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.function`
For function management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_name`

```python
get_name(s: str, i: int, sep_step: str = '## step') → str
```

Get name of the function. 



**Args:**
 
 - <b>`s`</b> (str):  lines in markdown format. 
 - <b>`sep_step`</b> (str, optional):  separator marking the start of a step. Defaults to "## step". 
 - <b>`i`</b> (int):  index of the step. 



**Returns:**
 
 - <b>`str`</b>:  name of the function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`sep_step`</b> (str, optional):  separator marking the start of a step. Defaults to "## step". 
 - <b>`sep_step_end`</b> (str, optional):  separator marking the end of a step. Defaults to "## tests". 
 - <b>`notebook_suffix`</b> (str, optional):  suffix of the notebook file to be considered as a "task". 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 
 - <b>`validate`</b> (bool, optional):  validate output. Defaults to False. 
 - <b>`path_prefix`</b> (_type_, optional):  prefix to the path. Defaults to None. 
 - <b>`verbose`</b> (bool, optional):  show verbose. Defaults to True. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`str`</b>:  lines of the code. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/function.py#L388"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_global_imports`

```python
get_global_imports() → DataFrame
```

Get the metadata of the functions imported from `from roux import global_imports`. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.fit`
For fitting data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/fit.py#L383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.sets`
For set related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_overlap`

```python
get_overlap(
    items_set: list,
    items_test: list,
    output_format: str = 'list'
) → list
```

Get overlapping items as a string. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`output_format`</b> (str, optional):  format of the output. Defaults to 'list'. 



**Raises:**
 
 - <b>`ValueError`</b>:  output_format can be list or str 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_overlap_size`

```python
get_overlap_size(
    items_set: list,
    items_test: list,
    fraction: bool = False,
    perc: bool = False,
    by: str = None
) → float
```

Percentage Jaccard index. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`fraction`</b> (bool, optional):  output fraction. Defaults to False. 
 - <b>`perc`</b> (bool, optional):  output percentage. Defaults to False. 
 - <b>`by`</b> (str, optional):  fraction by. Defaults to None. 



**Returns:**
 
 - <b>`float`</b>:  overlap size. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_item_set_size_by_background`

```python
get_item_set_size_by_background(items_set: list, background: int) → float
```

Item set size by background 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`background`</b> (int):  background size 



**Returns:**
 
 - <b>`float`</b>:  Item set size by background 



**Notes:**

> Denominator of the fold change. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_fold_change`

```python
get_fold_change(items_set: list, items_test: list, background: int) → float
```

Get fold change. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`background`</b> (int):  background size 



**Returns:**
 
 - <b>`float`</b>:  fold change 



**Notes:**

> 
>fc = (intersection/(test items))/((items in the item set)/background) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_hypergeom_pval`

```python
get_hypergeom_pval(items_set: list, items_test: list, background: int) → float
```

Calculate hypergeometric P-value. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`background`</b> (int):  background size 



**Returns:**
 
 - <b>`float`</b>:  hypergeometric P-value 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_contigency_table`

```python
get_contigency_table(items_set: list, items_test: list, background: int) → list
```

Get a contingency table required for the Fisher's test. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`background`</b> (int):  background size 



**Returns:**
 
 - <b>`list`</b>:  contingency table 



**Notes:**

> 
>within item (/referenece) set: True            False within test item: True  intersection    True False False   False False     total-size of union 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_odds_ratio`

```python
get_odds_ratio(items_set: list, items_test: list, background: int) → float
```

Calculate Odds ratio and P-values using Fisher's exact test. 



**Args:**
 
 - <b>`items_set`</b> (list):  items in the reference set 
 - <b>`items_test`</b> (list):  items to test 
 - <b>`background`</b> (int):  background size 



**Returns:**
 
 - <b>`float`</b>:  Odds ratio 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/sets.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_enrichment`

```python
get_enrichment(
    df1: DataFrame,
    df2: DataFrame,
    colid: str,
    colset: str,
    background: int,
    coltest: str = None,
    test_type: list = None,
    verbose: bool = False
) → DataFrame
```

Calculate the enrichments. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  table containing items to test 
 - <b>`df2`</b> (pd.DataFrame):  table containing refence sets and items 
 - <b>`colid`</b> (str):  column with IDs of items 
 - <b>`colset`</b> (str):  column sets 
 - <b>`coltest`</b> (str):  column tests 
 - <b>`background`</b> (int):  background size. 
 - <b>`test_type`</b> (list):  hypergeom or Fisher. Defaults to both. 
 - <b>`verbose`</b> (bool):  verbose 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output table 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.ds`
For wrappers around pandas Series plotting attributes. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ds.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `hist`

```python
hist(ds: Series, ax: Axes = None, kws_set_label_n={}, **kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.blends`
Blends of plotting functions. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/blends.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_ranks`

```python
plot_ranks(
    data: DataFrame,
    kws_plot: dict,
    col: str,
    colid: str,
    col_label: str = None,
    xlim_min: float = -20,
    ax=None
)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.colors`
For setting up colors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `rgbfloat2int`

```python
rgbfloat2int(rgb_float)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_colors_default`

```python
get_colors_default() → list
```

get default colors. 



**Returns:**
 
 - <b>`list`</b>:  colors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/colors.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`color`</b> (str, optional):  color. Defaults to '#D3DDDC'. 
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.diagram`
For diagrams e.g. flowcharts 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/diagram.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `diagram_nb`

```python
diagram_nb(
    graph: str,
    counts: dict = None,
    out: bool = False,
    test: bool = False
)
```

Show a diagram in jupyter notebook using mermaid.js. 



**Parameters:**
 
 - <b>`graph`</b> (str):  markdown-formatted graph. Please see https://mermaid.js.org/intro/n00b-syntaxReference.html 
 - <b>`out`</b> (bool):  Output the URL. Defaults to False. 

References: 
 - <b>`1. https`</b>: //mermaid.js.org/config/Tutorials.html#jupyter-integration-with-mermaid-js 



**Examples:**
 

graph LR;  i1(["input1"]) & d1[("data1")] 
        -->  p1[["process1"]] 
                --> o1(["output1"])  p1 
                --> o2["output2"]:::ends classDef ends fill:#fff,stroke:#fff 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow`




**Global Variables**
---------------
- **io**
- **log**
- **task**
- **nb**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.global_imports`
For importing commonly used functions at the development phase. 

Requirements:  

 pip install roux[all]  

Usage: in interactive sessions (e.g. in jupyter notebooks) to facilitate faster code development. 

Note: Post-development, to remove *s from the code, use removestar (pip install removestar). 

 removestar file 



<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.annot`
For annotations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    scatter_marker='|',
    scatter_alpha=0.75,
    lines=True,
    offx3: float = 0.15,
    offymin: float = 0.1,
    offymax: float = 0.9,
    length_axhline: float = 3,
    text=True,
    text_offx: float = 0,
    text_offy: float = 0,
    invert_xaxis: bool = False,
    break_pt: int = 25,
    va: str = 'bottom',
    zorder: int = 2,
    color: str = 'gray',
    kws_line: dict = {},
    kws_scatter: dict = {},
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `annot_side_curved`

```python
annot_side_curved(
    data,
    colx: str,
    coly: str,
    col_label: str,
    off: float = 0.5,
    lim: tuple = None,
    limf: tuple = None,
    loc: str = 'right',
    ax=None,
    test: bool = False,
    kws_text={},
    **kws_line
)
```

Annot elements of the plots on the of the side plot using bezier lines. 

Usage:   1. Allows m:1 mappings between points and labels 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_outlines`

```python
show_outlines(
    data: DataFrame,
    colx: str,
    coly: str,
    column_outlines: str,
    outline_colors: dict,
    style=None,
    legend: bool = True,
    kws_legend: dict = {},
    zorder: int = 3,
    ax: Axes = None,
    **kws_scatter
) → Axes
```

Outline points on the scatter plot by categories. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L388"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_confidence_ellipse`

```python
show_confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs)
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L456"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    edgecolor: str = 'k',
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
 - <b>`edgecolor`</b> (str, optional):  edge color. Defaults to 'k'. 
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L513"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_n_legend`

```python
show_n_legend(ax, df1: DataFrame, colid: str, colgroup: str, **kws)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_scatter_stats`

```python
show_scatter_stats(
    ax: Axes,
    data: DataFrame,
    x,
    y,
    z,
    method: str,
    resample: bool = False,
    show_n: bool = True,
    show_n_prefix: str = '',
    prefix: str = '',
    loc=None,
    zorder: int = 5,
    verbose: bool = True,
    kws_stat={},
    **kws_set_label
)
```

resample (bool, optional): resample data. Defaults to False. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_crosstab_stats`

```python
show_crosstab_stats(
    data: DataFrame,
    cols: list,
    method: str = None,
    alpha: float = 0.05,
    loc: str = None,
    xoff: float = 0,
    yoff: float = 0,
    linebreak: bool = False,
    ax: Axes = None,
    **kws_set_label
) → Axes
```

Annotate a confusion matrix. 



**Args:**
 
 - <b>`data`</b> (pd.DataFrame):  input data. 
 - <b>`cols`</b> (list):  list of columns with the categories. 
 - <b>`method`</b> (str, optional):  method used to calculate the statistical significance. 
 - <b>`alpha`</b> (float, optional):  alpha for the stats. Defaults to 0.05. 
 - <b>`loc`</b> (str, optional):  location. Over-rides kws_set_label. Defaults to None. 
 - <b>`xoff`</b> (float, optional):  x offset. Defaults to 0. 
 - <b>`yoff`</b> (float, optional):  y offset. Defaults to 0. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws_set_label`</b>:  keyword parameters provided to `set_label`. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `show_confusion_matrix_stats`

```python
show_confusion_matrix_stats(
    df_: DataFrame,
    ax: Axes = None,
    off: float = 0.5
) → Axes
```

Annotate a confusion matrix. 



**Args:**
 
 - <b>`df_`</b> (pd.DataFrame):  input data. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`off`</b> (float, optional):  offset. Defaults to 0.5. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/annot.py#L863"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_suptitle`

```python
set_suptitle(axs, title, offy=0, **kws_text)
```

Combined title for a list of subplots. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/vizi"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.vizi`






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.set`
For processing list-like sets. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_non_overlaps_with`

```python
check_non_overlaps_with(l1: list, l2: list, out_count: bool = False, log=True)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_overlaps_with`

```python
validate_overlaps_with(l1, l2, **kws_check)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_overlaps_with`

```python
assert_overlaps_with(l1, l2, out_count=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `jaccard_index`

```python
jaccard_index(l1, l2)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `unique_sorted`

```python
unique_sorted(l)
```

Unique items in a list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 



**Notes:**

> The function can return list of lists if used in `pandas.core.groupby.DataFrameGroupBy.agg` context. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list2str`

```python
list2str(x, fmt=None, ignore=False)
```

Returns string if single item in a list. 



**Parameters:**
 
 - <b>`x`</b> (list):  list 



**Returns:**
 
 - <b>`s`</b> (str):  string. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `lists2str`

```python
lists2str(ds: DataFrame, **kws_list2str) → str
```

Combining lists with ids to to unified string 

Usage:  `pandas` aggregation functions. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

TODOs: 1. feed as an estimator to `df.corr()`. 2. faster processing by filling up the symetric half of the adjacency matrix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L331"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list2ranges`

```python
list2ranges(l)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/set.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_pairs`

```python
get_pairs(
    items: list,
    items_with: list = None,
    size: int = 2,
    with_self: bool = False,
    unique: bool = False
) → DataFrame
```

Creates a dataframe with the paired items. 



**Parameters:**
 
 - <b>`items`</b>:  the list of items to pair. 
 - <b>`items_with`</b>:  list of items to pair with. 
 - <b>`size`</b>:  size of the combinations. 
 - <b>`with_self`</b>:  pair with self or not. 
 - <b>`unique`</b> (bool):  get unique pairs (defaults to False). 



**Returns:**
 table with pairs of items. 



**Notes:**

> 1. the ids of the items are sorted e.g. 'a'-'b' not 'b'-'a'. 2. itertools.combinations does not pair self. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.solve`
For solving equations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/solve.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.preprocess`
For classification. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dropna_matrix`

```python
dropna_matrix(
    df1,
    coff_cols_min_perc_na=5,
    coff_rows_min_perc_na=5,
    test=False,
    verbose=False
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cols_x_for_comparison`

```python
get_cols_x_for_comparison(
    df1: DataFrame,
    cols_y: list,
    cols_index: list,
    cols_drop: list = [],
    cols_dropby_patterns: list = [],
    dropby_low_complexity: bool = True,
    min_nunique: int = 5,
    max_inflation: int = 50,
    dropby_collinearity: bool = True,
    coff_rs: float = 0.7,
    dropby_variance_inflation: bool = True,
    verbose: bool = False,
    test: bool = False
) → dict
```

Identify X columns. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`cols_y`</b> (list):  y columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Preprocess data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input table. 
 - <b>`colindex`</b> (str):  column containing items. 
 - <b>`colsample`</b> (str):  column containing samples. 
 - <b>`coff_samples_min`</b> (int):  minimum number of samples. 
 - <b>`colsubset`</b> (str):  column containing subsets. 
 - <b>`coff_subsets_min`</b> (int):  minimum number of subsets. Defaults to 2. 



**Returns:**
 pd.DataFrame 



**Examples:**
 

**Parameters:**
  colindex='genes id',  colsample='sample id',  coff_samples_min=3,  colsubset= 'pLOF or WT'  coff_subsets_min=2, 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/preprocess.py#L405"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_cvsplits`

```python
get_cvsplits(
    X: <built-in function array>,
    y: <built-in function array> = None,
    cv: int = 5,
    random_state: int = None,
    outtest: bool = True
) → dict
```

Get cross-validation splits. A friendly wrapper around `sklearn.model_selection.KFold`. 



**Args:**
 
 - <b>`X`</b> (np.array):  X matrix. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`cv`</b> (int, optional):  cross validations. Defaults to 5. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to None. 
 - <b>`outtest`</b> (bool, optional):  output test data. Defaults to True. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.io`
For input/output of stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/io.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `perc_label`

```python
perc_label(a, b=None, bracket=True)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/io.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.task`
For task management. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/task.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_params`

```python
validate_params(d: dict) → bool
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/task.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_task`

```python
run_task(
    parameters: dict,
    input_notebook_path: str,
    kernel: str = None,
    output_notebook_path: str = None,
    start_timeout: int = 480,
    verbose=False,
    force=False,
    **kws_papermill
) → str
```

Run a single task. 

Prameters:  parameters (dict): parameters including `output_path`s.  input_notebook_path (dict): path to the input notebook which is parameterized.  kernel (str): kernel to be used.  output_notebook_path: path to the output notebook which is used as a report.  verbose (bool): verbose. 

Keyword parameters:  kws_papermill: parameters provided to the `pm.execute_notebook` function. 



**Returns:**
  Output path. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/task.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `apply_run_task`

```python
apply_run_task(
    x: str,
    input_notebook_path: str,
    kernel: str,
    force=False,
    **kws_papermill
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/task.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `run_tasks`

```python
run_tasks(
    input_notebook_path: str,
    kernel: str = None,
    inputs: list = None,
    output_path_base: str = None,
    parameters_list=None,
    fast: bool = False,
    fast_workers: int = 6,
    to_filter_nbby_patterns_kws=None,
    input_notebook_temp_path=None,
    out_paths: bool = True,
    test1: bool = False,
    force: bool = False,
    test: bool = False,
    verbose: bool = False,
    **kws_papermill
) → list
```

Run a list of tasks. 

Prameters:  input_notebook_path (dict): path to the input notebook which is parameterized.  kernel (str): kernel to be used.  inputs (list): list of parameters without the output paths, which would be inferred by encoding.  output_path_base (str): output path with a placeholder e.g. 'path/to/{KEY}/file'.  parameters_list (list): list of parameters including the output paths.  out_paths (bool): return paths of the reports (Defaults to True).  test1 (bool): test only first task in the list (Defaults to False).  fast (bool): enable parallel-processing.  fast_workers (bool): number of parallel-processes.  force (bool): overwrite the outputs.  test (bool): test-mode.  verbose (bool): verbose. 

Keyword parameters:  kws_papermill: parameters provided to the `pm.execute_notebook` function e.g. working directory (cwd=)  to_filter_nbby_patterns_kws (list): dictionary containing parameters to be provided to `to_filter_nbby_patterns` function (Defaults to None). 



**Returns:**
 
 - <b>`parameters_list`</b> (list):  list of parameters including the output paths, inferred if not provided. 

TODOs: 0. Ignore temporary parameters e.g test, verbose etc while encoding inputs. 1. Integrate with apply_on_paths for parallel processing etc. 



**Notes:**

> 1. To resolve `RuntimeError: This event loop is already running in python` from `multiprocessing`, execute import nest_asyncio nest_asyncio.apply() 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.heatmap`
For heatmaps. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/heatmap.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.paired`
For paired stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ratio_sorted`

```python
get_ratio_sorted(a: float, b: float, increase=True) → float
```

Get ratio sorted. 



**Args:**
 
 - <b>`a`</b> (float):  value #1. 
 - <b>`b`</b> (float):  value #2. 
 - <b>`increase`</b> (bool, optional):  check for increase. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `diff`

```python
diff(a: float, b: float, absolute=True) → float
```

Get difference 



**Args:**
 
 - <b>`a`</b> (float):  value #1. 
 - <b>`b`</b> (float):  value #2. 
 - <b>`absolute`</b> (bool, optional):  get absolute difference. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_diff_sorted`

```python
get_diff_sorted(a: float, b: float) → float
```

Difference sorted/absolute. 



**Args:**
 
 - <b>`a`</b> (float):  value #1. 
 - <b>`b`</b> (float):  value #2. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `balance`

```python
balance(a: float, b: float, absolute=True) → float
```

Balance. 



**Args:**
 
 - <b>`a`</b> (float):  value #1. 
 - <b>`b`</b> (float):  value #2. 
 - <b>`absolute`</b> (bool, optional):  absolute difference. Defaults to True. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_paired_sets_stats`

```python
get_paired_sets_stats(l1: list, l2: list, test: bool = False) → list
```

Paired stats comparing two sets. 



**Args:**
 
 - <b>`l1`</b> (list):  set #1. 
 - <b>`l2`</b> (list):  set #2. 
 - <b>`test`</b> (bool):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`list`</b>:  tuple (overlap, intersection, union, ratio). 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/paired.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.variance`
For variance related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/variance.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/variance.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ci`

```python
get_ci(rs, ci_type, outstr=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/variance.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_variance_inflation`

```python
get_variance_inflation(data, coly: str, cols_x: list = None)
```

Variance Inflation Factor (VIF). A wrapper around `statsmodels`'s '`variance_inflation_factor` function. 



**Parameters:**
 
 - <b>`data`</b> (pd.DataFrame):  input data. 
 - <b>`coly`</b> (str):  dependent variable. 
 - <b>`cols_x`</b> (list):  independent variables. 



**Returns:**
 pd.Series 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.norm`
For normalisation. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_norm`

```python
to_norm(x, off=1e-05)
```

Normalise a vector bounded between 0 and 1. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `norm_by_quantile`

```python
norm_by_quantile(X: <built-in function array>) → <built-in function array>
```

Quantile normalize the columns of X. 

Params:  X : 2D array of float, shape (M, N). The input data, with M rows (genes/features) and N columns (samples). 



**Returns:**
 
 - <b>`Xn `</b>:  2D array of float, shape (M, N). The normalized data. 



**Notes:**

> Faster processing (~5 times compared to other function tested) because of the use of numpy arrays. 
>TODOs: Use `from sklearn.preprocessing import QuantileTransformer` with `output_distribution` parameter allowing rescaling back to the same distribution kind. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `zscore`

```python
zscore(df: DataFrame, cols: list = None) → DataFrame
```

Z-score. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input table. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output table. 

TODOs: 1. Use scipy or sklearn's zscore because of it's additional options  from scipy.stats import zscore  df.apply(zscore) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/norm.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `norm_covariance_PCA`

```python
norm_covariance_PCA(
    X: <built-in function array>,
    use_svd: bool = True,
    use_sklearn: bool = True,
    rescale_centered: bool = True,
    random_state: int = 0,
    test: bool = False,
    verbose: bool = False
) → <built-in function array>
```

Covariance normalization by PCA whitening. 



**Args:**
 
 - <b>`X`</b> (np.array):  input array 
 - <b>`use_svd`</b> (bool, optional):  use SVD method. Defaults to True. 
 - <b>`use_sklearn`</b> (bool, optional):  use `skelearn` for SVD method. Defaults to True. 
 - <b>`rescale_centered`</b> (bool, optional):  rescale to centered input. Defaults to True. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to 0. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`verbose`</b> (bool, optional):  verbose. Defaults to False. 



**Returns:**
 
 - <b>`np.array`</b>:  transformed data. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.diff`
For difference related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_classes`

```python
compare_classes(x, y, method=None)
```

Compare classes 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_classes_many`

```python
compare_classes_many(df1: DataFrame, cols_y: list, cols_x: list) → DataFrame
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    func=None
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
 - <b>`func`</b> (function, optional):  function. Defaults to None. 



**Raises:**
 
 - <b>`ArgumentError`</b>:  colvalue or colsubset not found in df. 
 - <b>`ValueError`</b>:  need only 2 subsets. 



**Returns:**
 
 - <b>`tuple`</b>:  stat,p-value 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    stats=['mean', 'median', 'var', 'size'],
    coff_samples_min=None,
    verb=False,
    func=None,
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    stats=['mean', 'median', 'var', 'size'],
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L358"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_significant_changes`

```python
get_significant_changes(
    df1: DataFrame,
    coff_p=0.025,
    coff_q=0.1,
    alpha=None,
    change_type=['diff', 'ratio'],
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L478"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    func=None,
    **kws
) → DataFrame
```

Wrapper around the `get_stats_groupby` 

Keyword parameters:  cols=['variable x','variable y'],  coff_p=0.05,  coff_q=0.01,  colindex=['id'], 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/diff.py#L581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.df`
For management of tables. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/df.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.dict`
For processing dictionaries. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `head_dict`

```python
head_dict(d, lines=5)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dicts_deep`

```python
merge_dicts_deep(left: dict, right: dict) → dict
```

Merge nested dictionaries. Overwrites left with right. 



**Parameters:**
 
 - <b>`left`</b> (dict):  dictionary #1 
 - <b>`right`</b> (dict):  dictionary #2 

TODOs: 1. In python>=3.9, `merged = d1 | d2`? 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dict.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.nb`
For operations on jupyter notebooks. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `read_nb_md`

```python
read_nb_md(p: str, n: int = None) → list
```

Read notebook's documentation in the markdown cells. 



**Args:**
 
 - <b>`p`</b> (str):  path of the notebook. 
 - <b>`n`</b> (int):  number of the markdown cells to extract. 



**Returns:**
 
 - <b>`list`</b>:  lines of the strings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_info`

```python
to_info(p: str, outp: str, linkd: str = '') → str
```

Save README.md file with table of contents obtained from jupyter notebooks. 



**Args:**
 
 - <b>`p`</b> (str, optional):  path of the notebook files that would be converted to "tasks". 
 - <b>`outp`</b> (str, optional):  path of the output file, e.g. 'README.md'. 



**Returns:**
 
 - <b>`str`</b>:  path of the output file. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_replaced_nb`

```python
to_replaced_nb(
    nb_path,
    output_path,
    replaces: dict = {},
    cell_type: str = 'code',
    drop_lines_with_substrings: list = None,
    test=False
)
```

Replace text in a jupyter notebook. 

Parameters  nb: notebook object obtained from `nbformat.reads`.  replaces (dict): mapping of text to 'replace from' to the one to 'replace with'.  cell_type (str): the type of the cell. 



**Returns:**
 
 - <b>`new_nb`</b>:  notebook object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_filtered_nb`

```python
to_filtered_nb(
    p: str,
    outp: str,
    header: str,
    kind: str = 'include',
    validate_diff: int = None
)
```

Filter sections in a notebook based on markdown headings. 



**Args:**
 
 - <b>`header`</b> (str):  exact first line of a markdown cell marking a section in a notebook. validate_diff 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_filter_nbby_patterns`

```python
to_filter_nbby_patterns(p, outp, patterns=None, **kws)
```

Filter out notebook cells if the pattern string is found. 



**Args:**
 
 - <b>`patterns`</b> (list):  list of string patterns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_clear_unused_cells`

```python
to_clear_unused_cells(
    notebook_path,
    new_notebook_path,
    validate_diff: int = None
)
```

Remove code cells with all lines commented. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_clear_outputs`

```python
to_clear_outputs(notebook_path, new_notebook_path)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/nb.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_filtered_outputs`

```python
to_filtered_outputs(input_path, output_path, warnings=True, strings=True)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.sets`
For plotting sets. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/sets.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/sets.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_intersection_counts`

```python
plot_intersection_counts(
    df1: DataFrame,
    cols: list = None,
    kind: str = 'table',
    method: str = None,
    show_counts: bool = True,
    show_pval: bool = True,
    confusion: bool = False,
    rename_cols: bool = False,
    sort_cols: tuple = [True, True],
    order_x: list = None,
    order_y: list = None,
    cmap: str = 'Reds',
    ax: Axes = None,
    kws_show_stats: dict = {},
    **kws_plot
) → Axes
```

Plot counts for the intersection between two sets. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data 
 - <b>`cols`</b> (list, optional):  columns. Defaults to None. 
 - <b>`kind`</b> (str, optional):  kind of plot: table or barplot. Detaults to table. 
 - <b>`method`</b> (str, optional):  method to check the association ['chi2','FE']. Defaults to None. 
 - <b>`rename_cols`</b> (bool, optional):  rename the columns. Defaults to True. 
 - <b>`show_pval`</b> (bool, optional):  annotate p-values. Defaults to True. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 
 - <b>`kws_show_stats`</b> (dict, optional):  arguments provided to stats function. Defaults to {}. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Raises:**
 
 - <b>`ValueError`</b>:  `show_pval` position should be the allowed one. 

Keyword Args: 
 - <b>`kws_plot`</b>:  keyword arguments provided to the plotting function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: 1. Use `compare_classes` to get the stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/sets.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/sets.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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




---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/sets.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_pie`

```python
plot_pie(
    counts: list,
    labels: list,
    scales_line_xy: tuple = (1.1, 1.1),
    remove_wedges: list = None,
    remove_wedges_index: list = [],
    line_color: str = 'k',
    annot_side: bool = False,
    kws_annot_side: dict = {},
    ax: Axes = None,
    **kws_pie
) → Axes
```

Pie plot. 



**Args:**
 
 - <b>`counts`</b> (list):  counts. 
 - <b>`labels`</b> (list):  labels. 
 - <b>`scales_line_xy`</b> (tuple, optional):  scales for the lines. Defaults to (1.1,1.1). 
 - <b>`remove_wedges`</b> (list, optional):  remove wedge/s. Defaults to None. 
 - <b>`remove_wedges_index`</b> (list, optional):  remove wedge/s by index. Defaults to []. 
 - <b>`line_color`</b> (str, optional):  line color. Defaults to 'k'. 
 - <b>`annot_side`</b> (bool, optional):  annotations on side using the `annot_side` function. Defaults to False. 
 - <b>`kws_annot_side`</b> (dict, optional):  keyword arguments provided to the `annot_side` function. Defaults to {}. 
 - <b>`ax`</b> (plt.Axes, optional):  subplot. Defaults to None. 

Keyword Args: 
 - <b>`kws_pie`</b>:  keyword arguments provided to the `pie` chart function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  subplot 

References: 
 - <b>`https`</b>: //matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.compare`
For comparison related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/compare.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/compare.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

TODOs: 1. Add option for semantic similarity. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.dfs`
For processing multiple pandas DataFrames/Series 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dfs.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `filter_dfs`

```python
filter_dfs(dfs: list, cols: list, how: str = 'inner') → DataFrame
```

Filter dataframes based items in the common columns. 



**Parameters:**
 
 - <b>`dfs`</b> (list):  list of dataframes. 
 - <b>`cols`</b> (list):  list of columns. 
 - <b>`how`</b> (str):  how to filter ('inner') 

Returns 
 - <b>`dfs`</b> (list):  list of dataframes. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dfs.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dfs.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_paired`

```python
merge_paired(
    df1: DataFrame,
    df2: DataFrame,
    left_ons: list,
    right_on: list,
    common: list = [],
    right_ons_common: list = [],
    how: str = 'inner',
    validates: list = ['1:1', '1:1'],
    suffixes: list = None,
    test: bool = False,
    verb: bool = True,
    **kws
) → DataFrame
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dfs.py#L221"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `merge_dfs`

```python
merge_dfs(dfs: list, **kws) → DataFrame
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/dfs.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compare_rows`

```python
compare_rows(df1, df2, test=False, **kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.scatter`
For scatter plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/scatter.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_scatter_agg`

```python
plot_scatter_agg(
    dplot: DataFrame,
    x: str = None,
    y: str = None,
    z: str = None,
    kws_legend={'bbox_to_anchor': [1, 1], 'loc': 'upper left'},
    title=None,
    label_colorbar=None,
    ax=None,
    kind=None,
    verbose=False,
    cmap='Blues',
    gridsize=10,
    **kws
)
```

UNDER DEV. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/scatter.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_scatter`

```python
plot_scatter(
    data: DataFrame,
    x: str = None,
    y: str = None,
    z: str = None,
    kind: str = 'scatter',
    scatter_kws={},
    line_kws={},
    stat_method: str = 'spearman',
    stat_kws={},
    hollow: bool = False,
    ax: Axes = None,
    verbose: bool = True,
    **kws
) → Axes
```

Plot scatter with multiple layers and stats. 



**Args:**
 
 - <b>`data`</b> (pd.DataFrame):  input dataframe. 
 - <b>`x`</b> (str):  x column. 
 - <b>`y`</b> (str):  y column. 
 - <b>`z`</b> (str, optional):  z column. Defaults to None. 
 - <b>`kind`</b> (str, optional):  kind of scatter. Defaults to 'hexbin'. 
 - <b>`trendline_method`</b> (str, optional):  trendline method ['poly','lowess']. Defaults to 'poly'. 
 - <b>`stat_method`</b> (str, optional):  method of annoted stats ['mlr',"spearman"]. Defaults to "spearman". 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Reds'. 
 - <b>`label_colorbar`</b> (str, optional):  label of the colorbar. Defaults to None. 
 - <b>`gridsize`</b> (int, optional):  number of grids in the hexbin. Defaults to 25. 
 - <b>`bbox_to_anchor`</b> (list, optional):  location of the legend. Defaults to [1,1]. 
 - <b>`loc`</b> (str, optional):  location of the legend. Defaults to 'upper left'. 
 - <b>`title`</b> (str, optional):  title of the plot. Defaults to None. 
 - <b>`#params_plot (dict, optional)`</b>:  parameters provided to the `plot` function. Defaults to {}. 
 - <b>`line_kws`</b> (dict, optional):  parameters provided to the `plot_trendline` function. Defaults to {}. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 

Keyword Args: 
 - <b>`kws`</b>:  parameters provided to the `plot` function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 



**Notes:**

> 1. For a rasterized scatter plot set `scatter_kws={'rasterized': True}` 2. This function does not apply multiple colors, similar to `sns.regplot`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/scatter.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/scatter.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_ranks`

```python
plot_ranks(
    df1: DataFrame,
    col: str,
    colid: str,
    ranks_on: str = 'y',
    ascending: bool = True,
    col_rank: str = None,
    line: bool = True,
    kws_line={},
    show_topn: int = None,
    show_ids: list = None,
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

Usage: Combined with annotations using `annot_side`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/scatter.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_volcano`

```python
plot_volcano(
    data: DataFrame,
    colx: str,
    coly: str,
    colindex: str,
    hue: str = 'x',
    style: str = 'P=0',
    style_order: list = ['o', '^'],
    markers: list = ['o', '^'],
    show_labels: int = None,
    labels_layout: str = None,
    labels_kws: dict = {},
    show_outlines: int = None,
    outline_colors: list = ['k'],
    collabel: str = None,
    show_line=True,
    line_pvalue=0.1,
    line_x: float = 0.0,
    line_x_min: float = None,
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

Volcano plot. 



**Parameters:**
 

Keyword parameters: 



**Returns:**
  plt.Axes 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.run`
For access to a few functions from the terminal. 



<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.str`
For processing strings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `filter_list`

```python
filter_list(l: list, patterns: list, kind='out') → list
```

Filter a list of strings. 



**Args:**
 
 - <b>`l`</b> (list):  list of strings. 
 - <b>`patterns`</b> (list):  list of regex patterns. patterns are applied after stripping the whitespaces. 



**Returns:**
 (list) list of filtered strings. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 
 - <b>`s1`</b> (str):  string #1. 
 - <b>`s2`</b> (str):  string #2. 
 - <b>`prefix`</b> (str):  prefix. 
 - <b>`suffix`</b> (str):  suffix. 
 - <b>`common`</b> (str):  common substring. 



**Returns:**
 
 - <b>`l`</b> (list):  output list. 



**Notes:**

> 1. Code to test: [ get_prefix(source,target,common=False), get_prefix(source,target,common=True), get_suffix(source,target,common=False), get_suffix(source,target,common=True),] 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_prefix`

```python
get_prefix(s1, s2: str = None, common: bool = True, clean: bool = True) → str
```

Get the prefix of the strings 



**Parameters:**
 
 - <b>`s1`</b> (str|list):  1st string. 
 - <b>`s2`</b> (str):  2nd string (default:None). 
 - <b>`common`</b> (bool):  get the common prefix (default:True). 
 - <b>`clean`</b> (bool):  clean the leading and trailing whitespaces (default:True). 



**Returns:**
 
 - <b>`s`</b> (str):  prefix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L413"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_suffix`

```python
get_suffix(s1, s2: str = None, common: bool = True, clean: bool = True) → str
```

Get the suffix of the strings 



**Parameters:**
 
 - <b>`s1`</b> (str|list):  1st string. 
 - <b>`s2`</b> (str):  2nd string (default:None). 
 - <b>`common`</b> (bool):  get the common prefix (default:True). 
 - <b>`clean`</b> (bool):  clean the leading and trailing whitespaces (default:True). 



**Returns:**
 
 - <b>`s`</b> (str):  prefix. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `removesuffix`

```python
removesuffix(s1: str, suffix: str) → str
```

Remove suffix. 

Paramters:  s1 (str): input string.  suffix (str): suffix. 



**Returns:**
 
 - <b>`s1`</b> (str):  string without the suffix. 

TODOs: 1. Deprecate in py>39 use .removesuffix() instead. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L484"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L512"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L537"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L599"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L646"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `decode`

```python
decode(s, out=None, **kws_out)
```

Decode data from a string. 



**Parameters:**
 
 - <b>`s`</b> (string):  encoded string. 
 - <b>`out`</b> (str):  output format (dict|df). 

Keyword parameters: 
 - <b>`kws_out`</b>:  parameters provided to `dict2df`. 



**Returns:**
 
 - <b>`d`</b> (dict|DataFrame):  output data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/str.py#L677"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_formula`

```python
to_formula(
    replaces={' ': 'SPACE', '(': 'LEFTBRACKET', ')': 'RIGHTTBRACKET', '.': 'DOT', ',': 'COMMA', '%': 'PERCENT', "'": 'INVCOMMA', '+': 'PLUS', '-': 'MINUS'},
    reverse=False
) → dict
```

Converts strings to the formula format, compatible with `patsy` for example. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.workflow.monitor`
For workflow monitors. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/workflow/monitor.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.text`
For processing text files. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/text.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/text.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/vizi.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.vizi.scatter`





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/vizi/scatter.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_scatters_grouped`

```python
plot_scatters_grouped(
    data: DataFrame,
    cols_groupby: list,
    aggfunc: dict,
    orient='h',
    **kws_encode
)
```

Scatters grouped by categories. 



**Args:**
 
 - <b>`data`</b> (pd.DataFrame):  input data, 
 - <b>`cols_groupby`</b> (list):  list of colummns to groupby, 
 - <b>`aggfunc`</b> (dict):  columns mapped to the aggregation function, 

Keyword Args: 
 - <b>`kws_encode`</b>:  parameters provided to the `encode` attribute 



**Returns:**
 Altair figure 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.network`
For network related stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/network.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.google`
Processing files form google-cloud services. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object. 

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object. 

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `list_files_in_folder`

```python
list_files_in_folder(service, folderid, filetype=None, fileext=None, test=False)
```

Lists files in a google drive folder. 

:param service: service object e.g. drive :param folderid: folder id from google drive :param filetype: specify file type :param fileext: specify file extension :param test: True if verbose else False ... :return: list of files in the folder 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_file_id`

```python
get_file_id(p)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `upload_file`

```python
upload_file(service, filep, folder_id, test=False)
```

Uploads a local file onto google drive. 

:param service: google service object :param filep: path of the file :param folder_id: id of the folder on google drive where the file will be uploaded :param test: True is verbose else False ... :return: id of the uploaded file 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `upload_files`

```python
upload_files(service, ps, folder_id, **kws)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `download_drawings`

```python
download_drawings(folderid, outd, service=None, test=False)
```

Download specific files: drawings 

TODOs: 1. use download_file 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L420"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `search`

```python
search(query, results=1, service=None, **kws_search)
```

Google search. 

:param query: exact terms ... :return: dict 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L441"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_search_strings`

```python
get_search_strings(text, num=5, test=False)
```

Google search. 

:param text: string :param num: number of results :param test: True if verbose else False ... :return lines: list 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L466"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

:params user_permission: user_permission = {  'type': 'anyone',  'role': 'reader',  'email':'@' } Ref: https://developers.google.com/drive/api/v3/manage-sharing 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `slides`







---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `create_image`

```python
create_image(service, presentation_id, page_id, image_id)
```

image less than 1.5 Mb 

---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/google.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `get_page_ids`

```python
get_page_ids(service, presentation_id)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz`




**Global Variables**
---------------
- **ds**
- **theme**
- **ax_**
- **colors**
- **diagram**
- **io**


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.ax_`
For setting up subplots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_axes_minimal`

```python
set_axes_minimal(ax, xlabel=None, ylabel=None, off_axes_pad=0) → Axes
```

Set minimal axes labels, at the lower left corner. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_axes_arrows`

```python
set_axes_arrows(
    ax: Axes,
    length: float = 0.1,
    pad: float = 0.2,
    color: str = 'k',
    head_width: float = 0.03,
    head_length: float = 0.02,
    length_includes_head: bool = True,
    clip_on: bool = False,
    **kws_arrow
)
```

Set arrows next to the axis labels. 



**Parameters:**
 
 - <b>`ax`</b> (plt.Axes):  subplot. color= 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_ax_labels`

```python
get_ax_labels(ax: Axes)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_labels`

```python
format_labels(
    ax,
    axes: list = ['x', 'y'],
    fmt='cap1',
    title_fontsize=15,
    rename_labels=None,
    rotate_ylabel=True,
    y=1.05,
    test=False
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_ticklabels`

```python
format_ticklabels(
    ax: Axes,
    axes: tuple = ['x', 'y'],
    interval: float = None,
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
 - <b>`fmt`</b> (str, optional):  format e.g. '.0f'. Defaults to None. 
 - <b>`font`</b> (str, optional):  font. Defaults to 'DejaVu Sans Mono'. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 

TODOs: 1. include color_ticklabels 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `split_ticklabels`

```python
split_ticklabels(
    ax: Axes,
    fmt: str,
    axis='x',
    group_x=-0.45,
    group_y=-0.25,
    group_prefix=None,
    group_suffix=False,
    group_loc='center',
    group_colors=None,
    group_alpha=0.2,
    show_group_line=True,
    group_line_off_x=0.15,
    group_line_off_y=0.1,
    show_group_span=False,
    group_span_kws={},
    sep: str = '-',
    pad_major=6,
    off: float = 0.2,
    test: bool = False,
    **kws
) → Axes
```

Split ticklabels into major and minor. Two minor ticks are created per major tick. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`fmt`</b> (str):  'group'-wise or 'pair'-wise splitting of the ticklabels. 
 - <b>`axis`</b> (str):  name of the axis: x or y. 
 - <b>`sep`</b> (str, optional):  separator within the tick labels. Defaults to ' '. 
 - <b>`test`</b> (bool, optional):  test-mode. Defaults to False. 

**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L619"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L652"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L671"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L708"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_axlims`

```python
set_axlims(
    ax: Axes,
    off: float,
    axes: list = ['x', 'y'],
    equal=False,
    **kws_set_equallim
) → Axes
```

Set axis limits. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`off`</b> (float):  offset. 
 - <b>`axes`</b> (list, optional):  axis name/s. Defaults to ['x','y']. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L742"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_grids`

```python
set_grids(ax: Axes, axis: str = None) → Axes
```

Show grids based on the shape (aspect ratio) of the plot. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 
 - <b>`axis`</b> (str, optional):  axis name. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L763"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_legends`

```python
format_legends(ax: Axes, **kws_legend) → Axes
```

Format legend text. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L803"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L836"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L853"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L883"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `drop_duplicate_legend`

```python
drop_duplicate_legend(ax, **kws)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L887"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L903"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `set_legends_merged`

```python
set_legends_merged(axs, **kws_legend)
```

Reset legend colors. 



**Args:**
 
 - <b>`axs`</b> (list):  list of `plt.Axes` objects. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  first `plt.Axes` object in the list. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L931"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L1013"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L1032"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L1077"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/ax_.py#L1096"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `format_ax`

```python
format_ax(
    ax=None,
    kws_fmt_ticklabels={},
    kws_fmt_labels={},
    kws_legend={},
    rotate_ylabel=False
)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.io`
For input/output of plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    verbose: bool = False,
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L216"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L241"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_script`

```python
to_script(
    srcp: str,
    plotp: str,
    defn: str = 'plot_',
    s4: str = '    ',
    test: bool = False,
    validate: bool = False,
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L571"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L663"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`1. https`</b>: //pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif 
 - <b>`2. https`</b>: //stackoverflow.com/a/57751793/3521099 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L703"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L721"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L743"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/io.py#L792"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_rasters`

```python
to_rasters(plotd, ext='svg')
```

Convert many images to raster. Uses inkscape. 



**Args:**
 
 - <b>`plotd`</b> (str):  directory. 
 - <b>`ext`</b> (str, optional):  extension of the output. Defaults to 'svg'. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.corr`
For correlation stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/corr.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `resampled`

```python
resampled(
    x: <built-in function array>,
    y: <built-in function array>,
    method_fun: object,
    method_kws: dict = {},
    ci_type: str = 'max',
    cv: int = 5,
    random_state: int = 1,
    verbose: bool = False
) → tuple
```

Get correlations after resampling. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`method_fun`</b> (str, optional):  method function. 
 - <b>`ci_type`</b> (str, optional):  confidence interval type. Defaults to 'max'. 
 - <b>`cv`</b> (int, optional):  number of resamples. Defaults to 5. 
 - <b>`random_state`</b> (int, optional):  random state. Defaults to 1. 
 - <b>`verbose`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`dict`</b>:  results containing mean correlation coefficient, CI and CI type. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/corr.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_corr`

```python
get_corr(
    x: str,
    y: str,
    method: str,
    df: DataFrame = None,
    method_kws: dict = {},
    pval: bool = True,
    preprocess: bool = True,
    n_min=10,
    preprocess_kws: dict = {},
    resample: bool = False,
    cv=5,
    resample_kws: dict = {},
    verbose: bool = False,
    test: bool = False
) → dict
```

Correlation between vectors. A unifying wrapper around `scipy`'s functions to calculate correlations and distances. Allows application of resampling on those functions. 

Usage:  1. Linear table with paired values. For a matrix, use `pd.DataFrame.corr` instead. 



**Args:**
 
 - <b>`x`</b> (str):  x column name or a vector. 
 - <b>`y`</b> (str):  y column name or a vector. 
 - <b>`method`</b> (str):  method name. 
 - <b>`df`</b> (pd.DataFrame):  input table. 
 - <b>`pval`</b> (bool):  calculate p-value. 
 - <b>`resample`</b> (bool, optional):  resampling. Defaults to False. 
 - <b>`preprocess`</b> (bool):  preprocess the input 
 - <b>`preprocess_kws (dict) `</b>:  parameters provided to the pre-processing function i.e. `_pre`. 
 - <b>`resample`</b> (bool):  resampling. 
 - <b>`resample_kws`</b> (dict):  parameters provided to the resampling function i.e. `resample`. 
 - <b>`verbose`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`res`</b> (dict):  a dictionary containing results. 



**Notes:**

> `res` directory contains following values: 
>method : method name r : correlation coefficient or distance p : pvalue of the correlation. n : sample size rr: resampled average 'r' ci: CI ci_type: CI type 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/corr.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_corrs`

```python
get_corrs(
    data: DataFrame,
    method: str,
    cols: list = None,
    cols_with: list = None,
    coff_inflation_min: float = None,
    get_pairs_kws={},
    fast: bool = False,
    test: bool = False,
    verbose: bool = False,
    **kws_get_corr
) → DataFrame
```

Correlate many columns of a dataframes. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`method`</b> (str):  method of correlation `spearman` or `pearson`. 
 - <b>`cols`</b> (str):  columns. 
 - <b>`cols_with`</b> (str):  columns to correlate with i.e. variable2. 
 - <b>`fast`</b> (bool):  use parallel-processing if True. 

Keyword arguments: 
 - <b>`kws_get_corr`</b>:  parameters provided to `get_corr` function. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 



**Notes:**

> In the fast mode (fast=True), to set the number of processes, before executing the `get_corrs` command, run 
>from pandarallel import pandarallel pandarallel.initialize(nb_workers={},progress_bar=True,use_memory_fs=False) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/corr.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_collinearity`

```python
check_collinearity(
    df1: DataFrame,
    threshold: float = 0.7,
    colvalue: str = 'r',
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
 
 - <b>`DataFrame`</b>:  output dataframe with minimum correlation among correlated subnetwork of columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/corr.py#L425"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.line`
For line plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/line.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_range`

```python
plot_range(
    df00: DataFrame,
    colvalue: str,
    colindex: str,
    k: str,
    headsize: int = 15,
    headcolor: str = 'lightgray',
    ax: Axes = None,
    **kws_area
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

Keyword args: 
 - <b>`kws`</b>:  keyword parameters provided to `area` function. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/line.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `plot_bezier`

```python
plot_bezier(
    pt1,
    pt2,
    pt1_guide=None,
    pt2_guide=None,
    direction='h',
    off_guide=0.25,
    ax=None,
    test=False,
    **kws_line
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/line.py#L259"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/line.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.lib.df`
For processing individual pandas DataFrames/Series. Mainly used in piped operations. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_name`

```python
log_name(df1: DataFrame, **kws_get_name)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `dropby_patterns`

```python
dropby_patterns(
    df1,
    patterns=None,
    strict=False,
    test=False,
    verbose=True,
    errors='raise'
)
```

Deletes columns containing substrings i.e. patterns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`patterns`</b> (list):  list of substrings. 
 - <b>`test`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `compress`

```python
compress(df1: DataFrame, coff_categories: int = None, verbose: bool = True)
```

Compress the dataframe by converting columns containing strings/objects to categorical. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`coff_categories`</b> (int):  if the number of unique values are less than cutoff the it will be converted to categories. 
 - <b>`verbose`</b> (bool):  verbose. 



**Returns:**
 
 - <b>`df1`</b> (DataFrame):  output dataframe. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `clean_compress`

```python
clean_compress(df: DataFrame, kws_compress: dict = {}, **kws_clean)
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L380"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_na`

```python
check_na(df, subset=None, out=True, perc=False, log=True)
```

Number of missing values in columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (bool):  output, else not which can be applicable in chained operations. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L418"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_str`

```python
to_str(data, log=False)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L472"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_nunique`

```python
check_nunique(
    df: DataFrame,
    subset: list = None,
    groupby: str = None,
    perc: bool = False,
    auto=False,
    out=True,
    log=True
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_dups`

```python
check_dups(df, subset=None, perc=False, out=True)
```

Check duplicates. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`perc`</b> (bool):  output percentages. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L585"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_duplicated`

```python
check_duplicated(df, **kws)
```

Check duplicates (alias of `check_dups`) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L594"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_no_dups`

```python
validate_no_dups(df, subset=None, log: bool = True)
```

Validate that no duplicates. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L614"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `validate_no_duplicates`

```python
validate_no_duplicates(df, subset=None, **kws)
```

Validate that no duplicates (alias of `validate_no_dups`) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_no_dups`

```python
assert_no_dups(df, subset=None)
```

Assert that no duplicates 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L638"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_len`

```python
assert_len(df: DataFrame, count: int) → DataFrame
```

Validate length in pipe'd operations. 



**Example:**
  (  df  .rd.assert_len(10)  ) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_nunique`

```python
assert_nunique(df: DataFrame, col: str, count: int) → DataFrame
```

Validate unique counts in pipe'd operations. 



**Example:**
  (  df  .rd.assert_nunique('id',10)  ) 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L753"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `classify_mappings`

```python
classify_mappings(df1: DataFrame, subset, clean: bool = False) → DataFrame
```

Classify mappings between items in two columns. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`col1`</b> (str):  column #1. 
 - <b>`col2`</b> (str):  column #2. 
 - <b>`clean`</b> (str):  drop columns with the counts. 



**Returns:**
 
 - <b>`(pd.DataFrame)`</b>:  output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L806"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `check_mappings`

```python
check_mappings(df: DataFrame, subset: list = None, out=True) → DataFrame
```

Mapping between items in two columns. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (str):  format of the output. 



**Returns:**
 
 - <b>`ds`</b> (Series):  output stats. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L838"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `assert_1_1_mappings`

```python
assert_1_1_mappings(df: DataFrame, subset: list = None) → DataFrame
```

Validate that the papping between items in two columns is 1:1. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (str):  format of the output. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L859"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_mappings`

```python
get_mappings(
    df1: DataFrame,
    subset=None,
    keep='all',
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L905"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L930"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L994"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1009"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_bin_labels`

```python
get_bin_labels(bins: list, dtype: str = 'int')
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_bins`

```python
get_bins(
    df: DataFrame,
    col: str,
    bins: list,
    dtype: str = 'int',
    labels: list = None,
    **kws_cut
)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_qbins`

```python
get_qbins(df: DataFrame, col: str, bins: list, labels: list = None, **kws_qcut)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sample_near_quantiles`

```python
sample_near_quantiles(data: DataFrame, col: str, n: int, clean: bool = False)
```

Get rows with values closest to the quantiles. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_group`

```python
get_group(groups, i: int = None, verbose: bool = True) → DataFrame
```

Get a dataframe for a group out of the `groupby` object. 



**Parameters:**
 
 - <b>`groups`</b> (object):  groupby object. 
 - <b>`i`</b> (int):  index of the group. default None returns the largest group. 
 - <b>`verbose`</b> (bool):  verbose (True). 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Notes:**

> Useful for testing `groupby`. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_sample`

```python
groupby_sample(
    df: DataFrame,
    groupby: list,
    i: int = None,
    **kws_get_group
) → DataFrame
```

Samples a group (similar to .sample) 



**Parameters:**
 
 - <b>`df`</b> (pd.DataFrame):  input dataframe. 
 - <b>`groupby`</b> (list):  columns to group by. 
 - <b>`i`</b> (int):  index of the group. default None returns the largest group. 

Keyword arguments: keyword parameters provided to the `get_group` function 



**Returns:**
 pd.DataFrame 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_sort_values`

```python
groupby_sort_values(
    df: DataFrame,
    groupby: str,
    col: str,
    func: str,
    col_temp: str = 'temp',
    ascending=True,
    **kws_sort_values
) → DataFrame
```

Groupby and sort 



**Parameters:**
 
 - <b>`df`</b> (pd.DataFrame):  input dataframe. 
 - <b>`groupby`</b> (list):  columns to group by. 

Keyword arguments: keyword parameters provided to the `.sort_values` attribute 



**Returns:**
 pd.DataFrame 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1451"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `groupby_agg_nested`

```python
groupby_agg_nested(
    df1: DataFrame,
    groupby: list,
    subset: list,
    func: dict = None,
    cols_value: list = None,
    verbose: bool = False,
    **kws_agg
) → DataFrame
```

Aggregate serially from the lower level subsets to upper level ones. 



**Parameters:**
 
 - <b>`df1`</b> (pd.DataFrame):  input dataframe. 
 - <b>`groupby`</b> (list):  groupby columns i.e. list of columns to be used as ids in the output. 
 - <b>`subset`</b> (list):  nested groups i.e. subsets. 
 - <b>`func`</b> (dict):  map betweek columns with value to aggregate and the function for aggregation. 
 - <b>`cols_value`</b> (list):  columns with value to aggregate, (optional). 
 - <b>`verbose`</b> (bool):  verbose. 

Keyword arguments: 
 - <b>`kws_agg `</b>:  keyword arguments provided to pandas's `.agg` function. 



**Returns:**
 output dataframe with the aggregated values. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1656"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1675"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `to_cat`

```python
to_cat(ds1: Series, cats: list, ordered: bool = True)
```

To series containing categories. 



**Parameters:**
 
 - <b>`ds1`</b> (Series):  input series. 
 - <b>`cats`</b> (list):  categories. 
 - <b>`ordered`</b> (bool):  if the categories are ordered (True). 



**Returns:**
 
 - <b>`ds1`</b> (Series):  output series. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1695"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `astype_cat`

```python
astype_cat(df1: DataFrame, col: str, cats: list)
```






---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1708"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_valuesby_list`

```python
sort_valuesby_list(
    df1: DataFrame,
    by: str,
    cats: list,
    by_more: list = [],
    **kws
)
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1729"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1753"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1776"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1798"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `sort_columns_by_values`

```python
sort_columns_by_values(
    df: DataFrame,
    subset: list,
    suffixes: list = None,
    order: list = None,
    clean=False
) → DataFrame
```

Sort the values in columns in ascending order. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  columns. 
 - <b>`suffixes`</b> (list):  suffixes. 
 - <b>`order`</b> (list):  ordered list. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 



**Notes:**

> In the output dataframe, `sorted` means values are sorted because gene1>gene2. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1876"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_ids`

```python
make_ids(
    df: DataFrame,
    cols: list,
    ids_have_equal_length: bool,
    sep: str = '--',
    sort: bool = False
) → Series
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1916"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `make_ids_sorted`

```python
make_ids_sorted(
    df: DataFrame,
    cols: list,
    ids_have_equal_length: bool,
    sep: str = '--',
    sort: bool = False
) → Series
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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1938"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `get_alt_id`

```python
get_alt_id(s1: str, s2: str, sep: str = '--')
```

Get alternate/partner id from a paired id. 



**Parameters:**
 
 - <b>`s1`</b> (str):  joined id. 
 - <b>`s2`</b> (str):  query id. 



**Returns:**
 
 - <b>`s`</b> (str):  partner id. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1955"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L1978"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2004"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>function</kbd> `log_shape_change`

```python
log_shape_change(d1, fun='')
```

Report the changes in the shapes of a DataFrame. 



**Parameters:**
 
 - <b>`d1`</b> (dic):  dictionary containing the shapes. 
 - <b>`fun`</b> (str):  name of the function. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2079"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>class</kbd> `log`
Report (log) the changes in the shapes of the dataframe before and after an operation/s. 



**TODO:**
  Create the attribures (`attr`) using strings e.g. setattr.  import inspect  fun=inspect.currentframe().f_code.co_name 

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2089"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `check_dups`

```python
check_dups(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `check_na`

```python
check_na(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `clean`

```python
clean(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `drop`

```python
drop(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `drop_duplicates`

```python
drop_duplicates(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `dropna`

```python
dropna(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `explode`

```python
explode(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `filter_`

```python
filter_(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `filter_rows`

```python
filter_rows(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `groupby`

```python
groupby(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `join`

```python
join(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `melt`

```python
melt(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `melt_paired`

```python
melt_paired(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `merge`

```python
merge(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `pivot`

```python
pivot(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `pivot_table`

```python
pivot_table(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `query`

```python
query(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `stack`

```python
stack(**kws)
```





---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/lib/df.py#L2160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

#### <kbd>method</kbd> `unstack`

```python
unstack(**kws)
```






<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.stat.binary`
For processing binary data. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/stat/binary.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 - <b>`variable`</b>:  threshold (index), TPR, FPR, TP counts, precision, recall values: Plots: AUC ROC, TPR vs TP counts PR Specificity vs TP counts Dictionary: Thresholds from AUC, PR 

TODOs: 1. Separate the plotting functions. 


<!-- markdownlint-disable -->

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>module</kbd> `roux.viz.bar`
For bar plots. 


---

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/rraadd88/roux/blob/master/notebooks/roux/viz/bar.py#L491"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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






