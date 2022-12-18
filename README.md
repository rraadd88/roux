# `roux` : Helper functions

[![PyPI](https://img.shields.io/pypi/v/roux?style=flat-square&colorB=blue)![PyPI](https://img.shields.io/pypi/pyversions/roux?style=flat-square&colorB=blue)](https://pypi.org/project/roux)  
[![build](https://img.shields.io/github/workflow/status/rraadd88/roux/build?style=flat-square&colorB=blue)](https://github.com/rraadd88/roux/actions/workflows/build.yml)  

# Installation
    
```
pip install roux
```

# Examples


---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_global_imports.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Importing commonly used helper functions for use in jupyter notebooks for example.
<details><summary>Expand</summary>

### Usage


```python
# import helper functions
from roux.global_imports import *
```

    The history saving thread hit an unexpected error (DatabaseError('database disk image is malformed')).History will not be written to the database.



    0it [00:00, ?it/s]


    INFO:root:Python implementation: CPython
    Python version       : 3.7.13
    IPython version      : 7.34.0
    scipy     : 1.7.3
    seaborn   : 0.12.1
    numpy     : 1.21.6
    tqdm      : 4.64.1
    matplotlib: 3.5.3
    sys       : 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0]
    logging   : 0.5.1.2
    pandas    : 1.3.5
    re        : 2.2.1
    


    INFO: Pandarallel will run on 6 workers.
    INFO: Pandarallel will use standard multiprocessing data transfer (pipe) to transfer data between the main process and workers.


### Documentation
[`roux.global_imports`](https://github.com/rraadd88/roux#module-roux.global_imports)

For details on `roux.global_imports` such as which helper functions are imported, see [`detailed_roux_global_imports.ipynb`](https://github.com/rraadd88/roux/blob/master/examples/detailed_roux_global_imports.ipynb).

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_lib_df.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper attributes of pandas dataframes.
<details><summary>Expand</summary>


```python
# import helper functions
from roux.global_imports import *
```


    0it [00:00, ?it/s]


    INFO:root:pandarallel.initialize(nb_workers=4,progress_bar=True)
    WARNING:root:not found: metadata


    INFO: Pandarallel will run on 6 workers.
    INFO: Pandarallel will use Memory file system to transfer data between the main process and workers.


### Helper functions for basic data validations 


```python
## demo data
import seaborn as sns
df1=sns.load_dataset('iris')
```


```python
# .rd (roux data) attributes
## validate no missing values in the table
assert df1.rd.validate_no_na()
## validate no duplicates in the table
df1.rd.validate_no_dups()
```

    WARNING:root:duplicate rows found





    False



### Helper functions for checking duplicates in a table


```python
df1.rd.check_dups()
```

    INFO:root:duplicate rows: 1% (2/150)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>142</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
</div>



### Helper functions for logging changes in the dataframe shapes


```python
_=df1.log.drop_duplicates()
```

    INFO:root:drop_duplicates: shape changed: (150, 5)->(149, 5), width constant



```python
## within pipes
_=(df1
   .log.drop_duplicates()
   .log.check_nunique(groupby='species',subset='sepal_length')
  )
```

    INFO:root:drop_duplicates: shape changed: (150, 5)->(149, 5), width constant
    INFO:root:unique {'groupby': 'species', 'subset': 'sepal_length'}
    INFO:root:species
    setosa        15
    versicolor    21
    virginica     21
    Name: sepal_length, dtype: int64


### Helper functions to filter dataframe using a dictionary


```python
_=df1.rd.filter_rows({'species':'setosa'})
```

    INFO:root:(150, 5)
    INFO:root:(50, 5)


### Helper functions to merge tables while validating the changes in shapes  


```python
df2=df1.groupby('species').head(1)
```


```python
df1.log.merge(right=df2,
              how='inner',
              on='species',
              validate='m:1',
             validate_equal_length=True,
             # validate_no_decrease_length=True,
             ).head(1)
```

    INFO:root:merge: shape changed: (150, 5)->(150, 9), length constant





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length_x</th>
      <th>sepal_width_x</th>
      <th>petal_length_x</th>
      <th>petal_width_x</th>
      <th>species</th>
      <th>sepal_length_y</th>
      <th>sepal_width_y</th>
      <th>petal_length_y</th>
      <th>petal_width_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



#### Documentation
[`roux.lib.df`](https://github.com/rraadd88/roux#module-roux.lib.df)
[`roux.lib.dfs`](https://github.com/rraadd88/roux#module-roux.lib.dfs)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_lib_io.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for input/output.
<details><summary>Expand</summary>

### Saving and reading dictionaries

Unifies input/output functions for `.yaml`, `.json`, `.joblib` and `.pickle` files.


```python
d={'a':1,'b':2,'c':3}
d
```




    {'a': 1, 'b': 2, 'c': 3}



    The history saving thread hit an unexpected error (DatabaseError('database disk image is malformed')).History will not be written to the database.



```python
from roux.lib.io import to_dict
to_dict(d,'tests/output/data/dict.json')
```




    'tests/output/data/dict.json'




```python
from roux.lib.io import read_dict
read_dict('tests/output/data/dict.json')
```




    {'a': 1, 'b': 2, 'c': 3}



### Saving and reading tables

Unifies several of `pandas`'s input/output functions.


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')
```


```python
from roux.lib.io import to_table
to_table(df1,'tests/output/data/table.tsv')
```




    'tests/output/data/table.tsv'




```python
from roux.lib.io import read_table
read_table('tests/output/data/table.tsv')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



#### Documentation
[`roux.viz.io`](https://github.com/rraadd88/roux#module-roux.viz.io)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_lib_str.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions applicable to strings.
<details><summary>Expand</summary>


```python
# import helper functions
from roux.lib.str import encode,decode
```

### Encoding and decoding data

#### Reversible


```python
# example data
parameters=dict(
    colindex='drug id',
    colsample='sample id',
    coly='auc',
    formulas={
            f"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm',
        },
    kws_get_stats_regression=dict(
            groups='screen_id',
        ),
    colsubset='sample subset',
    variable="C(sample_subset, Treatment(reference='ref'))[T.test]",
)
```


```python
## encoding
encoded=encode(parameters)
print(encoded)
```

    eNqVj00KwjAQRq8Ssqli8QCCK6_gTiSk7WcJNkmZSbRF9OwmjYtuhSwm7_HNz0u2fjCuwyQPQnYUe2E6WYuMWdtxQOalWpnYMMLK_ECxcxY6tvl782TjoDmhV2biI06bElIlVIszQQcLFzaEGwiuxbFKZbXdip0YyVhNs_KkLILm9ExuJ62Z0A1WvtOY-5NVj6CSDawIPYHZeLeM7cnHcYlwS4BT6Y4cemgyuikX_rPU5bwP4HCV7y_fP20r



```python
## decoding
decoded=decode(encoded,out='dict')
print(decoded)
```

    {'colindex': 'drug id', 'colsample': 'sample id', 'colsubset': 'sample subset', 'coly': 'auc', 'formulas': {"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm'}, 'kws_get_stats_regression': {'groups': 'screen_id'}, 'variable': "C(sample_subset, Treatment(reference='ref'))[T.test]"}



```python
## test reversibility
assert parameters==decoded
```

#### Non-reversible


```python
## clear variables
%reset_selective -f "encoded.*"
```


```python
## encoding
encoded=encode(parameters,short=True)
print(encoded)
```

    e11fafe6bf21d3db843f8a0e4cea21bc600832b3ed738d2b09ee644ce8008e44



```python
## dictionary shuffled
from random import sample
parameters_shuffled={k:parameters[k] for k in sample(parameters.keys(), len(parameters))}
```


```python
## encoding dictionary shuffled
encoded_shuffled=encode(parameters_shuffled,short=True)
print(encoded_shuffled)
```

    e11fafe6bf21d3db843f8a0e4cea21bc600832b3ed738d2b09ee644ce8008e44



```python
## test equality
assert encoded==encoded_shuffled
```

#### Documentation
[`roux.lib.str`](https://github.com/rraadd88/roux#module-roux.lib.str)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_lib_sys.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions applicable to file-system.
<details><summary>Expand</summary>

### Encoding and decoding data


```python
# example data
parameters=dict(
    colindex='drug id',
    colsample='sample id',
    coly='auc',
    formulas={
            f"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm',
        },
    kws_get_stats_regression=dict(
            groups='screen_id',
        ),
    colsubset='sample subset',
    variable="C(sample_subset, Treatment(reference='ref'))[T.test]",
)
```


```python
## simulate modifications in dictionaries by subsetting for example
from random import sample
inputs=[]
for n in range(2,len(parameters),1):
    inputs.append({k:parameters[k] for k in list(parameters.keys())[:n]})
```


```python
# import helper functions
from roux.lib.sys import to_output_paths
output_paths=to_output_paths(
    input_paths=['tests/input/1/output.tsv','tests/input/2/output.tsv'],
    replaces_output_path={'input':'output'},
    inputs=inputs,
    output_path='tests/output/{KEY}/output.tsv',
    )
output_paths
```




    {'tests/output/2/output.tsv': 'tests/input/2/output.tsv',
     'tests/output/eae9949969282a98c356fe9f0e6d6aa9025eccc46d914226e3b36280d340e4fb/output.tsv': {'colindex': 'drug id',
      'colsample': 'sample id',
      'coly': 'auc'},
     'tests/output/f501f7c1674e142f9632f46cc921179f62f38a94c8cc59e0e5c15a2944689eb7/output.tsv': {'colindex': 'drug id',
      'colsample': 'sample id',
      'coly': 'auc',
      'formulas': {"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm'}},
     'tests/output/28d279121c3b1e82d60de99397e4d0f90f4f096127d06fed0c1d0422034a8967/output.tsv': {'colindex': 'drug id',
      'colsample': 'sample id',
      'coly': 'auc',
      'formulas': {"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm'},
      'kws_get_stats_regression': {'groups': 'screen_id'}},
     'tests/output/3b765e060a1cc9b06ec9e3010a797a7f6c77479488e1bdde90565bfa912fa3ea/output.tsv': {'colindex': 'drug id',
      'colsample': 'sample id',
      'coly': 'auc',
      'formulas': {"auc ~ C(sample_subset, Treatment(reference='ref')) + primary_or_metastasis": 'mixedlm'},
      'kws_get_stats_regression': {'groups': 'screen_id'},
      'colsubset': 'sample subset'},
     'tests/output/1/output.tsv': 'tests/input/1/output.tsv',
     'tests/output/032180568f7e29dd4a3042301f150f0988f968c2d093fa8a79a669ceee8359b6/output.tsv': {'colindex': 'drug id',
      'colsample': 'sample id'}}




```python
len(output_paths)
```




    7



#### Documentation
[`roux.lib.sys`](https://github.com/rraadd88/roux#module-roux.lib.sys)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_stat_cluster.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for clustering.
<details><summary>Expand</summary>

### Requirements


```python
# installing the required roux subpackage
!pip install roux[stat]
```


```python
from roux.lib.io import read_table,to_table
## reading a table generated using the roux_query.ipynb notebook
df01=read_table('tests/output/data/biomart/00_raw.tsv')
```

    WARNING:root:dropped columns: Unnamed: 0
    INFO:root:shape = (167181, 5)



```python
from roux.lib.io import *
```


```python
df1=df01.log.drop_duplicates(subset=['Gene stable ID','Gene % GC content'])
```

    INFO:root:drop_duplicates: shape changed: (167181, 5)->(22802, 5), width constant



```python
df1.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gene stable ID</th>
      <th>HGNC symbol</th>
      <th>Gene % GC content</th>
      <th>Transcript count</th>
      <th>Transcript length (including UTRs and CDS)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ENSG00000198888</td>
      <td>MT-ND1</td>
      <td>47.7</td>
      <td>1</td>
      <td>956</td>
    </tr>
  </tbody>
</table>
</div>




```python
to_table(df1,'tests/output/data/biomart/01_dedup.tsv')
```




    'data/biomart/01_dedup.tsv'



#### Documentation
[`roux.lib.io`](https://github.com/rraadd88/roux#module-roux.lib.io)

### Fitting a Gaussian-Mixture Model


```python
from roux.lib.io import read_table
df1=read_table('tests/output/data/biomart/01_dedup.tsv')
```

    WARNING:root:dropped columns: Unnamed: 0
    INFO:root:shape = (22802, 5)



```python
from roux.stat.cluster import cluster_1d
from roux.viz.io import to_plot
d1=cluster_1d(
    ds=df1['Gene % GC content'].copy(),
    n_clusters=2,
    clf_type='gmm',
    random_state=88,
    returns=['coff','mix_pdf','two_pdfs','weights'],
    ax=None,
    bins=60,
    test=True,
)
ax=plt.gca()
ax.set(xlabel='Gene % GC content',ylabel='density')
to_plot('plot/hist_gmm.png')
assert exists('tests/output/plot/hist_gmm.png')
```

    INFO:root:intersections [46.95]
    WARNING:root:overwritting: plot/hist_gmm.png



    
<img src="data:image/png;base64,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" />
    


#### Documentation
[`roux.stat.cluster`](https://github.com/rraadd88/roux#module-roux.stat.cluster)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_viz_annot.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for annotating visualisations.
<details><summary>Expand</summary>


```python
# installing the required roux subpackage
# !pip install roux[viz]
# loading requirements
import matplotlib.pyplot as plt
```

### Example of annotated scatter plot


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')

# plot
_,ax=plt.subplots(figsize=[3,3])
from roux.viz.scatter import plot_scatter
ax=plot_scatter(df1,colx='sepal_length',coly='petal_width',ax=ax)
from roux.viz.annot import annot_side
ax=annot_side(ax=ax,
           df1=df1.sample(5),
           colx='sepal_length',coly='petal_width',cols='species',length_axhline=1.3)
ax=annot_side(ax=ax,
           df1=df1.sort_values('petal_width',ascending=False).head(5),
           colx='sepal_length',coly='petal_width',cols='species',length_axhline=1,
           loc='top',)
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/scatter_annotated.png')
```

    WARNING:root:overwritting: tests/output/plot/scatter_annotated.png



    
<img src="data:image/png;base64,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" />
    


#### Documentation
[`roux.viz.annot`](https://github.com/rraadd88/roux#module-roux.viz.annot)
[`roux.viz.scatter`](https://github.com/rraadd88/roux#module-roux.viz.scatter)

### Example of annotated histogram


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')

# plot
_,ax=plt.subplots(figsize=[3,3])
from roux.viz.dist import hist_annot
ax=hist_annot(
    df1,colx='sepal_length',colssubsets=['species'],bins=10,
    params_scatter=dict(marker='|',alpha=1),
    ax=ax)
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/hist_annotated.png')
```

    WARNING:root:overwritting: tests/output/plot/hist_annotated.png



    
<img src="data:image/png;base64,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" />
    


#### Documentation
[`roux.viz.dist`](https://github.com/rraadd88/roux#module-roux.viz.dist)

### Example of annotated heatmap


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')
df1=(df1
    .set_index('species')
    .melt(ignore_index=False)
    .reset_index()
    .pivot_table(index='variable',columns='species',values='value',aggfunc='mean'))

# plot
_,ax=plt.subplots(figsize=[3,3])
ax=sns.heatmap(df1,
    cmap='Blues',
    cbar_kws=dict(label='mean value'),
    ax=ax,)
from roux.viz.annot import show_box
ax=show_box(ax=ax,xy=[1,2],width=2,height=1,ec='red',lw=2)
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/heatmap_annotated.png')
```

    WARNING:root:overwritting: tests/output/plot/heatmap_annotated.png



    
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYAAAAFcCAYAAADBO2nrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAABErUlEQVR4nO3deVhUZfvA8e8BWWUVUQQ3CERQUco0NRWzREtTKTW11zUrlVxxIQVBE9RyKTWXNFF/LpWpmZVLJGruGuCGu4j5YphrqCDC/P7wdd6ZF1QGmUXm/nSd62LOnHnOfZDOPc9ynkdRqVQqhBBCmB0LYwcghBDCOCQBCCGEmZIEIIQQZkoSgBBCmClJAEIIYaYkAQghhJmSBCCEEGZKEoAQQpgpSQBCCGGmyhk7AGFchy9mGzsEoxjy/WFjh2AUKQfOGTsEo7ix4t1iH2sXHK5T2XeT5+gajsmQBCCEEJoU82kYkQQghBCaFMXYERiMJAAhhNAkNQAhhDBTFpbGjsBgJAEIIYQmaQISQggzJU1AQghhpqQGIIQQZkpqAEIIYaakBiCEEGZKagBCCGGmpAYghBBmSmoAQghhpiQBCCGEmbKUJ4GFEMI8SR+AEEKYKTNqAjKfKxVCiOJQFN02HVy6dIl3330XNzc37OzsqFevHgcPHtTThTyZ1ACEEEKTnmoA169fp1mzZrRq1YpffvkFd3d3Tp8+jaurq17OVxySAIQQQpOe+gCmTp1KtWrVWLJkiXqft7e3Xs5VXNIEJIQQmhQLnbbc3Fxu3bqlteXm5hYqdsOGDTRs2JAuXbpQqVIlgoOD+eqrr4xwgf8lCUAIITTp2AcQHx+Ps7Oz1hYfH1+o2HPnzjFv3jz8/PzYvHkzAwcOZMiQISxdutQIF/mA2SaApKQkFEXhxo0bTzw2ISEBFxcXvcdUXDVr1mTWrFnGDkOIsknHGkBkZCQ3b97U2iIjIwsVW1BQwPPPP09cXBzBwcG8//77DBgwgPnz5xvhIh945hOAqd2cS1NZvjYhTJaFpU6bjY0NTk5OWpuNjU2hYqtUqUJgYKDWvoCAADIyMgx1ZYVIJ7AQQmjS0yigZs2acfLkSa19p06dokaNGno5X3EYvQYQEhJCeHg44eHhODs7U7FiRaKiolCpVADk5uYSERGBl5cX5cuXp3HjxiQlJQEPmnH69u3LzZs3URQFRVGIiYkBYPny5TRs2BBHR0c8PDzo0aMHWVlZpRb3Dz/8wPPPP4+trS0+Pj7ExsZy//599fuKorBo0SI6d+6Mvb09fn5+bNiwQauMDRs24Ofnh62tLa1atWLp0qXqZqnHXRvAnTt36NevH46OjlSvXp2FCxeW2rUJYdb09BzA8OHD2bt3L3FxcZw5c4aVK1eycOFCBg8erMeLeTyjJwCApUuXUq5cOfbv38/nn3/OjBkzWLRoEQDh4eHs2bOH1atXc/jwYbp06ULbtm05ffo0TZs2ZdasWTg5OZGZmUlmZiYREREA5OXlMWnSJFJTU1m/fj3p6en06dOnVOLduXMnvXr1YujQoRw/fpwFCxaQkJDA5MmTtY6LjY2la9euHD58mNdff52ePXty7do1AM6fP8/bb79Np06dSE1N5YMPPmDcuHHqzz7u2gCmT59Ow4YNSU5OZtCgQQwcOLDQtwshRAno2AdQXC+++CLr1q1j1apV1K1bl0mTJjFr1ix69uypx4t5PJNoAqpWrRozZ85EURT8/f05cuQIM2fOJDQ0lCVLlpCRkYGnpycAERERbNq0iSVLlhAXF4ezszOKouDh4aFVZr9+/dQ/+/j48MUXX/Diiy+SnZ2Ng4PDU8UbGxvL2LFj6d27t7r8SZMmMXr0aCZMmKA+rk+fPnTv3h2AuLg4vvjiC/bv30/btm1ZsGAB/v7+fPrppwD4+/tz9OhRdRKxtrZ+5LUBvP766wwaNAiAMWPGMHPmTLZt24a/v/9TXZsQZk+PcwG1b9+e9u3b6618XZlEAnjppZdQNH7pTZo0Yfr06Rw5coT8/Hxq1aqldXxubi5ubm6PLfPQoUPExMSQmprK9evXKSgoACAjI6NQR4yuUlNT2bVrl9Y3/vz8fHJycrhz5w729vYABAUFqd8vX748Tk5O6maokydP8uKLL2qV26hRo2LHoFn2wyTxpCau3NzcQuOT7+XmYV1Eh5UQZsuM5gIyiQTwKNnZ2VhaWnLo0CEs/2eK1sd9i799+zahoaGEhoayYsUK3N3dycjIIDQ0lHv37pVKXLGxsYSFhRV6z9bWVv2zlZWV1nuKoqgT0dMqSdnx8fHExsZq7ftwWCQDR3xcKjEJUSbIbKCGtW/fPq3Xe/fuxc/Pj+DgYPLz88nKyqJ58+ZFftba2pr8/HytfSdOnODq1atMmTKFatWqAZTqhEvPP/88J0+exNfXt8Rl+Pv78/PPP2vtO3DggNbroq7taURGRjJixAitfaey8kqtfCHKAsWMEoBJ1HUyMjIYMWIEJ0+eZNWqVcyePZuhQ4dSq1YtevbsSa9evVi7di3nz59n//79xMfH89NPPwEPHorKzs4mMTGRv//+mzt37lC9enWsra2ZPXs2586dY8OGDUyaNKnU4o2OjmbZsmXExsZy7Ngx0tLSWL16NePHjy92GR988AEnTpxgzJgxnDp1im+//ZaEhATgv3+ARV3b0yhqvLI0/wih7eGou+JuzzKTSAC9evXi7t27NGrUiMGDBzN06FDef/99AJYsWUKvXr0YOXIk/v7+dOrUiQMHDlC9enXgwWiZDz/8kG7duuHu7s60adNwd3cnISGB7777jsDAQKZMmcJnn31WavGGhoayceNGtmzZwosvvshLL73EzJkzdRrP6+3tzZo1a1i7di1BQUHMmzdPPQro4UMkRV2bEEK/FAtFp+1ZpqgeDrg3kpCQEBo0aCBTGwCTJ09m/vz5XLx40WDnPHwx22DnMiVDvj9s7BCMIuXAOWOHYBQ3Vrxb7GMdu+k2N88/3/TWNRyTYRJ9AObqyy+/5MUXX8TNzY1du3bx6aefEh4ebuywhDBrz3qzji5MognI2Nq1a4eDg0ORW1xcnN7Oe/r0aTp27EhgYCCTJk1i5MiRWk/7CiEMz5z6AIxeA3g4rYMxLVq0iLt37xb5XoUKFfR23pkzZzJz5ky9lS+EKIFn+56uE6MnAFPg5eVl7BCEECbiWf9WrwtJAEIIoUESgBBCmClJAEIIYaYkAQghhJl61h/u0oUkACGE0CA1ACGEMFOSAIQQwlyZz/1fEoAQQmiSGoAQQpgpSQBCCGGmJAEIIYSZkgQghBDmynzu/5IAhBBCk9QAhBDCTFlYmM8yKZIAhBBCk/lUAGRFMCGE0KSvFcFiYmIKfbZ27dp6vJInkxqAEEJo0GcfQJ06dfj111/Vr8uVM+4tWBKAEEJo0GcCKFeuHB4eHnorX1fSBCSEEBr0uSj86dOn8fT0xMfHh549e5KRkaGnqygeqQEIIYQmHSsAubm55Obmau2zsbHBxsZGa1/jxo1JSEjA39+fzMxMYmNjad68OUePHsXR0fFpoy4RRaVSqYxyZmEScu4bOwLjOJ9129ghGEXmrRxjh2AUr9R2K/axPiN+1qnsXk77iY2N1do3YcIEYmJiHvu5GzduUKNGDWbMmEH//v11OmdpkRqAEEJo0LVZJzIykhEjRmjt+99v/0VxcXGhVq1anDlzRqfzlSZJAEIIocFCxyUhi2ruKY7s7GzOnj3Lv/71L50/W1qkE1gIITQoim5bcUVERLB9+3bS09PZvXs3nTt3xtLSku7du+vvYp5AagBCCKFBX8NA//zzT7p3787Vq1dxd3fn5ZdfZu/evbi7u+vlfMUhCUAIITTo6zGA1atX66fgpyAJQAghNOjaB/AskwQghBAazGg2aEkAQgihSdYDEEIIM2VG939JAEIIoUlqAEIIYaakE1gIIcyU1ACEEMJMmdH9XxKAEEJokhqAEEKYKTO6/0sCEEIITeZUA5DZQIUQQoO+ZgMtTTk5pbOwjyQAIYTQoM81gZ9GQUEBkyZNwsvLCwcHB86dOwdAVFQUixcvLlGZkgCEEEKDqdYAPvnkExISEpg2bRrW1tbq/XXr1mXRokUlKlMSgBBCaLCwUHTaDGXZsmUsXLiQnj17Ymlpqd5fv359Tpw4UaIypRNYCCE0mGon8KVLl/D19S20v6CggLy8vBKVKTWA/5GUlISiKNy4caPUy1YUhfXr1z/y/fT0dBRFISUl5bHlhISEMGzYsFKNTQjxgKk2AQUGBrJz585C+9esWUNwcHCJyiyzNYCEhASGDRumlxt5SWVmZuLq6lrs45OSkmjVqhXXr1/HxcVFf4EJIdRMtQYQHR1N7969uXTpEgUFBaxdu5aTJ0+ybNkyNm7cWKIypQZgQB4eHtjY2Bg7DCHEY5jqKKCOHTvy448/8uuvv1K+fHmio6NJS0vjxx9/5LXXXitRmSabAEJCQggPDyc8PBxnZ2cqVqxIVFQUKpUKgNzcXCIiIvDy8qJ8+fI0btyYpKQk4ME35759+3Lz5k31P1JMTAwAy5cvp2HDhjg6OuLh4UGPHj3IysrSOT6VSoW7uztr1qxR72vQoAFVqlRRv/7999+xsbHhzp07QOEmoP379xMcHIytrS0NGzYkOTlZ/V56ejqtWrUCwNXVFUVR6NOnj/r9goICRo8eTYUKFfDw8FBfnxDi6ZhqExBA8+bN2bp1K1lZWdy5c4fff/+dNm3alLg8k00AAEuXLqVcuXLs37+fzz//nBkzZqiHO4WHh7Nnzx5Wr17N4cOH6dKlC23btuX06dM0bdqUWbNm4eTkRGZmJpmZmURERACQl5fHpEmTSE1NZf369aSnp2vdWItLURRatGihTjrXr18nLS2Nu3fvqnvkt2/fzosvvoi9vX2hz2dnZ9O+fXsCAwM5dOgQMTEx6hgBqlWrxvfffw/AyZMnyczM5PPPP9f63ZQvX559+/Yxbdo0Jk6cyNatW3W+DiGENlOtAeiDSfcBVKtWjZkzZ6IoCv7+/hw5coSZM2cSGhrKkiVLyMjIwNPTE4CIiAg2bdrEkiVLiIuLw9nZGUVR8PDw0CqzX79+6p99fHz44osvePHFF8nOzsbBwUGn+EJCQliwYAEAO3bsIDg4GA8PD5KSkqhduzZJSUm0bNmyyM+uXLmSgoICFi9ejK2tLXXq1OHPP/9k4MCBAFhaWlKhQgUAKlWqVKgPICgoiAkTJgDg5+fHnDlzSExMLHFVUAjxgKne0y0sLB6bcPLz83Uu06QTwEsvvaR1wU2aNGH69OkcOXKE/Px8atWqpXV8bm4ubm5ujy3z4bft1NRUrl+/TkFBAQAZGRkEBgbqFF/Lli0ZOnQoV65cYfv27YSEhKgTQP/+/dm9ezejR48u8rNpaWkEBQVha2urdX3FFRQUpPW6SpUqT2zKys3NJTc3V2ufytJG+iWE0GCq3+rXrVun9TovL4/k5GSWLl1KbGxsico06QTwKNnZ2VhaWnLo0CGtByKAx36Lv337NqGhoYSGhrJixQrc3d3JyMggNDSUe/fu6RxHvXr1qFChAtu3b2f79u1MnjwZDw8Ppk6dyoEDB8jLy6Np06Y6l1scVlZWWq8VRVEns0eJj48v9IcyLmoC46NjSjs8IZ5Zlia6IljHjh0L7Xv77bepU6cO33zzDf3799e5TJNOAPv27dN6vXfvXvz8/AgODiY/P5+srCyaN29e5Getra0LVYlOnDjB1atXmTJlCtWqVQPg4MGDJY5PURSaN2/ODz/8wLFjx3j55Zext7cnNzeXBQsW0LBhQ8qXL1/kZwMCAli+fDk5OTnqWsDevXsLXQOUrGpXlMjISEaMGKG1T2Up3/6F0GSiFYBHeumll3j//fdL9FmT7gTOyMhgxIgRnDx5klWrVjF79myGDh1KrVq16NmzJ7169WLt2rWcP3+e/fv3Ex8fz08//QRAzZo1yc7OJjExkb///ps7d+5QvXp1rK2tmT17NufOnWPDhg1MmjTpqWIMCQlh1apVNGjQAAcHBywsLGjRogUrVqx4ZPs/QI8ePVAUhQEDBnD8+HF+/vlnPvvsM61jatSogaIobNy4kStXrpCdnf1UsdrY2ODk5KS1SfOPENqepU7gu3fv8sUXX+Dl5VWiz5coAZw9e5bx48fTvXt3dbvzL7/8wrFjx0oUxKP06tWLu3fv0qhRIwYPHszQoUPVmW7JkiX06tWLkSNH4u/vT6dOnThw4ADVq1cHoGnTpnz44Yd069YNd3d3pk2bhru7OwkJCXz33XcEBgYyZcqUQjddXbVs2ZL8/HxCQkLU+0JCQgrt+18ODg78+OOPHDlyhODgYMaNG8fUqVO1jvHy8iI2NpaxY8dSuXJlwsPDnypWIcSTWSi6bSU1ZcoUFEUp9lP9rq6uVKhQQb25urri6OjI119/zaefflqiGBTVw4H1xbR9+3batWtHs2bN2LFjB2lpafj4+DBlyhQOHjyoNS7+aYSEhNCgQQNmzZpVKuWJouXcN3YExnE+67axQzCKzFulM4/8s+aV2o8fHKLp9fn7dSr75w8b6RoOBw4coGvXrjg5OdGqVati3ecSEhK0ahwWFha4u7vTuHFjnWYY0KRzH8DYsWP55JNPGDFiBI6Ojur9r7zyCnPmzClREEIIYSr03aqTnZ1Nz549+eqrr/jkk0+K/bmSPK/0JDongCNHjrBy5cpC+ytVqsTff/9dKkGZinbt2hU5+RLAxx9/zMcff2zgiIQQ+qagWwYoani1jc2jh1cPHjyYN954g1dfffWJCeDw4cPFjuN/h4YXh84JwMXFhczMTLy9vbX2Jycnl7gjoigPn7A1pkWLFnH37t0i33v4kJYQomzRtV2/qOHVEyZMKHJ6ltWrV/PHH39w4MCBYpXdoEEDFEXhSS31iqIY5kGwd955hzFjxvDdd9+px57v2rWLiIgIevXqpXMApqw0E5oQ4tmg68ieooZXF/Xt/+LFiwwdOpStW7dqPQD6OOfPn9cpFl3p3Al87949Bg8eTEJCAvn5+ZQrV478/Hx69OhBQkJCoQezhGmTTmDzIp3AT9ZpkW7PBq1/r2Hxjlu/ns6dO2vdI/Pz81EUBQsLC3Jzcw1+/9S5BmBtbc1XX31FVFQUR48eJTs7m+DgYPz8/PQRnxBCGJS+ngRu3bo1R44c0drXt29fateuzZgxY4p98z9+/DgZGRmFZi948803dY6pxE8CV69eXT3mXgghygp9Pdzl6OhI3bp1tfaVL18eNze3QvuLcu7cOTp37syRI0e0+gUexqu3PoD/bd96nBkzZugchBBCmApTnQpi6NCheHt7k5iYiLe3N/v37+fq1auMHDmyxA+0FisBaC5U8jjGfixaCCGeloUB72O6jHbcs2cPv/32GxUrVsTCwgILCwtefvll4uPjGTJkSLHv05qKlQC2bdumc8FCCPEsMtWvsfn5+eqHbytWrMi///1v/P39qVGjBidPnixRmU81G+jFixcB1DNrCiHEs85UWzLq1q1Lamoq3t7eNG7cmGnTpmFtbc3ChQvx8fEpUZk6TwZ3//59oqKicHZ2pmbNmtSsWRNnZ2fGjx9PXl5eiYIQQghTYajJ4HQ1fvx49ZofEydO5Pz58zRv3pyff/6ZL774okRl6lwD+Oijj1i7di3Tpk1Tr2C1Z88eYmJiuHr1KvPmzStRIEIIYQpMtQYQGhqq/tnX15cTJ05w7do1XF1dSxyzzglg5cqVrF69mnbt2qn3BQUFUa1aNbp37y4JQAjxTDPR+z//93//R+fOnbUWmXraKWl0bgKysbGhZs2ahfZ7e3urV7ASQohnlaWFotNmKMOHD6dy5cr06NGDn3/+uVRWCtQ5AYSHhzNp0iSt2e9yc3OZPHmyLFgihHjmmeqKYJmZmaxevRpFUejatStVqlRh8ODB7N69u8RlFqsJKCwsTOv1r7/+StWqValfvz4Aqamp3Lt3j9atW5c4ECGEMAUm2gJEuXLlaN++Pe3bt+fOnTusW7eOlStX0qpVK6pWrcrZs2d1L7M4Bzk7O2u9fuutt7ReyzBQIURZYcgHwUrK3t6e0NBQrl+/zoULF0hLSytROcVKAEuWLClR4UII8awx5fv/w2/+K1asIDExUT34pqRL8T7Vg2BCCFHWmOow0HfeeYeNGzdib29P165diYqKUg/FL6kSJYA1a9bw7bffFjkl6R9//PFUAQkhhDGZ6P0fS0tLvv32W0JDQ0tt3QCdRwF98cUX9O3bl8qVK5OcnEyjRo1wc3Pj3LlzWs8GCCHEs8hCUXTaDGXFihW8/vrrpbpojM41gC+//JKFCxfSvXt3EhISGD16ND4+PkRHR3Pt2rVSC0wIfbFu3BDff2caOwyj8NZtAcCyo5wFeHjAwSev9mWqNQB90DkBZGRk0LRpUwDs7Oz4559/APjXv/7FSy+9xJw5c0o3QqFXAaN+MnYIBrf7VDpVsq8aOwyjsDJ2AM8ASzPKADonAA8PD65du0aNGjWoXr06e/fupX79+pw/f/6JK9cLYUryUbhs6/zkA8uSfPObsNEj7w6WFP/eZKqdwPqgcwJ45ZVX2LBhA8HBwfTt25fhw4ezZs0aDh48WOiBMSFM2WVbZ3xbTzJ2GIZ16YSxIzC4M8cS8Mq7XezjDTnDp7HpnAAWLlyonpJ08ODBuLm5sXv3bt58800++OCDUg9QCCEMyZQTQEFBAWfOnCErK0t9H36oRYsWOpencwJ4uBTZQ++88w7vvPOOzicWQghTZKpNQHv37qVHjx5cuHChUHO7oij6WxT+8OHD1K1bFwsLCw4fPvzYY4OCgnQOQgghTIWp1gA+/PBDGjZsyE8//USVKlVKJVEVKwE0aNCAy5cvU6lSJRo0aICiKEV2+JY0CwkhhKkw0QoAp0+fZs2aNfj6+pZamcVKAOfPn8fd3V39sxBClFWmOhlc48aNOXPmjOETQI0aNQDIy8sjNjaWqKgovL29Sy0IIYQwFTpPj2AgH330ESNHjuTy5cvUq1cPKyvtpzpK0vyuUyewlZUV33//PVFRUTqfSAghngWGXOVLFw+n4e/Xr59638PmeL12Amvq1KkT69evZ/jw4TqfTAghTJ2JtgDppfld5wTg5+fHxIkT2bVrFy+88ILWAsUAQ4YMKbXghBDC0PRVAZg3bx7z5s0jPT0dgDp16hAdHV3sSTQfNsWXJp0TwOLFi3FxceHQoUMcOnRI6z1FUSQBCCGeafrqBK5atSpTpkzBz88PlUrF0qVL6dixI8nJydSpU6fY5Rw/frzIqfjffPNNnWPSOQHIKCAhRFmmryagDh06aL2ePHky8+bNY+/evcVKAOfOnaNz584cOXJEayj+w+cBStIHYKod3kIIYRQWim5bSeTn57N69Wpu375d7FW9hg4dire3N1lZWdjb23Ps2DF27NhBw4YNSUpKKlEcJVoR7M8//2TDhg1FVkNmzJhRokCEEMIUKOh2V8/NzSU3N1drn42NDTY2NoWOPXLkCE2aNCEnJwcHBwfWrVtHYGBgsc6zZ88efvvtNypWrKiekufll18mPj6eIUOGkJycrFPcUIIEkJiYyJtvvomPjw8nTpygbt26pKeno1KpeP7553UOQAghTImu3+rj4+OJjY3V2jdhwgRiYmIKHevv709KSgo3b95kzZo19O7dm+3btxcrCeTn5+Po6AhAxYoV+fe//42/vz81atTg5MmTugX9HzongMjISCIiIoiNjcXR0ZHvv/+eSpUq0bNnT9q2bVuiIIQQwlTomgAiIyMZMWKE1r6ivv0DWFtbq5/kfeGFFzhw4ACff/45CxYseOJ56tatS2pqKt7e3jRu3Jhp06ZhbW3NwoUL8fHx0S3o/9A5AaSlpbFq1aoHHy5Xjrt37+Lg4MDEiRPp2LEjAwcOLFEgQghhCnSdZO1RzT3FUVBQUKj56FHGjx/P7dsP1jWYOHEi7du3p3nz5ri5ufHNN9+U6Pw6J4Dy5cur2/2rVKnC2bNn1T3Yf//9d4mCEEIIU2Gpp6ExkZGRtGvXjurVq/PPP/+wcuVKkpKS2Lx5c7E+Hxoaqv7Z19eXEydOcO3aNVxdXUs8M6jOCeCll17i999/JyAggNdff52RI0dy5MgR1q5dy0svvVSiIIQQwlTo6zmArKwsevXqRWZmJs7OzgQFBbF582Zee+01nco5c+YMZ8+epUWLFlSoUOGpluLVOQHMmDGD7OxsAGJjY8nOzuabb77Bz89PRgAJIZ55+noSePHixU/1+atXr9K1a1e2bduGoiicPn0aHx8f+vfvj6urK9OnT9e5TJ0rO3FxcVy7dg140Bw0f/58Dh8+zPfff6+XR5X1qU+fPnTq1KlYx4aEhDBs2DC9xlNcSUlJKIrCjRs3jB2KEGWOoui2Gcrw4cOxsrIiIyMDe3t79f5u3bqxadOmEpWpcwK4cuUKbdu2pVq1aowaNYrU1NQSnVgUjyklHiHMgQWKTpuhbNmyhalTp1K1alWt/X5+fly4cKFEZeqcAH744QcyMzOJioriwIEDPP/889SpU4e4uDj1JEdCCPGsMtUawO3bt7W++T907dq1Eo9CKlF/t6urK++//z5JSUlcuHCBPn36sHz5cp1XqlmzZg316tXDzs4ONzc3Xn31VfUwp0WLFhEQEICtrS21a9fmyy+/VH8uPT0dRVFYvXo1TZs2xdbWlrp167J9+3b1Mfn5+fTv3x9vb2/s7Ozw9/fn888/L8nlFik3N5eIiAi8vLwoX748jRs31nocOyEhARcXFzZv3kxAQAAODg60bduWzMxM9TH3799nyJAhuLi44ObmxpgxY+jdu7e6WapPnz5s376dzz//HEVRUBRFK8keOnSIhg0bYm9vT9OmTUv8MIgQ4r8MMRVESTRv3pxly5apXyuKQkFBAdOmTaNVq1YlKvOpBjzl5eVx8OBB9u3bR3p6OpUrVy72ZzMzM+nevTv9+vUjLS2NpKQkwsLCUKlUrFixgujoaCZPnkxaWhpxcXFERUWxdOlSrTJGjRrFyJEjSU5OpkmTJnTo0IGrV68CD8bXVq1ale+++47jx48THR3Nxx9/zLfffvs0l6wWHh7Onj17WL16NYcPH6ZLly60bduW06dPq4+5c+cOn332GcuXL2fHjh1kZGQQERGhfn/q1KmsWLGCJUuWsGvXLm7dusX69evV73/++ec0adKEAQMGkJmZSWZmJtWqVVO/P27cOKZPn87BgwcpV66c1kIRQoiSsVAUnTZDmTZtGgsXLqRdu3bcu3eP0aNHU7duXXbs2MHUqVNLVGaJ5gLatm0bK1eu5Pvvv6egoICwsDA2btzIK6+8UuwyMjMzuX//PmFhYerO43r16gEPHqOePn06YWFhAHh7e3P8+HEWLFhA79691WWEh4erV8mZN28emzZtYvHixYwePRorKyutx7O9vb3Zs2cP3377LV27di3JZatlZGSwZMkSMjIy8PT0BCAiIoJNmzaxZMkS4uLigAcJcv78+Tz33HPqeCdOnKguZ/bs2URGRtK5c2cA5syZw88//6x+39nZGWtra+zt7fHw8CgUx+TJk2nZsiUAY8eO5Y033iAnJwdbW9unuj4hzJmpLghTt25dTp06xZw5c3B0dCQ7O5uwsDAGDx5MlSpVSlSmzgnAy8uLa9eu0bZtWxYuXEiHDh1K1P5Uv359WrduTb169QgNDaVNmza8/fbbWFtbc/bsWfr378+AAQPUx9+/fx9nZ2etMjRn0StXrhwNGzYkLS1NvW/u3Ll8/fXXZGRkcPfuXe7du0eDBg10jvV/HTlyhPz8fGrVqqW1Pzc3Fzc3N/Vre3t79c0fHjw4l5WVBcDNmzf566+/aNSokfp9S0tLXnjhBQoKCooVh+YaoA//ALKysqhevXqRxxc1aZXqfh5KOasijxfCHJnqkpDw4EvhuHHjSq08nRNATEwMXbp0wcXF5alObGlpydatW9m9ezdbtmxh9uzZjBs3jh9//BGAr776isaNGxf6THGtXr2aiIgIpk+fTpMmTXB0dOTTTz9l3759TxU3QHZ2NpaWlhw6dKhQTA4ODuqf/3fRZs05vEuDZvkPnwR8XPIoatIq58bdcW3Ss9RiEuJZZ8pz5Ofk5HD48GGysrIK/b9ukAVhNL+VPy1FUWjWrBnNmjUjOjqaGjVqsGvXLjw9PTl37hw9ez7+xrR3715atGgBPKghHDp0iPDwcAB27dpF06ZNGTRokPr4s2fPlkrcwcHB5Ofnk5WVRfPmzUtUhrOzM5UrV+bAgQPqa8jPz+ePP/7QqqVYW1uXaKGHohQ1aVXQuN9KpWwhyoqSTqugb5s2baJXr15FTrljsEXhS8u+fftITEykTZs2VKpUiX379nHlyhUCAgKIjY1lyJAhODs707ZtW3Jzczl48CDXr1/XuoHNnTsXPz8/AgICmDlzJtevX1d3hPr5+bFs2TI2b96Mt7c3y5cv58CBA3h7ez917LVq1aJnz5706tWL6dOnExwczJUrV0hMTCQoKIg33nijWOV89NFHxMfH4+vrS+3atZk9ezbXr1/X+gOsWbOmupPdwcGBChUqlDjuoiatkuYfIbSZ5u3/wf2iS5cuREdH6zTg5nGMlgCcnJzYsWMHs2bN4tatW9SoUYPp06erF0i2t7fn008/ZdSoUZQvX5569eoVeiBqypQpTJkyhZSUFHx9fdmwYQMVK1YE4IMPPiA5OZlu3bqhKArdu3dn0KBB/PLLL6US/5IlS/jkk08YOXIkly5domLFirz00ku0b9++2GWMGTOGy5cv06tXLywtLXn//fcJDQ3ValaKiIigd+/eBAYGcvfuXVmSUwg9M+TIHl389ddfjBgxotRu/gCKqjQbpQ0kPT0db29vkpOTS6VT11QUFBQQEBBA165dmTRpkkHO6T38J4Ocx5TsXtSbKtlXuWTrgm9rw/yeTcalE8aOwODOHEvAK+82eHnBn38+8fgVh558jKaeL1R98kGloF+/fjRr1oz+/fuXWplGqwEIuHDhAlu2bKFly5bk5uYyZ84czp8/T48ePYwdmhBmy0QrAMyZM4cuXbqwc+dO6tWrV2iQyZAhQ3QuUxIAD8b1P25JtuPHjz9yaOXTsLCwICEhgYiICFQqFXXr1uXXX38lICCg1M8lhCgeU+0EXrVqFVu2bMHW1lY9IeRDiqKYTwKoWbNmqQ6n9PT0JCUl5bHv60O1atXYtWuXXsoWQpSMqQ4DHTduHLGxsYwdOxYLi9KJ8plMAKWtXLlyOs9jJIQom0y1E/jevXt069at1G7+YLrJTgghjOLhxIvF3Qyld+/eJV7791GkBiCEEBpM9Vtxfn4+06ZNY/PmzQQFBRXqBC7JioySAIQQQoOpdgIfOXKE4OBgAI4ePar1nsEWhRdCiLLMNG//D2ZhLm2SAIQQQoOJVgD0QhKAEEJoMOQ6v8YmCUAIITRIDUAIIcyUIjUAIYQwT5ZmVAWQBCCEEBrM6P5vss88CCGEUSiKbltxxcfH8+KLL+Lo6EilSpXo1KkTJ0+e1N+FFIMkACGE0KDo+F9xbd++ncGDB7N37162bt1KXl4ebdq04fbt23q8mseTJiAhhNBgoacmoE2bNmm9TkhIoFKlShw6dEi9LrihSQIQQggNuo4Cys3NJTc3V2tfUetv/6+bN28CPNU6309LmoCEEEKDrn0A8fHxODs7a23x8fGPPUdBQQHDhg2jWbNm1K1b10BXVpjUAIQQQoOuNYDIyEhGjBihte9J3/4HDx7M0aNH+f3333WOrzRJAhBCCA269gEUp7lHU3h4OBs3bmTHjh1UrWqYBeUfRRKAEEJo0NeTwCqVio8++oh169aRlJSEt7e3Xs6jC0kAQgihQV+jgAYPHszKlSv54YcfcHR05PLlywA4OztjZ2enn5M+gSQAYbY8cm5yJjHK2GEYVn6esSMwOI+8Ozodr681gefNmwdASEiI1v4lS5bQp08fvZzzSSQBmLnXW/gYOwSDs1324M/eEhVeOTeMG4wwOfqaCUKlUump5JKTBCDMzi3XilhZmtGELxoKTPAmZAhONuXAw6N4B5vRn4YkAGF2pn36HXUqG6fN1dhu5eQbOwSjGPvKc8U+VqaDFkIIM2VOs4FKAhBCCA1mdP+XBCCEEFrMKANIAhBCCA3SByCEEGZKXw+CmSJJAEIIoUkSgBBCmCdpAhJCCDMlw0CFEMJMmdH9XxKAEEJoMaMMIAlACCE0SB+AEEKYKekDEEIIM2VG939JAEIIoUkxoyqAJAAhhNBgRvd/SQBCCKHJjO7/kgCEEEKLGWUASQBCCKFBhoEKIYSZMqc+AAtjB2CK+vTpQ6dOnUq93ISEBFxcXB57TExMDA0aNHjsMenp6SiKQkpKSqnFJoR4QNFxe5ZJAjCgbt26cerUKZ0+o69kJIR4BDPKANIEZEB2dnbY2dkZOwwhxGOYUx+ASdYA1qxZQ7169bCzs8PNzY1XX32V27dvA7Bo0SICAgKwtbWldu3afPnll+rPPWwaWb16NU2bNsXW1pa6deuyfft29TH5+fn0798fb29v7Ozs8Pf35/PPPy9RnBs3bsTFxYX8/HwAUlJSUBSFsWPHqo957733ePfdd4Gim4CmTJlC5cqVcXR0pH///uTk5Kjfi4mJYenSpfzwww8oioKiKCQlJanfP3fuHK1atcLe3p769euzZ8+eEl2HEOK/LBTdtmeZySWAzMxMunfvTr9+/UhLSyMpKYmwsDBUKhUrVqwgOjqayZMnk5aWRlxcHFFRUSxdulSrjFGjRjFy5EiSk5Np0qQJHTp04OrVqwAUFBRQtWpVvvvuO44fP050dDQff/wx3377rc6xNm/enH/++Yfk5GQAtm/fTsWKFbVu0tu3byckJKTIz3/77bfExMQQFxfHwYMHqVKlilZCi4iIoGvXrrRt25bMzEwyMzNp2rSp+v1x48YRERFBSkoKtWrVonv37ty/f1/n6xBCaNBjE9COHTvo0KEDnp6eKIrC+vXrSy/uEjDJBHD//n3CwsKoWbMm9erVY9CgQTg4ODBhwgSmT59OWFgY3t7ehIWFMXz4cBYsWKBVRnh4OG+99RYBAQHMmzcPZ2dnFi9eDICVlRWxsbE0bNgQb29vevbsSd++fUuUAJydnWnQoIH6hp+UlMTw4cNJTk4mOzubS5cucebMGVq2bFnk52fNmkX//v3p378//v7+fPLJJwQGBqrfd3BwwM7ODhsbGzw8PPDw8MDa2lr9fkREBG+88Qa1atUiNjaWCxcucObMGZ2vQwjxX4qO/+ni9u3b1K9fn7lz5+opet2YXAKoX78+rVu3pl69enTp0oWvvvqK69evc/v2bc6ePUv//v1xcHBQb5988glnz57VKqNJkybqn8uVK0fDhg1JS0tT75s7dy4vvPAC7u7uODg4sHDhQjIyMkoUb8uWLUlKSkKlUrFz507CwsIICAjg999/Z/v27Xh6euLn51fkZ9PS0mjcuPEjY3+SoKAg9c9VqlQBICsr65HH5+bmcuvWLa0tP+9esc8nhDlQFN02XbRr145PPvmEzp076yd4HZlcArC0tGTr1q388ssvBAYGMnv2bPz9/Tl69CgAX331FSkpKert6NGj7N27t9jlr169moiICPr378+WLVtISUmhb9++3LtXshthSEgIv//+O6mpqVhZWVG7dm1CQkJISkpi+/btj/z2XxqsrKzUPz+cwKqgoOCRx8fHx+Ps7Ky1Hfp+od7iE+JZZEaDgEwvAcCDm1mzZs2IjY0lOTkZa2trdu3ahaenJ+fOncPX11dr8/b21vq8ZkK4f/8+hw4dIiAgAIBdu3bRtGlTBg0aRHBwML6+voVqELp42A8wc+ZM9c3+YQJISkp6ZPs/QEBAAPv27Xtk7ADW1tbqTuanFRkZyc2bN7W2F956v1TKFqLM0DEDFFWzzs3NNVLwujG5YaD79u0jMTGRNm3aUKlSJfbt28eVK1cICAggNjaWIUOG4OzsTNu2bcnNzeXgwYNcv36dESNGqMuYO3cufn5+BAQEMHPmTK5fv06/fv0A8PPzY9myZWzevBlvb2+WL1/OgQMHCiWR4nJ1dSUoKIgVK1YwZ84cAFq0aEHXrl3Jy8t7bA1g6NCh9OnTh4YNG9KsWTNWrFjBsWPH8PHxUR9Ts2ZNNm/ezMmTJ3Fzc8PZ2blEcQLY2NhgY2Ojtc/SyvoRRwthnnRt14+Pjyc2NlZr34QJE4iJiSnFqPTD5BKAk5MTO3bsYNasWdy6dYsaNWowffp02rVrB4C9vT2ffvopo0aNonz58tSrV49hw4ZplTFlyhSmTJlCSkoKvr6+bNiwgYoVKwLwwQcfkJycTLdu3VAUhe7duzNo0CB++eWXEsfcsmVLUlJS1N/2K1SoQGBgIH/99Rf+/v6P/Fy3bt04e/Yso0ePJicnh7feeouBAweyefNm9TEDBgwgKSmJhg0bkp2dzbZt26hZs2aJYxVCPJ6u7fqRkZFaX0CBQl+0TJWiUqlUxg6itKSnp+Pt7U1ycvITp1MQDwxel/bkg8qgOpXN84G8Wzml05z4rBn7ynPFPvbiNd2ab6pVKNnNXlEU1q1bZ9Qn/U2uBiCEEMakz8ngsrOztYZqnz9/npSUFCpUqED16tX1d+JHkATwGBkZGVrj8v/X8ePHjfKPJoTQH30uCXnw4EFatWqlfv2w6ah3794kJCTo7byPUqYSQM2aNSnNFi1PT8/Hzrjp6elZaucSQpgGfQ7tDAkJKdV71NMqUwmgtJUrVw5fX19jhyGEMCBzWg9AEoAQQmgwp9lAJQEIIYQm87n/SwIQQghNZnT/lwQghBCapA9ACCHMlPQBCCGEuTKf+78kACGE0PSsL/OoC0kAQgihQZqAhBDCTJlTJ7BJLggjhBBC/6QGIIQQGsypBiAJQAghNEgfgBBCmCmpAQghhJkyo/u/JAAhhNBiRhlAEoAQQmiwMKM2IEkAQgihwXxu/5IAhBBCmxllAEkAQgihQYaBCiGEmTKjLgAUlSktUS/MRm5uLvHx8URGRmJjY2PscAxGrtu8rtvUSQIQRnHr1i2cnZ25efMmTk5Oxg7HYOS6zeu6TZ1MBieEEGZKEoAQQpgpSQBCCGGmJAEIo7CxsWHChAlm1yEo121e123qpBNYCCHMlNQAhBDCTEkCEEIIMyUJQAghzJQkACGEMFOSAITQk7y8PFq3bs3p06eNHYoQRZLJ4ITB5eTkcO/ePa19ZXF6ACsrKw4fPmzsMIR4JKkBCIO4c+cO4eHhVKpUifLly+Pq6qq1lVXvvvsuixcvNnYYBpefn89nn31Go0aN8PDwoEKFClqbMA1SAxAGMWrUKLZt28a8efP417/+xdy5c7l06RILFixgypQpxg5Pb+7fv8/XX3/Nr7/+ygsvvED58uW13p8xY4aRItOv2NhYFi1axMiRIxk/fjzjxo0jPT2d9evXEx0dbezwxH/Ig2DCIKpXr86yZcsICQnBycmJP/74A19fX5YvX86qVav4+eefjR2iXrRq1eqR7ymKwm+//WbAaAznueee44svvuCNN97A0dGRlJQU9b69e/eycuVKY4cokBqAMJBr167h4+MDPGjvv3btGgAvv/wyAwcONGZoerVt2zZjh2AUly9fpl69egA4ODhw8+ZNANq3b09UVJQxQxMapA9AGISPjw/nz58HoHbt2nz77bcA/Pjjj7i4uBgxMsP5888/+fPPP40dhkFUrVqVzMxM4EFtYMuWLQAcOHBA5gMyIZIAhEH07duX1NRUAMaOHcvcuXOxtbVl+PDhjBo1ysjR6U9BQQETJ07E2dmZGjVqUKNGDVxcXJg0aRIFBQXGDk9vOnfuTGJiIgAfffQRUVFR+Pn50atXL/r162fk6MRD0gcgjOLChQscOnQIX19fgoKCjB2O3kRGRrJ48WJiY2Np1qwZAL///jsxMTEMGDCAyZMnGzlCw9i7dy+7d+/Gz8+PDh06GDsc8R+SAITR3Lhxo8w3/3h6ejJ//nzefPNNrf0//PADgwYN4tKlS0aKTAhpAhIGMnXqVL755hv1665du+Lm5oaXl5e6aagsunbtGrVr1y60v3bt2uqO8LIoPj6er7/+utD+r7/+mqlTpxohIlEUSQDCIObPn0+1atUA2Lp1K1u3buWXX36hXbt2ZboPoH79+syZM6fQ/jlz5lC/fn0jRGQYCxYsKDLx1alTh/nz5xshIlEUGQYqDOLy5cvqBLBx40a6du1KmzZtqFmzJo0bNzZydPozbdo03njjDX799VeaNGkCwJ49e7h48WKZffYBHvx7V6lSpdB+d3d39eggYXxSAxAG4erqysWLFwHYtGkTr776KgAqlYr8/HxjhqZXLVu25NSpU3Tu3JkbN25w48YNwsLCOHnyJM2bNzd2eHpTrVo1du3aVWj/rl278PT0NEJEoihSAxAGERYWRo8ePfDz8+Pq1au0a9cOgOTkZHx9fY0cnX55enqazWifhwYMGMCwYcPIy8vjlVdeASAxMZHRo0czcuRII0cnHpIEIAxi5syZ1KxZk4sXLzJt2jQcHBwAyMzMZNCgQUaOrnTpMgNoWR0CO2rUKK5evcqgQYPUM7/a2toyZswYIiMjjRydeEiGgQpRyiwsLFAUhSf9r6UoSplu/gLIzs4mLS0NOzs7/Pz85ClgEyMJQBjM2bNnmTVrFmlpaQAEBgYybNgw9RxBZcWFCxeKfWyNGjX0GIkQjycJQBjE5s2befPNN2nQoIH6idhdu3aRmprKjz/+yGuvvWbkCMXTCgsLIyEhAScnJ8LCwh577Nq1aw0UlXgc6QMQBjF27FiGDx9eaO7/sWPHMmbMmDKdAIqq+QwdOpTnnnvOyJGVLmdnZxRFUf8sTJ/UAIRB2NracuTIEfz8/LT2nzp1iqCgIHJycowUmX5JzUeYMqkBCINwd3cnJSWlUAJISUmhUqVKRopK/8y55iNMnyQAYRADBgzg/fff59y5czRt2hR48E146tSpjBgxwsjR6U9aWpp67QNN/fr1Y9asWYYPyED++usvIiIiSExMJCsrq9CIqLI++ulZIQlAGERUVBSOjo5Mnz5dPQ7c09OTmJgYhgwZYuTo9Mdcaz59+vQhIyODqKgoqlSpou4bEKZF+gCEwf3zzz8AODo6GjkS/Zs4cSIzZ85k7NixRdZ8yuryiI6OjuzcuZMGDRoYOxTxGFIDEAbxyiuvsHbtWlxcXLRu/Ldu3aJTp05ldnF0c635VKtW7YkPwgnjkxqAMAgLCwsuX75cqNkjKysLLy8v8vLyjBSZ4ZhTzWfLli1Mnz6dBQsWULNmTWOHIx5BagBCrzTnxTl+/DiXL19Wv87Pz2fTpk14eXkZIzSDOH/+PPfv38fPz0/rxn/69GmsrKzK7M2xW7du3Llzh+eeew57e3usrKy03i/Li+E8SyQBCL1q0KABiqKgKIp6VkhNdnZ2zJ492wiRGUafPn3o169foU7gffv2sWjRIpKSkowTmJ6V5RFOZYk0AQm9unDhAiqVCh8fH/bv34+7u7v6PWtraypVqoSlpaURI9QvJycn/vjjj0JTXp85c4aGDRty48YN4wQmBFIDEHr2cLKzgoICI0diHIqiqNv+Nd28ebPMjYW/desWTk5O6p8f5+FxwrhkRTBhMMuXL6dZs2Z4enqqZ8ycOXMmP/zwg5Ej058WLVoQHx+vdbPPz88nPj6el19+2YiRlT5XV1eysrIAcHFxwdXVtdD2cL8wDVIDEAYxb948oqOjGTZsGJMnT1bfEF1dXZk1axYdO3Y0coT6MXXqVFq0aIG/v796CcidO3dy69atMjf09bfffqNChQoAbNu2zcjRiOKQPgBhEIGBgcTFxdGpUyccHR1JTU3Fx8eHo0ePEhISwt9//23sEPXm3//+N3PmzCE1NRU7OzuCgoIIDw9X3yyFMBapAQiDOH/+PMHBwYX229jYcPv2bSNEZDienp7ExcUZOwyDetSymIqiYGtrS/Xq1WV1MBMgCUAYhLe3NykpKYVWwNq0aRMBAQFGiko/Dh8+TN26dbGwsHji+sBldU3gh8N/H8XKyopu3bqxYMECbG1tDRiZ0CQJQBjEiBEjGDx4MDk5OahUKvbv38+qVauIj49n0aJFxg6vVDVo0ED91PPDG2FRLa1leU3gdevWMWbMGEaNGkWjRo0A2L9/P9OnT2fChAncv3+fsWPHMn78eD777DMjR2u+pA9AGMyKFSuIiYnh7NmzAHh5eRETE0P//v2NHFnpunDhAtWrV0dRlCeuD1xW1wRu1KgRkyZNIjQ0VGv/5s2biYqKYv/+/axfv56RI0eq/x6E4UkCEAZx9+5dVCoV9vb23Llzh6NHj7Jr1y4CAwML3STEs8/Ozo7k5GRq166ttf/EiRMEBwdz9+5d0tPTCQwM5M6dO0aKUshzAMIgOnbsyLJlywC4d+8eb775JjNmzKBTp07MmzfPyNHpz9KlS/npp5/Ur0ePHo2LiwtNmzZ9Yu3gWVa7dm2mTJnCvXv31Pvy8vKYMmWKOilcunSJypUrGytEgSQAYSB//PGHehz8mjVrqFy5MhcuXGDZsmV88cUXRo5Of+Li4rCzswNgz549zJkzh2nTplGxYkWGDx9u5Oj0Z+7cuWzcuJGqVavy6quv8uqrr1K1alU2btyoTvjnzp1j0KBBRo7UvEkTkDAIe3t7Tpw4QfXq1enatSt16tRhwoQJXLx4EX9//zLbDKB53WPGjCEzM5Nly5Zx7NgxQkJCuHLlirFD1Jt//vmHFStWcOrUKQD8/f3p0aOHWUyH/ayQUUDCIHx9fVm/fj2dO3dm8+bN6m+/WVlZZXpeGAcHB65evUr16tXZsmWLev1jW1tb7t69a+To9CMvL4/atWuzceNGPvzwQ2OHIx5DmoCEQURHRxMREUHNmjVp3LgxTZo0AR4sHFLUA2JlxWuvvcZ7773He++9x6lTp3j99dcBOHbsWJldC8DKyoqcnBxjhyGKQZqAhMFcvnyZzMxM6tevj4XFg+8e+/fvx8nJqdBokbLixo0bREVFkZGRwcCBA2nbti0AEyZMwNramnHjxhk5Qv2Ii4vj1KlTLFq0iHLlpKHBVEkCEEJP7t+/T1xcHP369aNq1arGDsegOnfuTGJiIg4ODtSrV4/y5ctrvb927VojRSY0SQIQQo8cHBw4evRomW3ueZS+ffs+9v0lS5YYKBLxOJIAhNCjjh07EhYWRu/evY0dihCFSOOcEHrUrl07xo4dy5EjR3jhhRcKNYW8+eabRopMCKkBCKFXDzu7i1LWJoN7/vnnSUxMxNXVleDg4MfOBvrHH38YMDLxKFIDEEKPzGkt5I4dO6rn+O/UqZNxgxHFIjUAIQwkJyfHbOa+f++99+jZsyetWrUydijiMeRBMCH0KD8/n0mTJuHl5YWDgwPnzp0DICoqisWLFxs5Ov25cuUK7dq1o1q1aowePZrU1FRjhySKIAlACD2aPHkyCQkJTJs2DWtra/X+unXrlrmFcDT98MMPZGZmquf+f/7556lTpw5xcXGkp6cbOzzxH9IEJIQe+fr6smDBAlq3bo2joyOpqan4+Phw4sQJmjRpwvXr140dokH8+eefrFq1iq+//prTp09z//59Y4ckkBqAEHp16dIlfH19C+0vKCggLy/PCBEZXl5eHgcPHmTfvn2kp6fLGgAmRBKAEHoUGBjIzp07C+1fs2ZNmZ4ED2Dbtm0MGDCAypUr06dPH5ycnNi4cSN//vmnsUMT/yHDQIXQo+joaHr37s2lS5coKChg7dq1nDx5kmXLlrFx40Zjh6c3Xl5eXLt2jbZt27Jw4UI6dOigHiIqTIf0AQihZzt37mTixImkpqaSnZ3N888/T3R0NG3atDF2aHrz1Vdf0aVLF1xcXIwdingMSQBC6NF7773Hu+++S0hIiLFDEaIQ6QMQQo+uXLlC27ZtZTy8MElSAxBCz65fv853333HypUr2blzJ7Vr16Znz5706NHD7KaJFqZFEoAQBiTj4YUpkSYgIQxExsMLUyMJQAg9k/HwwlRJE5AQeqQ5Hr5nz54yHl6YFEkAQuiRjIcXpkwSgBBCmCnpAxBCCDMlCUAIIcyUJAAhhDBTkgCEEMJMSQIQwkT06dOHTp06GTsMYUZkFJAQJuLmzZuoVCoZMioMRhKAEEKYKWkCEkLDmjVrqFevHnZ2dri5ufHqq69y+/ZtdfNMbGws7u7uODk58eGHH3Lv3j31ZwsKCoiPj8fb2xs7Ozvq16/PmjVrtMo/duwY7du3x8nJCUdHR5o3b87Zs2eBwk1ATyrv+vXr9OzZE3d3d+zs7PDz82PJkiX6/QWJMkWWhBTiPzIzM+nevTvTpk2jc+fO/PPPP+zcuZOHleTExERsbW1JSkoiPT2dvn374ubmxuTJkwGIj4/n//7v/5g/fz5+fn7s2LGDd999F3d3d1q2bMmlS5do0aIFISEh/Pbbbzg5ObFr165Hzgj6pPKioqI4fvw4v/zyCxUrVuTMmTPcvXvXYL8vUQaohBAqlUqlOnTokApQpaenF3qvd+/eqgoVKqhu376t3jdv3jyVg4ODKj8/X5WTk6Oyt7dX7d69W+tz/fv3V3Xv3l2lUqlUkZGRKm9vb9W9e/eKPH/v3r1VHTt2VKlUqmKV16FDB1Xfvn1LfL1CSA1AiP+oX78+rVu3pl69eoSGhtKmTRvefvttXF1d1e/b29urj2/SpAnZ2dlcvHiR7Oxs7ty5w2uvvaZV5r179wgODgYgJSWF5s2bY2Vl9cRYzpw588TyBg4cyFtvvcUff/xBmzZt6NSpE02bNn2q34EwL5IAhPgPS0tLtm7dyu7du9myZQuzZ89m3Lhx7Nu374mfzc7OBuCnn37Cy8tL672Hs3/a2dkVO5bilNeuXTsuXLjAzz//zNatW2ndujWDBw/ms88+K/Z5hHmTBCCEBkVRaNasGc2aNSM6OpoaNWqwbt06AFJTU7l79676Rr53714cHByoVq0aFSpUwMbGhoyMDFq2bFlk2UFBQSxdupS8vLwn1gICAwOfWB6Au7s7vXv3pnfv3jRv3pxRo0ZJAhDFJglAiP/Yt28fiYmJtGnThkqVKrFv3z6uXLlCQEAAhw8f5t69e/Tv35/x48eTnp7OhAkTCA8Px8LCAkdHRyIiIhg+fDgFBQW8/PLL3Lx5k127duHk5ETv3r0JDw9n9uzZvPPOO0RGRuLs7MzevXtp1KgR/v7+WrEUp7zo6GheeOEF6tSpQ25uLhs3biQgIMBIvz3xTDJ2J4QQpuL48eOq0NBQlbu7u8rGxkZVq1Yt1ezZs1Uq1X87aKOjo1Vubm4qBwcH1YABA1Q5OTnqzxcUFKhmzZql8vf3V1lZWanc3d1VoaGhqu3bt6uPSU1NVbVp00Zlb2+vcnR0VDVv3lx19uxZrXMUt7xJkyapAgICVHZ2dqoKFSqoOnbsqDp37pwBflOirJAHwYQohj59+nDjxg3Wr19v7FCEKDXyIJgQQpgpSQBCCGGmpAlICCHMlNQAhBDCTEkCEEIIMyUJQAghzJQkACGEMFOSAIQQwkxJAhBCCDMlCUAIIcyUJAAhhDBTkgCEEMJM/T+wQJbpfOAYHQAAAABJRU5ErkJggg==" />
    


#### Documentation
[`roux.viz.heatmap`](https://github.com/rraadd88/roux#module-roux.viz.heatmap)

### Example of annotated distributions


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')
df1=df1.loc[df1['species'].isin(['setosa','virginica']),:]
df1['id']=range(len(df1))

# plot
from roux.viz.dist import plot_dists
ax=plot_dists(df1,x='sepal_length',y='species',colindex='id',kind=['box','strip'])
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/dists_annotated.png')
```

    WARNING:root:overwritting: tests/output/plot/dists_annotated.png



    
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAQgAAADZCAYAAADc467BAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/NK7nSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA750lEQVR4nO29eXwURf7//+y5Z5JMQhISCIEkEI4g4ZZTXRWXQ2DdxRv4ia7gxxWP1UXQn4KLiqwiXgiKssIqILq6y+p6gSgrcoscAnKfJpBAEjKTzN1d3z8mGRiSwRASkpB6Ph55MF1dVf3upvvV73pXdZUihBBIJBJJJejq2gCJRFJ/kQIhkUgiIgVCIpFERAqERCKJiBQIiUQSESkQEokkIlIgJBJJRKRASCSSiBjq2gBJEE3TyM3NJSYmBkVR6tocySWMEAKn00lKSgo63bl9BCkQ9YTc3FxatmxZ12ZIGhFHjx4lNTX1nHmkQNQTYmJigOB/mt1ur2NrJJcyDoeDli1bhu65cyEFop5Q3qyw2+1SICQXhao0ZWWQUiKRREQKhEQiiYhsYjQihBAEAoFz7vd4PABYLJZzuqAGg0H2tjQCpEA0IgKBAAsXLoy43+/3s3jxYgBGjhyJ0WiMmHf06NHn3C+5NJAC0QjZtP94pemaetq72HIwD52+8tujR5tmtWKXpP4hBaKRktXnWhS9Piwt4POyc+0KADr0uRaDyRy2X6gqP6/75qLZKKl7pEA0UhS9Hv1ZHoLQq6Hfer2hwn4VSWND9mJIJJKISIGQSCQRkQIhkUgiIgVCIpFERAqERCKJiBQIiUQSESkQEokkIlIgJBJJRKRASCSSiEiBkEgkEZECIZFIIiK/xajnnM8cDQ2ZxnKeDQ3pQdRzPB4PQ4YMYciQIaEH6FKksZxnQ0MKhEQiiYgUCIlEEhEpEBKJJCJSICQSSUQalUAcOnQIRVHYsmVLXZsikTQIZDdnA2ZffgkbDxVSUOIlIdrM5enxZCZFV7u+AlcAZ1JnVIONrcc9xNk0nF6NUr9GlFFHil1/zvI1bY+k7mmQHsRHH31EdnY2VquVhIQErrvuOkpLSwGYN28eWVlZWCwWOnTowJw5c0LlMjIyAOjWrRuKonD11VcDwZW1n376aVJTUzGbzXTt2pUvv/wyVM7n83H//ffTvHlzLBYLaWlpTJ8+PbT/pZdeIjs7m6ioKFq2bMl9991HSUlJrV6DffklfLo1l+PFHvyq4Hixh/9uy2VffklYnvc3HOH1b/by/oYj7D9RGrG+glI/u054CZiiETod+SV+Vh10suO4i8OFHvaedPNjjosSYa60fFXskTQ8GpwHcezYMW6//XZeeOEF/vCHP+B0Olm1ahVCCBYtWsSUKVN4/fXX6datG5s3b2bcuHFERUUxZswYNmzYQK9evfj666+57LLLMJlMALz66qvMnDmTuXPn0q1bN9555x1+97vfsWPHDtq2bctrr73GJ598wocffkirVq04evQoR48eDdmk0+l47bXXyMjI4MCBA9x3331MnDgxTJxqmo2HCiukCRFMz0yKDj2w5Rwv9vB5kQvVW7kXcPSUL2z7pEul1Keh1ylEm/R4AwJvIECJFlste87mm135fLo1lxNOD01jLAzKSjjn+UrqhgYpEIFAgBEjRpCWlgZAdnY2AE899RQzZ85kxIgRQNBj2LlzJ3PnzmXMmDE0bdoUgISEBJo1O722w4svvsikSZO47bbbAHj++ef59ttveeWVV5g9ezZHjhyhbdu2XHHFFSiKEjpuOX/+859Dv9PT03n22We59957a1wgDpwsZduxExSUeNn2SzHJdgvxUabQ/sJSHz8fc1BQ4mX/iVKizYaw/QLBYXfli92U+jVcfg1fVDKazojfraKgoCDOKA8OLICvQvmCEi+FpT5yT7lx+QLYTAaizIaQPWc2Ob7Zlc/c/+0Plc095ead1Ucojc0gqvjghV8oSY3R4ASiS5cuDBgwgOzsbAYNGsTAgQO56aabMJlM7N+/n7vvvptx48aF8gcCAWJjK3/rQXAp9NzcXPr37x+W3r9/f7Zu3QrAnXfeyW9/+1vat2/P4MGDGTZsGAMHDgzl/frrr5k+fTq7du3C4XAQCATweDy4XC5sNluNnLfPEs/n2/MwGIIPuKoJ9uQ5aZccQ3yUicJSH1uPnsKnajg9fvIcHqItRrqkxoVEorDUx95SE2hJ+I+5iLUYQzGGY8VeHJ4Amj6YVxOgCYFed/aQ58qHQKsa7MlzhrZPOL3sOu6gZRNbWJNjWOeUMM+mHIHAlZglBaKe0eBiEHq9nuXLl/PFF1/QsWNHZs2aRfv27dm+fTsAb7/9Nlu2bAn9bd++nXXr1l3QMbt3787Bgwd55plncLvd3HLLLdx0001AsGdk2LBhdO7cmY8//phNmzYxe/ZsIBi7qCm89tSw7ZQ4KxB8+wLszXNSUOrDpNehiaB7f7zYzfd7T7DhYCHrDxSw7ZdiNKGgCYX8kgDrjpSQ5/ShaQJvQFDi0wiY7ARMdoQQaJrApwocHpUSn0pAFcTirtxAReD2q+Q7POQUuck55cbtU8lzethwsJDtOcUUlPjYeKiQE86KQ6ndfg1vVDKnUvrw4aYcGbuoJzQ4gQBQFIX+/fszdepUNm/ejMlkYvXq1aSkpHDgwAEyMzPD/sqDk+UxB1U9vQSM3W4nJSWF1atXhx1j9erVdOzYMSzfrbfeyttvv80HH3zAxx9/TGFhIZs2bULTNGbOnEmfPn1o164dubkV35AXimoI90Tio0y0S45Br1MwGXQ4PH4SokxYTcEYg1Gvw+1TOeXyowlBnsNDocuPWdEAcHhUhACHN3gtAlqwKSF0QQ/FoFNQlKAXQXkzQ4FYpXKBOOXyh7KhgD+g4QlouHwqmhCUeAPsyXOyN89J0xhLWFm3T6Ww1IcitGCA1OmVAc56QoNrYqxfv54VK1YwcOBAkpKSWL9+PSdOnCArK4upU6fy4IMPEhsby+DBg/F6vfzwww8UFRXxyCOPkJSUhNVq5csvvyQ1NRWLxUJsbCyPPvooTz31FG3atKFr167Mnz+fLVu2sGjRIiDYS9G8eXO6deuGTqfjn//8J82aNSMuLo7MzEz8fj+zZs1i+PDhrF69mjfffLPGz1sfcFHk8pHndOHyqdhMelLirFzTIYnbe7Vi69Eiit2n19b0qxpWkyHUTFAUhXibCW+xrmy/AJSyf8GvCXQKKJofg8+BSR+PJgSaCJY16BSaWPR4SiuPYZxy+bGa9CGBcvtURCDodZxJkcvP8C4pvPXdfkTZLqfHDygYXSdC+c4V4JRcPBqcQNjtdr777jteeeUVHA4HaWlpzJw5kyFDhgBgs9mYMWMGjz76KFFRUWRnZ4eCiAaDgddee42nn36aKVOmcOWVV7Jy5UoefPBBiouL+ctf/kJ+fj4dO3bkk08+oW3btgDExMTwwgsvsHfvXvR6PZdffjmff/45Op2OLl268NJLL/H888/z+OOPc9VVVzF9+nTuuOOOGjlfUfYU6bxOdh8vQVcWE3C4AzjdPvq3jsXtdtMuycb6g6cof437VBWDDlrEWemWGsMOHZxwejjp1YGwofNrQNBLOHrKiy8QbE5oBis+g5mAV0MBLEaFGHPwoS9yq/i0GFbmg1hzgN6tE2idGAVAtElB07TQ8fW64C+9DjSt3GNTiDEp9E2Lwdcnlc9+yqOg1IfZoKN9kpW1O06VnXMwd2FpzTXRJNVDEeV3oOSi4vV68Xq9oW2Hw0HLli0pLi7GbreH0ouKioLduUmdcce0xG9LRNOb0Kk+jK6TWJ1Hicnfhs8Sz6mUPvijmqLpjaDo0QU8mJ2/oA+48Zvj8MakoggVvb+E6OR0HF4NRQEFhYAmUAWgBkARKGVNjSijQhObEb+qUepT8RTlYSvaC4AiwFawC5OnEGdSZ7y2pqhmO5rOiGawEvRRQBdwo9P86L0OzK4TxORvC7sWzqTOBEynPYXbbr0Vi8VKs1gLt/dqVWv/B40Vh8NBbGxshXutMhqcB3GpMH36dKZOnVrl/D5rAqo5Gp3qQacGg3yqKRqf5YzxA6FJVhQUzR9WXhjMKKoXdHoCphhKfRqaCD7AeqVMHIJFUTQVRW9EE+AJCByeAH5NBAOXehOemFR0mh8l4MXfog9GdyECBUWAqTQvaJvBSsCaiMF9En0gGLdQBJgdv1Q4N7PjF9SEDogzOkgUBS5Pj6/y9ZHUDlIg6ojHH3+cRx55JLRd7kGcTY4zOPzZY29J89RW2K1GrMbTg52a2S28MGIic/53kHUHwwcrefwaidFZdEmNZd3BQpzuAI6iAkq8Ppwi2ATQAWaDDp+qBRsHig6h6KFMPMpil6iaQBNgtFhpGmvF3iSeIneAxGgTV2YGRepwoZsSTwBvQCMx2kSPVrEoikKRy08Tm5GeaXGhJsnZ/JxTyLj//3lUg5WUJlH0b9dMxh/qAVIg6giz2YzZXPmw5XL25ZfwxY78oPutBQhoUOQKoIs6HQxsarditVrZne9CpwsfJWkz6zEaDPx5YBYPf7CFwtJTFPgN+NCjBjszwt7a5ShCRcFIeUTBExAEtPK9OhyqDpfTh16nw6eCwWCksNTHiRI/0WYDvVoF3/ynvIJhnZtX6UHPahEfanrc0uOvWK3WXy0jqX2kQNRjzhy+bAi4iY8yUurVKPUGaBpjJiXOStvkmLIc5w4lOT1+jhV7UMXpsZECEGUPvlJegyDoQZxBoMx7CHZ4BntBfIEy+YgK1pZbNu7hpNMb1ssieyIaNlIg6jEFJaeDmHqvA6vRQJRZh06n0KlFbFg7/bIWsazZV1ChjstaBINQJ0t8WIw6An5QhMCgI+gVKGVxCF1QLIQI9jicEZII9kSUdUQIFBQ0TAYdep2O8pGVBaVeCkq9ZQO1To97MOrl5LMNmQY5UKqmKCgoICkpiUOHDtVYnbfddhszZ86skboSok83QfQBN22Toog2G4gxG2gWa2FY55TQ23lodgrtm8UQbTag0ylEmw20bxbD0OwUAHQKmAx6TDqBAQ2LUYfVqGDSB3spbEY9sWYFg8+J3l9aYUC1QrkUCGKNGimxFox6HRZj8BbylHWbxljCx0kUufxIGi6NWiCmTZvGDTfcQHp6OhAcoXn235IlS8LKrFy5ku7du2M2m8nMzGTBggVh+5988kmmTZtGcXHxBdt3eXo8yhmPahObkezUWCYO7sDtvVqFue6ZSdHc0Ted6zomc0VmItd1TOaOvumhPGkJUcTbjOgVgUJw8JTdbKBNgoXbuyXStqmFaLMevb8Eg8+BXgneHLqyblCdjrI0gQIkxpjpkdaEjMRoTAYd8VGmsJGcZ9osabg02iaGy+Xi73//O1999VVY+vz58xk8eHBoOy4uLvT74MGDDB06lHvvvZdFixaxYsUKxo4dS/PmzRk0aBAAnTp1ok2bNixcuJDx48dfkI2ZSdEM6ZTE+74SVIOVpBjzOaP7mUnREfcN75LC3JUuYgwa+P3YTDp0ikLH5GAwsGmUkUAgQIHqQ9MZsZl0uP3Bz71NegVVC46VsKhe0mzQKcWOXq8PeTHxUSZ25jqCsQi/itWoPytGImmINFqB+PzzzzGbzfTp0ycsPS4uLuxT8DN58803ycjICDUhsrKy+P7773n55ZdDAgEwfPhwlixZcsECAdA6MapGovvXdkhCDQR47T+5lCoadrOOTs2iaJMYrK9lnIlity80jiGjfTtynCoBTSAEmE3BUZCGU070SjTJdgt92jQNCdLl6fHkOTxhn5fLsQwNn0bbxFi1ahU9evSokD5+/HgSExPp1asX77zzDmcONF27di3XXXddWP5Bgwaxdu3asLRevXqxYcOGsJGS9YGr2zdlRIqTPvpDDMuKC4kDQEKUkQ6JJgy+EhRNJSnayDVt7PRpFUOXlCh6pkZzTaaddvp8rkxwcWvP1ApNnGGdU2gWa8Fk0FWIkUgaJo3Wgzh8+DApKSlhaU8//TTXXnstNpuNZcuWhaaOe/DBBwE4fvw4ycnJYWWSk5NxOBy43e7Q2z0lJQWfz8fx48crTC5Tn0mwGULeSpdmIzCYwsdpqGqAvHOUP1cTR9IwabQC4Xa7sVjCPzuePHly6He3bt0oLS1lxowZIYGoKuVC4XK5LtxQiaQOabRNjMTERIqKis6Zp3fv3vzyyy+hpkKzZs3Iywt/h+bl5WG328NiA4WFwQFO5VPcSSQNlUYrEN26dWPnzp3nzLNlyxaaNGkSGhLdt29fVqxYEZZn+fLl9O3bNyxt+/btpKamkpiYWLNGSyQXmUYrEIMGDWLHjh0hL+LTTz9l3rx5bN++nX379vHGG2/w3HPP8cADD4TK3HvvvRw4cICJEyeya9cu5syZw4cffsjDDz8cVveqVavC5qyUSBoqjVYgsrOz6d69Ox9++CEARqOR2bNn07dvX7p27crcuXN56aWXeOqpp0JlMjIy+Oyzz1i+fDldunRh5syZzJs3L6yL0+PxsHTp0rCJcyWShkq1gpT/+Mc/SExMZOjQoQBMnDiRt956i44dO/L+++83mMj9lClTePTRRxk3bhyDBw8OGyAViauvvprNmzdH3D9//nx69epVYXxFdbFYLHzxxReh35cqjeU8GxrV8iCee+65UFBu7dq1zJ49mxdeeIHExMQK7nZ9ZujQodxzzz3k5OTUWJ1Go5FZs2bVWH2KomC1Bj/pVpRL98OnxnKeDY1qeRBHjx4lMzMTgKVLl3LjjTdyzz330L9//9Bydg2FMxe9qQnGjh1bo/VJJHVJtTyI6OhoCgqCnxYvW7aM3/72t0DQNXS7I6ybIJFIGhzV8iB++9vfMnbsWLp168aePXu4/vrrAdixY0foy0iJRNLwqZYHUR7tP3HiBB9//DEJCcE5CTdt2sTtt99eowZKJJK6o1oeRFxcHK+//nqF9POZpVkikdR/qj0OYtWqVYwePZp+/fqFegHee+89vv/++xozTiKR1C3VEoiPP/6YQYMGYbVa+fHHH0PfKhQXF/Pcc8/VqIESiaTuqJZAPPvss7z55pu8/fbbGI2npxTr378/P/74Y40ZJ5FI6pZqCcTu3bu56qqrKqTHxsZy6tSpC7VJIpHUE6olEM2aNWPfvn0V0r///ntat259wUZJJJL6QbUEYty4cTz00EOsX78eRVHIzc1l0aJFTJgwgT/96U81baNEIqkjqtXN+dhjj6FpGgMGDMDlcnHVVVdhNpuZMGFC2OfRkvqLUFXUs9JUNRD2W1H1FcpIGhfVEghFUXjiiSd49NFH2bdvHyUlJXTs2JHoaDkfYUPh53XfVEjTzhCIXeu+QadvtDMSXlK4XC6ysrK4+eabefHFF8+r7AXdASaTiY4dO15IFZI6oEebyqf19/v97CyboLtrRnJYD5Wk4TJt2rRqTz9QZYEYMWIECxYswG63M2LEiHPm/de//lUtYyS1i8FgYPTo0RH3CyG47bbbgOCHd+f67NpgkN5FQ2Dv3r3s2rWL4cOHs3379vMuX+UgZWxsbOiGiY2NPeefpH6iKApGozHin8lkwm63Y7fbMZlM58zbmOds+O677xg+fDgpKSkoisLSpUtrrY7Zs2eTnp6OxWKhd+/ebNiw4byOM2HCBKZPn37e9pVT5dfA/PnzK/0tkVzqFBUVYTQaQzG20tJSunTpwh//+Mdf9aYjUZU6PvjgAx555BHefPNNevfuzSuvvMKgQYPYvXs3SUlJAHTt2pVAIFCh7LJly9i4cSPt2rWjXbt2rFmzplp2IqrBgQMHxJ49eyqk79mzRxw8eLA6VTZ6iouLBSCKi4vr2pRLjmPHjglAvPLKK6Jr167CbDaLjh07ilWrVkUs4/f7xX//+19x0003CbPZLLZs2VJpPkD8+9//DkvLz88XycnJYtq0aaG01atXC6PRKL7++usq1SGEEL169RLjx48PbauqKlJSUsT06dN/5YyDPPbYYyI1NVWkpaWJhIQEYbfbxdSpU8/rXquWQFx11VViwYIFFdLfe+898Zvf/KY6VTZ6pEDUHl988YUAROfOncXKlSvFzz//LAYPHixatWolVFUNy7tt2zbxyCOPiOTkZBEfHy/+9Kc/iTVr1kSsO9LD/dlnnwmj0Sg2btwoHA6HaN26tXj44YerXIfX6xV6vb5C+h133CF+97vfVem8z2T+/PniL3/5ixDi/O61akWaNm/eTP/+/Suk9+nTh/vvv796roxEUkts3boVo9HIf/7zn9CERs8++yw9e/YkJycHm83GwoUL+cc//sGOHTu4/vrrmTNnDsOGDcNkMp278ghcf/31jBs3jlGjRtGzZ0+ioqLOKxZw8uRJVFWtdKnHXbt2Vcum6lDtcRBOp7NCenFxMaocTCOpZ2zZsoURI0aEzXZmt9tDv2fNmsXUqVO58sor2bdvHy1btqyR47744ot06tSJf/7zn2zatCm0AFNdcOedd1arXLWGWl911VVMnz49TAxUVWX69OlcccUV1TJEIqkttmzZQteuXcPS1q5dS2JiIi1atOCee+7hmWee4fjx41x22WXcddddfPPNN2iadkHH3b9/P7m5uWiaxqFDh86rbGJiInq9vtKlHps1q3wcS21QLYF4/vnn+eabb2jfvj133XUXd911F+3bt+e7775jxowZNW2jRFJt3G43e/fuDXuZaZrGK6+8wpgxY9DpdKSkpPDkk0+yZ88evvzyS0wmEyNGjCAtLY3HHnuMHTt2nPdxfT4fo0eP5tZbb+WZZ55h7Nix5OfnV7m8yWSiR48eYUs9aprGihUrKiz1WKucd7SjjJycHPH444+L66+/Xtx4441i6tSpoqCgoLrVNXpkkLJ2WL9+vTAYDKJDhw5izZo1YufOneKmm24SGRkZoqioKGI5t9st3n//fTFo0CCh1+vFtm3bQvucTqfYvHmz2Lx5swDESy+9JDZv3iwOHz4cyjNhwgSRnp4uiouLhaqq4oorrhBDhw49rzqWLFkizGazWLBggdi5c6e45557RFxcnDh+/PgFXZNa78WQ1DxSIGqHuXPnik6dOol3331XNG/eXNhsNvGHP/xBHDlypMp15OTkhP2/fPvttwKo8DdmzJjQfoPBENaNevDgQWG328WcOXOqVEc5s2bNEq1atRImk0n06tVLrFu3rvoXo4zzudcUIYSojuexatUq5s6dy4EDB/jnP/9JixYteO+998jIyJBxiGrgcDiIjY2luLg4LIAmuTDGjx9PUVERixcvrmtT6g3nc6/JOSkllwxCCPx+f9jf5s2bueyyyyqk+3w+HA4HDocDn89XYX9V/6r5fm0wVKubs3xOyjvuuIMlS5aE0vv378+zzz5bY8ZJJOdDIBBg4cKFoW0hBJs3b6Z3795h6RD8crXcqxg5cmS1v1wdPXr0Jf3Va7UEQs5JKanPbNp/PPT77r9MwX9WGoTPfbHlYF615r6I9Nn8pUS1BKJ8Tsqzl9mTc1JK6gtZfa5F0esj7g/4vOxcG+xC7NDnWgymqg9iEqpa6YQ7lyLVEojyOSnfeeed0JyUa9euZcKECUyePLmmbZRIzhtFr0d/Dq9A6E+Pi9DrDefMezaNaaywnJNSIpFERM5JKZFIInLBc1LGxMQQExMjxUEiuQSp1jiIQCDA5MmTiY2NJT09nfT0dGJjY3nyySfx+/01baNEIqkjquVBPPDAA/zrX//ihRdeCH04snbtWv76179SUFDAG2+8UaNGSiSSuqFaArF48WKWLFnCkCFDQmmdO3emZcuW3H777VIgJJJLhGo1Mcxmc4UxEAAZGRnVnoFHIpHUP6olEPfffz/PPPNM6BsMAK/Xy7Rp0+SUcxLJJUS156RcsWIFqampdOnSBQjO++fz+RgwYEDYNN5yER2JpOFSLYGIi4vjxhtvDEurqXn8JBJJ/aFaAjFnzhw0TSMqKgqAQ4cOsXTpUrKyshg0aFCNGiiRSOqOasUgbrjhBt577z0ATp06RZ8+fZg5cya///3vZQ+GpFKEELjdbtxu9yU/h0JdUtPXuVoC8eOPP3LllVcC8NFHH5GcnMzhw4d59913ee211y7YKMmlh8fjYciQIQwZMgSPx1PX5lyy1PR1rpZAuFwuYmJigOAagCNGjECn09GnTx8OHz58wUZJJJL6QbUEIjMzk6VLl3L06FG++uorBg4cCEB+fr6cT1EiuYSolkBMmTKFCRMmkJ6eTu/evUPDrZctW0a3bt1q1ECJRFJ3VEsgbrrpJo4cOcIPP/zAl19+GUofMGAAL7/8co0Y9te//rXCakjVYeXKlSiKcl5T4d155538/ve/v+BjSyQNnWp/7t2sWbMKS4D16tXrgg0qp6Ymn+nXrx/Hjh0jNja2ymVeffVVGWm/xCgo9XP0lI9Sv0aUUUfzqLq2qGFwQfNB1CbR0dHnnGPC5/NV6bsPk8l03msZno+YNGT25Zew8VAhBSVeEqLNXJ4eT2ZS3czrUZu2FJT62ZnnDm07vSrFbhWfJR6Tp7BGjnGpUq0mRk3w1ltvkZKSUmGB1BtuuIE//vGPFZoY5W7/tGnTSElJoX379gCsWbOGrl27YrFY6NmzJ0uXLkVRFLZs2QJUbGIsWLCAuLg4vvrqK7KysoiOjmbw4MEcO3aswrHK0TSNF154gczMTMxmM61atWLatGmh/ZMmTaJdu3bYbDZat27N5MmT6/28GPvyS/h0ay7Hiz34VcHxYg//3ZbLvvySemnLN7vyefiDLYyet46HP9jCN7vOvc5lgSvAlpxSVh9ysuaQE5f/rJkkhcBrT62N07mkqDMP4uabb+aBBx7g22+/ZcCAAQAUFhby5Zdf8vnnn7Nq1aoKZVasWIHdbmf58uVAcIWg4cOHc/3117N48WIOHz7Mn//85189tsvl4sUXX+S9995Dp9MxevRoJkyYwKJFiyrN//jjj/P222/z8ssvc8UVV3Ds2DF27doV2h8TE8OCBQtISUnhp59+Yty4ccTExDBx4sRqXJmLw8ZDFd+cQgTTL7YX8Wu2fLMrn7n/2x/al3vKzVvfBbev7ZBUoWyJMFOY70GnBN9/Tq+K5oWmUWAznp7pWjVYq2Tf2c2TFHvk2bIvBhfT86szgWjSpAlDhgxh8eLFIYH46KOPSExM5JprrqlUIKKiopg3b16oafHmm2+iKApvv/02FouFjh07kpOTw7hx4855bL/fz5tvvkmbNm2A4NepTz/9dKV5nU4nr776Kq+//jpjxowBoE2bNmHLCz755JOh3+np6UyYMIElS5bUa4EoKPFWml5Y6rvIlgRtKSz1kXvKjcsXwGYykBJnxWQIPuCfbs2tUMblVXltxV525haHHpK0JsGp6wtEFHq/Rok3gF8VuPwaegUcHjUkEC6/ht+awKmUPmw97iE9QUdCVMUFcCprnuzK92MUVZ8mvyYp97bKKfe2hnVOqRWRqNMYxKhRoxg3bhxz5szBbDazaNEibrvtNnS6yls+2dnZYXGH3bt307lzZywWSyitKoFSm80WEgeA5s2bR1ya/eeff8br9YZErDI++OADXnvtNfbv309JSQmBQKDejwdJiDZzvLjiSLv4qIs/n4eqwZ48Z2i7xBtgT56TxOjgQ3jCGW6n26dSUOpFr1PCmiSDOwa9CacwEygNoKAAoFcUSnwqbr+GTw0Gn30BFUX1InQ6SrwqO/PddEwiJBLlXsO+Ag9CCOwWfUhcBEERikRtvuF/zds6cLIUZ1JnVIONDzfl0L9dsws6dp3FIACGDx+OEILPPvuMo0ePsmrVKkaNGhUxf/nHYRfK2UulKYoSsdfCaj23G7p27VpGjRrF9ddfz3//+182b97ME088gc938d/E58Pl6fEoChw9epSPPvqIo0ePoijB9IuOEqHHqCy9aYwlLNnpCcZ3bKbT7zch4IfDRQD4K7z3BEIIVAEK4PZrwac8PAtHTwX/zwpK/fxwtIS9J92cKPGTX+Jn30kv+ws9HHf6cPk1fFTezKjt2E65t7U9p5gNBwvYnlNMYamPwlIf+/JL+Hx7HgFTNEKnI9/pveBj16lAWCwWRowYwaJFi3j//fdp37493bt3r3L59u3b89NPP4VNXLNx48YatbFt27ZYrVZWrFhR6f41a9aQlpbGE088Qc+ePWnbtm2DGG6emRTNwPYJbPp+BaXOYjZ9v4KB7RPqpBdDryi0S44h2mxAp1OINhtolxyDocyTHN4lBUU5nd+vCiBY5kzKm0cmAnj8GgWu4MNd6A6glXkBLePMWI06DHoF1Rzu5bn8wYD5z3luTpQG8AYEmhB4AhqegEapV8UbEBSUBtCEQmWc6w1fzr78Et7fcITXv9nL+xuOnNcDXO5tlXgDaOK0txVQRZWOfb7UeTfnqFGjGDZsGDt27GD06NHnVXbkyJE88cQT3HPPPTz22GMcOXKEF198EQh6BTWBxWJh0qRJTJw4EZPJRP/+/Tlx4gQ7duzg7rvvpm3bthw5coQlS5Zw+eWX89lnn/Hvf/+7Ro5d26xdthTdwbXECYGiKKxb/h86/vGPF92OhGgzflVUaN6Ub5cHIj/dmsvJEi+JMWZaxdvISDztURaW+nC4fTgKbBQLgdunlTkJAlUDdKe9RJNewaOBpgv3JG3GoCAdL/HjVwVeVcMbCJbXKQLfGR0hke6uymI7haU+fj7moKDEi6rByRJv6NzOO4agCNx+Fafbj1/VMOp1xFiNoIhaiSvVuUBce+21xMfHs3v3bkaOHHleZe12O59++il/+tOf6Nq1K9nZ2UyZMoWRI0eGxSUulMmTJ2MwGJgyZQq5ubk0b96ce++9F4Df/e53PPzww9x///14vV6GDh3K5MmT+etf/1pjx68NfvnlFxYvXhx6aIQQLF68mIEDB5KaWvPdf2c24c7+yrBzcxs5BSWIM/x+BYXOzW243cEAYd+0GPqmBbu2D5ws5Yvt+QQCwaZGkcvP3vxSWidYKQhoeDCiCoFRr6BXdChq0BPwBoL12y16PH4VnXZGV7QCLeOCD60voIV1i+qUYItECIHZoGDQKRQKMyvzdYg1B+jdOoHWZWIVY1LId55+IItcPvbmlxJlMuD2+tiR66TUp9I2yUYT22lBXL3nOC1iWvzqdTzpcKNpKkIIRFnTSdM0Chxu2jSNwll6elHi8kt+IXElRVxiQwYXLVrEXXfdRXFx8a/GD+oSr9cb1jRyOBy0bNmS4uLiWg9wCiGYOHEiP/74I6p65hqVerp3784LL7xQYx5YOUVFRfzhD3+IuN9nicdrT0U1WNEH3Jgdv5xzENOZ+f3WBBTViz4QFJOSxI5YYpogUDDqFAQEBUOno1MzGwAFpV4O7d6B0Blo3bYDnZpH0SYxeL/8c+tJ8kuC4uErExeAaJOe9HgzOac8FOYfRxdwo9P8GDwOYvI2Y/IU4rPE40roQHkLxBeVjGqIQtH8gEA1RqNoPgz+UkyleaHzUTSVuNz1v3KuNlxN2iAUJVzcAIPXSWzOurBj33brrVit1greicPhIDY2tkr3Wp17EBfKu+++S+vWrWnRogVbt25l0qRJ3HLLLfVaHACmT5/O1KlT6+TYR44cqTRWo6oqGzdu5MiRI6SlpdXoMQ8WuELRdX3AVUEATJ7Cao5qVNB0JvTqabHVqcGHRxPBN79ep2BSFEx6HXqdgiaCb14UBaHo8AY09hV4iLMaSIgyEmc1UOoLxh0MAgIaGHQKFqOOk6UBXH4BIgCKgqY34bMlUhrfHlPu2uA5FOwKiZdQDGg6AxjMCEWH0BlBmFDE6QGCqsGK0Js5ldKnwrXxWeIpTewQyisQaAYbBDgtEgJ0fleFYydFm/nNZRfW/dngBeL48eNMmTKF48eP07x5c26++eawUY71lccff5xHHnkktF3uQVwMWrVqxeWXX16pB9GjRw9atWpVo8fbl1/C13tOETAFb9Sb/r+7MRqMDOmUFHLND5ws5YfDpygs9RMfZSTZbibP4Q1t90yLC8v7+fbTb99yt711gpWcA7vZfUrFoSmhwKaqCTRFoWeqjW6p0Xx/wEGJT6Dpg663VxWcLA3wc56bK1obaRplxOVTOVEq0EQwZmE2KESb9JzyBDAZFNToeIx6A7H2aOxWI8kx3Xn9tikVzv2+97eyO+90ENKvBmMIidFJDO3Uv6wJ4qJtUhRNbMGYiIISujYfbsoh33la/HYcc5JT5KbYE8Ck12Ez6WmbZKNXejy39JiEx+MJeWq3934Km812Qf93DV4gJk6cWK8HJEXCbDZjNtfNYBtFUXjooYdCA7/OTq/p5sXGQ4VhvRAGgwG9wcC2Yy4ua5nIvvwSlu0q8x50evad9PD5jhO0S44hPspEoVtj+e5ChpktZCZFs+3YCQyG0wHGlgnRbD16ii05TrxeCx5UrEYFnaKgCcpiEbD2cAnrjpTgDmhYz+qlFALyypoVMWYdJT4Ni0GHxVB+baBrio31R0oodGto6EBRCGiCIpefWKupUq/Vp4Ki6CjvVzUZgnbpdTpsFjO5Dj8dmsdWiBOUXxunT4Sdq91qZm9+KSaDnhZxweMVuFRaJtorHL8m/h/rtJtTUnekpqYycuTI0E2kKAojR46kRYtfD5SdL5Ei+9/uyuf1b/Yy+9t9YZH23FPusH8hvLsuUrTepwbddo1gIDHarMNu0YMIfpvhCahoQhBQBU6fIGCofFyN06sRbdThCWg4vSqeQHDb6dUw6ZVKejAUrKbKx0VEmfUkRJkw6XUoZc2cZLuFNk2jGX9NJm2aRlUaRCy/HgnR4S+RUm+AhCgz0WYD+rIu4bZJMeQ5amcaPykQjZhRo0aRkJAAQGJi4nn3IlWVs2/yIpefPXlOVE2EBhPtyXOGHgpXWX+i+6wPrCI9NLmn3FhNetITbKTZ/JgJ4FUFTq+GEFDsDaCK02Oj9LpgfEI12wmY7JT4NPyaIDkm+KY+UerHWeZBxJj1WAw6SvwaJ0v9xFoN2Iw6dGgoCEwGHQlRJlrFV+7KX9YiFqtJT5LdQos4K0l2C1aTnsta2Cs9l3LKRaN8QFs5Lp+K1aTn8vR4Lk+Pp1OLoPdRW0PkG7VAFBQUkJSUxKFDh2qszttuu42ZM2fWWH21icVi4ZFHHiE5OZmHH364RruGz+Ty9HjOfO/mngq+7VLKXGRb2du33GMo37Yaw9/K53poAFJiT7vYClD+obAa/sEweqVsHIOiC8vfvEwgPP6zChD0YNx+jaZRRprZjUTjI86okZEYRZeWcbQ9a9BWOUOzU2jfLHwQWPtmMQzNTqn0XICwEa2ZSdEM65xCs1gLJoOOZrGWUNOrsmtT0zRqgZg2bRo33HBDaJ1RRVEq/C1ZsiSszMqVK+nevTtms5nMzEwWLFgQtv/JJ59k2rRpFBcXX6SzuDD69evHBx98QL9+/WrtGJlJ0QzplITBV4Kiqeh1SthNXi4U5R5DSpwVRTmdDlV9aMqDfGA16sra+8EeiJAoAAIFvQ4UzY/BW0y8VU9zuxGnNygMFqOu4kgoJVhnyzgTUUYdsYqbNJufTil2EqJNEYeoZyZFc0ffdK7rmMwVmYlc1zGZO/qmh3oWKjuXs7slM5Oiub1XK8Zfk8n4azJJiA4Xg9ocIt/gg5TVxeVy8fe//52vvvoqLH3+/PkMHjw4tB0XFxf6ffDgQYYOHcq9997LokWLWLFiBWPHjqV58+ahBYM6depEmzZtWLhwIePHj78o59IQaJ0YRUz+NgCuajuOQvfpt3R8lIl2yTGUeAOYDDo6pti5pkMSeQ4PhaU+4qNMFT54ykyKDm3vyy/hv9tyQz0yeoKxgqZRRmwmPQYFfnH6TwtEsIcTs+MXzK48mkV3QKfXh4ZaN40yogmBw6PiV4MDruwWPYlRRhKijHRIsuA46kevCJLtFvq0aXrOrsQzba3O/rPzDuucwsZDhRGvTU3SaAXi888/x2w206dPn7D0uLi4iDNQvfnmm2RkZISaEFlZWXz//fe8/PLLYSuKDR8+nCVLlkiBiEDPtDiW7y7kzCF6CdEmxvRLr9aNXv7QrNt/Ar0iiFNcREUlh5oqzWPNoIDDq+FXBRaDQoJNoeBwXlg95UOtW8aZcPrUsLkjzhxpmWAzkKYrpEeCiVt7plb4+K+2OR9BuVAabRNj1apV9OjRo0L6+PHjSUxMpFevXrzzzjthQ4TXrl3LddddF5Z/0KBBrF27NiytV69ebNiwIWykpOQ0rROjftWtPl8yk6K5tWcqVya4aKfPp3sLGzFmPXqdQoxZz4C2cYztncyf+jXj1q4JNIs5q81+pgBEGemYZA0r3zHJWul8EZc6jdaDOHz4MCkpKWFpTz/9NNdeey02m41ly5Zx3333UVJSwoMPPggEB2UlJyeHlUlOTsbhcOB2u0P90CkpKfh8Po4fP17jIxIvFWr7LZhgM5AUU3nQNSHKSIdEE7t9JagGK9FmPekJ4QKQUNacaOw0WoFwu90VovaTJ08O/e7WrRulpaXMmDEjJBBVpVwoXC7XhRsqqRUSbIZQTKRLsxEYTFIMKqPRNjESExMpKio6Z57evXvzyy+/hJoKzZo1Iy8vvN2al5eH3R4+iq2wMDigp2nTpjVstURycWm0AtGtWzd27tx5zjxbtmyhSZMmoSHRffv2rTBxzPLly0Mri5Wzfft2UlNTSUxMrFmjJZKLTKMViEGDBrFjx46QF/Hpp58yb948tm/fzr59+3jjjTd47rnnwhbvuffeezlw4AATJ05k165dzJkzhw8//JCHH344rO5Vq1aF1iuVSBoyjVYgsrOz6d69Ox9++CEQnKdy9uzZ9O3bl65duzJ37lxeeuklnnrqqVCZjIwMPvvsM5YvX06XLl2YOXMm8+bNC+vi9Hg8LF269Fdn1pZIGgKNNkgJwUWIH330UcaNG8fgwYPDBkhF4uqrr2bz5s0R98+fP59evXpVGF/R2LFYLHzxxReh35Laoaavc6MWiKFDh7J3715ycnJqbC4Go9HIrFmzaqSuSwlFUer9JD6XAjV9nRu1QABVWonrfBg7dmyN1ieR1CWNNgYhkUh+HSkQEokkIlIgJBJJRKRASCSSiEiBkEgkEZECIZFIIiIFQiKRREQKhEQiiYgUCIlEEhEpEBKJJCKNfqi15NJEqCrqOfaraiDst6JWvjJWpLobC1IgJJckP6/75pz7tTMEYte6b9Dp5aNQGfKqSC45erSpfNmCM/H7/ewsm4y8a0byRZ+6vqGgiDPndZfUGQ6Hg9jYWIqLi7Hb7XVtToNECEEgEPj1jGV5PZ7gEoAWi6XaK2EbDIYaXw29tjmfe016EJJLBkVRzssTMJlqZz3LSwnZiyGRSCIiBUIikURECoREIomIFAiJRBIRGaSsJ5R3Jjkcjjq2RHKpU36PVaUDUwpEPcHpdALU2OzaEsmv4XQ6iY2NPWceOQ6inqBpGrm5ucTExDS4fvWLhcPhoGXLlhw9elSOFTkPzr5uQgicTicpKSnodOeOMkgPop6g0+lITU2tazMaBHa7XQpENTjzuv2a51CODFJKJJKISIGQSCQRkQIhaTCYzWaeeuopzGZzXZvSoLiQ6yaDlBKJJCLSg5BIJBGRAiGRSCIiBUIikURECoREIomIFAhJgyAnJ4fRo0eTkJCA1WolOzubH374oa7NqteoqsrkyZPJyMjAarXSpk0bnnnmmSp9g1GOHEkpqfcUFRXRv39/rrnmGr744guaNm3K3r17adKkSV2bVq95/vnneeONN/jHP/7BZZddxg8//MBdd91FbGwsDz74YJXqkN2cknrPY489xurVq1m1alVdm9KgGDZsGMnJyfz9738Ppd14441YrVYWLlxYpTpkE0NS7/nkk0/o2bMnN998M0lJSXTr1o233367rs2q9/Tr148VK1awZ88eALZu3cr333/PkCFDql6JkEjqOWazWZjNZvH444+LH3/8UcydO1dYLBaxYMGCujatXqOqqpg0aZJQFEUYDAahKIp47rnnzqsOKRCSeo/RaBR9+/YNS3vggQdEnz596siihsH7778vUlNTxfvvvy+2bdsm3n33XREfH39ewiqDlJJ6T/PmzenYsWNYWlZWFh9//HEdWdQwePTRR3nssce47bbbAMjOzubw4cNMnz6dMWPGVKkOGYOQ1Hv69+/P7t27w9L27NlDWlpaHVnUMHC5XBUmhNHr9WiaVuU6pAchqfc8/PDD9OvXj+eee45bbrmFDRs28NZbb/HWW2/VtWn1muHDhzNt2jRatWrFZZddxubNm3nppZf44x//WPVKarEJJJHUGJ9++qno1KmTMJvNokOHDuKtt96qa5PqPQ6HQzz00EOiVatWwmKxiNatW4snnnhCeL3eKtchx0FIJJKIyBiERCKJiBQIiUQSESkQEokkIlIgJBJJRKRASCSSiEiBkEgkEZECIZFIIiIFQlIvufPOO/n9739fpbxXX301f/7zn2vVnqqycuVKFEXh1KlTdW1KjSAFQiKpJvVJmGoLKRASiSQiUiAklfLRRx+RnZ2N1WolISGB6667jtLSUgDmzZtHVlYWFouFDh06MGfOnFC5Q4cOoSgKS5YsoV+/flgsFjp16sT//ve/UB5VVbn77rtDk6m2b9+eV199tcZs93q9TJgwgRYtWhAVFUXv3r1ZuXJlaP+CBQuIi4vjq6++Iisri+joaAYPHsyxY8dCeQKBAA8++CBxcXEkJCQwadIkxowZE2r23Hnnnfzvf//j1VdfRVEUFEXh0KFDofKbNm2iZ8+e2Gw2+vXrV+Fr1AZDrX0pImmw5ObmCoPBIF566SVx8OBBsW3bNjF79mzhdDrFwoULRfPmzcXHH38sDhw4ID7++OOwSUgOHjwoAJGamio++ugjsXPnTjF27FgRExMjTp48KYQQwufziSlTpoiNGzeKAwcOiIULFwqbzSY++OCDkA1jxowRN9xwQ5Xs/c1vfiMeeuih0PbYsWNFv379xHfffSf27dsnZsyYIcxms9izZ48QQoj58+cLo9EorrvuOrFx40axadMmkZWVJUaOHBmq49lnnxXx8fHiX//6l/j555/FvffeK+x2e8imU6dOib59+4px48aJY8eOiWPHjolAICC+/fZbAYjevXuLlStXih07dogrr7xS9OvX7wL+R+oOKRCSCmzatEkA4tChQxX2tWnTRixevDgs7ZlnngnN+FQuEH/7299C+/1+v0hNTRXPP/98xGOOHz9e3HjjjaHt6grE4cOHhV6vFzk5OWF5BgwYIB5//HEhRFAgALFv377Q/tmzZ4vk5OTQdnJyspgxY0ZoOxAIiFatWoXZdLYwCSFCAvH111+H0j777DMBCLfbXaXzqU/I+SAkFejSpQsDBgwgOzubQYMGMXDgQG666SZMJhP79+/n7rvvZty4caH8gUCA2NjYsDr69u0b+m0wGOjZsyc///xzKG327Nm88847HDlyBLfbjc/no2vXrhds+08//YSqqrRr1y4s3ev1kpCQENq22Wy0adMmtN28eXPy8/MBKC4uJi8vj169eoX26/V6evToUeXJVjp37hxWN0B+fj6tWrU6/5OqQ6RASCqg1+tZvnw5a9asYdmyZcyaNYsnnniCTz/9FIC3336b3r17VyhTVZYsWcKECROYOXMmffv2JSYmhhkzZrB+/foLtr2kpAS9Xs+mTZsq2BQdHR36bTQaw/YpinJeC8r8GmfWrygKwHnN5FRfkAIhqRRFUejfvz/9+/dnypQppKWlsXr1alJSUjhw4ACjRo06Z/l169Zx1VVXAUEPY9OmTdx///0ArF69mn79+nHfffeF8u/fv79G7O7WrRuqqpKfn8+VV15ZrTpiY2NJTk5m48aNoXNQVZUff/wxzMsxmUyoqloTZtdbpEBIKrB+/XpWrFjBwIEDSUpKYv369Zw4cYKsrCymTp3Kgw8+SGxsLIMHD8br9fLDDz9QVFTEI488Eqpj9uzZtG3blqysLF5++WWKiopCU521bduWd999l6+++oqMjAzee+89Nm7cSEZGxgXb3q5dO0aNGsUdd9zBzJkz6datGydOnGDFihV07tyZoUOHVqmeBx54gOnTp5OZmUmHDh2YNWsWRUVFIW8AID09nfXr13Po0CGio6OJj4+/YPvrG1IgJBWw2+189913vPLKKzgcDtLS0pg5c2ZowRWbzcaMGTN49NFHiYqKIjs7u8KAob/97W/87W9/Y8uWLWRmZvLJJ5+QmJgIwP/93/+xefNmbr31VhRF4fbbb+e+++7jiy++qBH758+fz7PPPstf/vIXcnJySExMpE+fPgwbNqzKdUyaNInjx49zxx13oNfrueeeexg0aFBYs2XChAmMGTOGjh074na7OXjwYI3YX5+QU85JapRDhw6RkZHB5s2bayToWF/QNI2srCxuueUWnnnmmbo256IhPQiJpBIOHz7MsmXL+M1vfoPX6+X111/n4MGDjBw5sq5Nu6jIkZSSes2RI0eIjo6O+HfkyJFaOa5Op2PBggVcfvnl9O/fn59++omvv/6arKysWjlefUU2MST1mkAgEDaE+WzS09MxGKQjXFtIgZBIJBGRTQyJRBIRKRASiSQiUiAkEklEpEBIJJKISIGQSCQRkQIhkUgiIgVCIpFERAqERCKJyP8DMc+wwq1JjFAAAAAASUVORK5CYII=" />
    



```python
# plot
from roux.viz.dist import plot_dists
ax=plot_dists(df1,x='species',y='sepal_length',colindex='id',kind=['box','strip'])
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/dists_annotated_vertical.png')
```

    WARNING:root:overwritting: tests/output/plot/dists_annotated_vertical.png



    
<img src="data:image/png;base64,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pdLQCDPa9C9COdcZxDVTs0WLFpGYmEhMTAy/+c1vgs5aUlWVJUuW0KNHDywWC+np6XzwwQdB7e/bt49bbrmFmJgYoqOjufrqqwMH2dV3nu752istLeXuu+8mMTERi8VCnz59WL16dfN2UASjhSlFICdPnmTSpEn86U9/4vbbb6eyspINGzYEtpyvX78es9nM119/TU5ODtOnTychISGQSnnJkiWsWbOGlStX0qdPH7799lumTJlCYmIio0eP5sSJE/z0pz/lmmuu4csvvyQmJoaNGzfi8/nqledC7T355JPs37+fjIwMOnbsSHZ2Nk6ns8X6K+JovbBDjXNxvjOIp02bJuLj44XD4QiUvfLKK8JmswlFUYTL5RJRUVFi06ZNQffde++9YtKkSUIIIZ544gnRo0cP4fF46n1+7WNqGtLeuHHjxPTp00P+e9sb2kgVgdQ+g3js2LFcf/313HHHHXTo0CFwPSoqKlB/5MiR2O128vLysNvtVFVVcd111wW16fF4GDp0KODfFHj11VfXOQK0PrKzsy/Y3gMPPMAvf/lLduzYwfXXX88vfvELrrzyyib1QVtGU6oIpOYM4k2bNvH555+zfPly5s2bx9atWy94r91uB+Czzz6jc+fOQddqdtY25gibhrR34403cuzYMf7973/z3//+lzFjxjBz5szAiYkXHa09VGpcGJ/PJzp37iz+/Oc/B6Z/VVVVgesrV64MTP8qKiqEyWQSb7755jnbe+qppxo8/WtIe2ezcuVKER0d3eD67Q1tpIpAzncG8e7du/F4PNx7773Mnz+fnJwcFi5cyEMPPYROpyM6OppHH32UWbNmoaoqV111FeXl5WzcuJGYmBimTZvGQw89xPLly5k4cSJPPPEEsbGxbNmyhREjRtCvX78gWRrS3oIFCxg2bBiXXHIJbrebtWvXMmDAgFbqvQigtbVaoy7nO4O4ZhRZsGBB4Azb+++/X7hcrsD9qqqKZcuWiX79+gmDwSASExPF2LFjxTfffBOos2vXLnH99deLqKgoER0dLa6++urAeb5nn6d7ofaefvppMWDAAGGxWER8fLy47bbbxJEjR1qgpyITzfnbxrjnnnsoKyvjk08+aW1RNM6B5vzV0AgzmlJpaIQZbfqnoRFmtJFKQyPMaEqloRFmNKXS0AgzmlJpaIQZTak0NMKMplQaGmFGUyoNjTCjKZWGRpj5/6BsldQwjFibAAAAAElFTkSuQmCC" />
    


#### Documentation
[`roux.viz.dist`](https://github.com/rraadd88/roux#module-roux.viz.dist)

### Example of annotated barplot


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')

# plot
from roux.viz.bar import plot_barh
ax=plot_barh(df1.sort_values('sepal_length',ascending=False).head(5),
          colx='sepal_length',coly='species',colannnotside='sepal_length')
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/bar_annotated.png')
```


    0it [00:00, ?it/s]


    INFO:root:Python implementation: CPython
    Python version       : 3.7.13
    IPython version      : 7.34.0
    sys       : 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0]
    pandas    : 1.3.5
    matplotlib: 3.5.3
    re        : 2.2.1
    logging   : 0.5.1.2
    seaborn   : 0.12.1
    scipy     : 1.7.3
    tqdm      : 4.64.1
    numpy     : 1.21.6
    
    WARNING:root:overwritting: tests/output/plot/bar_annotated.png


    INFO: Pandarallel will run on 6 workers.
    INFO: Pandarallel will use standard multiprocessing data transfer (pipe) to transfer data between the main process and workers.



    
<img src="data:image/png;base64,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" />
    


#### Documentation
[`roux.viz.bar`](https://github.com/rraadd88/roux#module-roux.viz.bar)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_viz_io.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for the input/output of visualizations.
<details><summary>Expand</summary>

### Saving plots with the source data


```python
# demo data
import seaborn as sns
data=sns.load_dataset('iris')
```


```python
# import helper functions
from roux.viz.io import *

## parameters
kws_plot=dict(y='sepal_width')
## log the code from this cell of the notebook
begin_plot()
fig,ax=plt.subplots(figsize=[3,3])
sns.scatterplot(data=data,x='sepal_length',y=kws_plot['y'],hue='species',
                ax=ax,)
## save the plot
to_plot('tests/output/plot/plot_saved.png',# filename
       data=data, #source data
       kws_plot=kws_plot,# plotting parameters
       )
assert exists('tests/output/plot/plot_saved.png')
```

    WARNING:root:overwritting: tests/output/plot/plot_saved.png


    Activating auto-logging. Current session state plus future input saved.
    Filename       : log_notebook.log
    Mode           : over
    Output logging : False
    Raw input log  : False
    Timestamping   : False
    State          : active



    
<img src="data:image/png;base64,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" />
    


### "Reading" the plots


```python
read_plot('tests/output/plot/plot_saved.png')
```


    0it [00:00, ?it/s]


    INFO:root:Python implementation: CPython
    Python version       : 3.7.13
    IPython version      : 7.31.1
    sys       : 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0]
    seaborn   : 0.11.2
    matplotlib: 3.5.1
    re        : 2.2.1
    tqdm      : 4.64.1
    logging   : 0.5.1.2
    scipy     : 1.7.3
    pandas    : 1.3.5
    numpy     : 1.18.1
    
    INFO:root:shape = (150, 5)


    INFO: Pandarallel will run on 6 workers.
    INFO: Pandarallel will use standard multiprocessing data transfer (pipe) to transfer data between the main process and workers.





    <AxesSubplot:xlabel='sepal_length', ylabel='sepal_width'>




    
<img src="data:image/png;base64,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008BePfddznttNOadQD7nXfeoX///hx11FHhsnHjxmGxWMLXmzBhAvfccw8vvPBCr7VOMsTMwKAVCILA0qVLGTJkCKeeeiqjR4/ml19+Cdc3zutan7uisrKSgoICAPLy8hAEIfzvo48+Ctft3buXPn36NKsvBQUF/PbbbxHXEkURv98fvt6rr77K8ccfzyWXXMLgwYNZvXp1m1+D7oaxZmbQakq9CjsqA2wq8ZHhkBmebiPVJiM305usp9OnTx+WLFnCt99+y6mnnsqsWbO45557AKIW+3ft2gWE/Pm8Xi8An376aVTy6cTERACSk5PZuXNns/qRmJjIwIEDeeONN6LqsrOzgZBv2Isvvsj8+fM544wzOP3009m5c2evWjszxMygVRTWBlnw2R4Kavd9aU2iwO0Tshjith4wggYwZswYzjrrLJ5++ulw2fvvv8/kyZPDf7/66qs4HA6OOuoovF4vJpOJ3bt3M2nSpJjXPOGEE1i8eHHYHqcpJkyYwKJFi3C5XPTr16/JtgMHDuTSSy9lzpw5VFZWkpSU1Ozn2d0xxMygxdQGVJ75tjhCyACCms7dX+zl4RP6kN7Ljy0VFBSwYMECpk+fTiAQ4LXXXosQpieeeAKn08kxxxzD0qVLefrpp1m4cCEJCQkkJCQwd+5cLrvsMn777TcmTJiAKIqsW7eOWbNmkZiYyF133cXixYsZP348t99+O3l5eXz//fecffbZpKWlRfTl5JNP5tBDD+XYY4/lpptuYvjw4Xg8HrZs2cJll11GMBjksssuY/r06dhsNp588klGjhzZq4QMDDEzaAXVAY3/FXhj1vlUnV3VgV4vZtXV1WzatIlXXnkFs9nM1KlTefjhh9m4cSMAjz76KC+++CL33HMPaWlp3HvvvVx33XXhx993331kZWXxxBNPsHDhQmw2G6NHj+aUU04hMTGRvn378sUXX3DjjTdyxRVXIAgCgwYN4vjjj48SM0mSWLp0KTfffDO33347xcXFpKSkcMwxx3DZZZdRWVlJYWEhF154IbquM27cOB555JFOfb06A0PMDFpMUNXRm6iv8mmd1peu4qCDDgrvFDakXswGDBjA8uXL4z5eEASuueYarrnmmrhtDjnkED744IOYdY3X5BITE1m0aFHMhNdut5v33nsv7n16C8ZupkGLsZkEkpowVOyXZO7E3hgYhDDEzKDFpNhkLjwktgf/4Vl2ko18mAZdgCFmBi1GFAQOzbJz47gMshJCa2N2WeTMoclcfmhai22wDQzaA+Mn9ACmJqCi6+BshfgkmCWOzElgUIqVgKojiQJJVumACsmIxeTJk6NOAxh0DoaYHYCUehS+L/Lw4ZYqdB2m9HNyWJYDdyssrI0ppUF3wfgkHmCUehXu+6qAX8r94bJfyv2856xk/sSsZifjNTDobhhrZgcYG4q8EUJWz67qIF/vrjWmSAY9FkPMDiA8QZVlv1XFrV++tZqqQO+PETPonRhiZmBg0CswxOwAwm6SOH5AYtz6qf2cJJqNj4RBz8T45B5gDE+zcVCyJao812niqBxHr7KE6Y14PJ4Ib3+DfRhidoCRapO5YVwmVx2exuAUCwclW7h8jJv5E7ONncweQEpKCu+//35Xd6Nb0i1DMx577DE+/fRTFi1aFNPL6e233+b9998nEAgwceJEZs2aZeQQbAGpdplj8hM5PNvR6qDZ9iSoalT4VDQdrLKAy9p576Wq62ws9lHuU0i2ygxNsyJ149Gp3x+9E20QotuNzDZu3Mh3330Xt37p0qUsWbKE6667jttuu401a9bw5ptvdmIPew8JZqnLhazUq/DPH0r504c7ufyDHdz+2V5+KPTgDXb8ruqqXTVc9t52bvtsD39bXcRtn+3hsve2s2pXTYffG+D1119n4MCBOBwOJk6ciKIo+Hw+5s6di9vtxul0cv7551NTU8O2bdvCSwDnn38+giCEXTs2b97MKaecQkJCAsnJycyaNStiKrpixQoOPvhg7HY7Y8aMobi4GIBnn32WkSNHYrfbyc/P55///GenPO+OoluJmaIoPP3000ybNi1mvaZpvP3225x22mkMGjSIgQMHctppp7F06dIoSxSD7k+5V+HeL/fy3q9V+NVQfNuOqgDzP9/L5jJfh9571a4a7l9VSKlXjSgv9arcv6qwwwWtsLCQCy64gGuuuYZNmzZx0003IcsyV199NatWrWL58uV8/fXXbN++nRtvvJG8vDzKy8sBePLJJykvL2f8+PEUFRUxceJErFYrGzZs4JNPPuGbb77h7LPPBkBVVc4880x+//vf88svv7BgwYKwH5rdbuexxx5j06ZNXHTRRVx88cVs3bq1Q593R9KtxGzJkiWYzWbGjh0bs37nzp2Ul5czatSocFm9q+aOHTs6qZcG7UVBTZBfywMx655dV0KFr2N+oFRd59l1JU22eW5dCWoHBhDv3buXQCDAsGHD6NOnDyeddBJ79uzhueee47nnnmPUqFEMHz6ce+65JzxiqneGtdvtJCUlIcsyTzzxBIIg8NJLL5Gfn8+oUaN48cUXWbZsGStXrqSqqoqysjIGDRpETk4Op5xySrgPM2fOZPz48eTl5XHrrbdis9nCeTZ7It1moamoqIi3336b+fPnx91Rq88003Adrf4NLikpoX///jEfN2fOnCbvnZoa287GoGNZXxzbrRZgZ1UQn9IxYrKx2Bc1ImtMiVdlY7GPEem2DunDIYccwvTp05k2bRqzZ8/mlltu4YcffkBRFI4++uhwO03TqKmpobCwMGbauTVr1jB+/HjM5n0ecmPGjMHlcrF27VomTJjAVVddxaWXXsqKFStYsGABAwYMAELfp7/97W98/PHH7N69G4/HQ2lpaYc8386g24zMnnvuOY455pi4ggShxU9BEDCZ9u262WyhD1tvzQXYm0luYqHfLApIHbQOX97MEV9z27UGQRB4++23efXVV/n000857LDDwlmbvvzyS9atW8e6dev44Ycf2Lp1KxkZGTGvo6pqzCNomqYhSaH10Icffphly5axbds2xowZw6+//kpVVRVHHnkka9asYf78+XzyySdkZ2f36ONs3WJktmbNGrZv387cuXObbGc2m9F1Ha/XGxax+g+AxRIdO1VPLCvheubPn9/yDhu0Cwen2xAF0GJ8f6b0c+KydszmRFMi2pp2rUUQBKZPn8748eNxu9389NNPAOzevZvjjz8+5mNkWUZV940qx4wZw3PPPYfP58NqtQKwatUqqqurOfzww8PtjjnmGFasWEFmZibvvPMO+fn57Nixgx9//JGEhAR0Xe/xO6XdYmT24YcfUl5ezuzZszn33HPDojZ37lz+85//hNu53W6AiKFw/f/3l47LoPuRbJO47qgMGlug9XOZOX1IEmapYz6eQ9OspNqaFkq3TWJomrVD7g/w66+/8p///Ic9e/awdOlSAKZNm8b06dO5/PLLWbp0KcXFxWzcuJG1a9eGH9e3b18++ugjCgsLCQaDXHPNNQQCAS644AJ+/vlnvvjiC2bNmsXxxx/PuHHjKC8v54UXXmDnzp189tlnVFdXM3DgwHBS4nfffZfdu3czb948Kisr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zq5du5Akic8//5xLLrmEK664gl9++YWHH364q7vWofz2229s3ryZE044oau70iEcfPDB5OXlsXbtWt5++20yMzNJSkpq0TUMMeslZGZmkp2dzfDhw7n66qsZNGgQS5Ys6eputTterxdZlrn22msZMGAAI0eO5IILLuCHH36grKysq7vXYbz//vuMHDmyV06tFUVh4cKF9O/fn0ceeYTHH38cs9nMwoUL8fv9zb6OIWa9lIMOOqhXfrkTEhKwWCwRC+P1X/Dy8vKu6laHUlFRwVdffcVJJ53U1V3pEDZs2MDu3buZOXMmoiiSmJjI5ZdfTnFxMT/++GOzr2OIWS9l+/bt5ObmdnU32p3BgwdTVVUVsVi8d+9eRFEkKyurC3vWcSxbtozU1NReF0dXj6qqAAQCgXBZfZhNSzKnGWLWC/D5fLz33nvs3LmToqIiXnrpJTZt2sT//d//dXXX2p2+ffsyevRoHnvsMbZv3862bdt46aWXmDJlCna7vau71+4oisKyZcs44YQTek1KxMYMGTKE5OTkcKzZnj17eOqpp8jLy2PEiBHNvo4RNNsLKCkp4R//+Ac7duzA5/MxcOBALrzwQvr169fVXesQPB4PL7zwAqtXr0YQBCZPnsx5553X7PyKPYnPP/+cp59+mieeeKJXxxDu2rWLl156ic2bNyOKIiNHjuS8884jJSWl2dcwxMzAwKBX0DvHrQYGBgcchpgZGBj0CgwxMzAw6BUYYmZgYNArMMTMwMCgV2CImYGBQa/AEDMDA4NegSFmBgYGvQJDzAy6NZ9++imCIPDFF180+zH5+flccsklHdirljF58mSmTp3a1d3o9RhiZmDQjvzvf//j448/7upuHJAYYmZg0I6cfvrpvPrqq13djQMSQ8wMDAx6BYaYGeyXn376iWOPPRan00m/fv144YUXACgrK2P27NmkpKSQlJTEFVdcgc/nCz8uPz+fRYsW8cADD9C3b19sNhvHH388v/76a7jNzp07ueCCC8jLy8NmszFhwgTWr1/frv3XNI177rmHvLw87HY7xx9/PNu3bw/XX3jhhZxzzjm89dZbHHzwwSQkJHDiiSeye/fuiOt8//33TJw4EZvNxvDhw3nvvfc49NBDuf3229m2bRuCILB9+3aeffZZBEHgwgsvjHj8W2+9xfDhw3G5XJx99tlUVVW16/M80Ol9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" />
    


### "Reading" the plot and modifying its parameters


```python
read_plot('tests/output/plot/plot_saved.png',kws_plot=dict(y='petal_length'),title='modified')
```

    INFO:root:shape = (150, 5)





    <AxesSubplot:title={'center':'modified'}, xlabel='sepal_length', ylabel='petal_length'>




    
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASgAAAFCCAYAAABCaEAlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAA9hAAAPYQGoP6dpAABzI0lEQVR4nO2dd3xUVfr/37dMTya9h44iogh2pQp2fyquomLDFV3UteDqrmVVxK64VsS2ll17Wdd1UVlRxIog+kVFxUIv6T2Zeu89vz8mGTLMTAghISGc9+vFS3Puuec+k8x85pznPOd5FCGEQCKRSHogancbIJFIJMmQAiWRSHosUqAkEkmPRQqURCLpsUiBkkgkPRYpUBKJpMciBUoikfRYpEBJJJIeixQoiUTSY5ECJemx3H777SiK0mafBQsWMHz4cFwuF3fccUenPHfjxo0oisJzzz3XKeNJOo4UKMkuw7XXXkv//v3x+XwAbN68mZNOOok999yTDz74gDPPPJNDDz2UyZMnd7Olks5C724DJJL2YlkWNpstOqtauHAhgUCAhx56iKKiIgCEEOi6fFv3FuRfUrLLMHv2bGbPnh39ecOGDaiqGhUngCVLlnSHaZIuQi7xJNtF//79efTRR7npppvIzc0lNzeXxx57jHA4zJ/+9Ceys7MpLi7m6aefjt4TDoe5+eab6du3L06nk5EjR/Lmm2/GjGsYBjfccAN5eXmkpqby+9//nsbGxpg+F154If379wfg/PPP54YbbsCyLBRFibYPHjyY888/P3qPZVnceeed9O3bF7fbzdFHH826detixv3f//7HiBEjcDqdHHjggSxdurTzfmGSHUNIJNtBv379RL9+/cQ555wjPvzwQzF58mQBiGOOOUZMnz5dfPjhh+LEE08UmqaJH374QQghxGmnnSbcbrd44oknxCeffCLOPfdcoSiKePXVV6PjXnTRRcJms4n7779fLFq0SEyfPl04nU7R+i06bdo00a9fPyGEEGvWrBEXXXSR0DRNfPXVV+K7774TQggxaNAgMXXq1Og9f/7zn4XX6xX/+Mc/xAcffCAOPvhgMXToUGGaphBCiI8//lhomiZOPvlk8dFHH4mnnnpKFBQUCEA8++yzXfvLlGwTKVCS7aJfv35i8ODBwjAMIYQQNTU1wmaziX322UdYliWEEKK0tFSoqipuu+02sWjRIgGI559/PjqGZVlizJgxon///kIIIVavXi0URRF33313zLMmTpyYVKCEEGLmzJlC07SYe1oL1MaNG4XD4RAvv/xy9Pr69euFoijigw8+EEIIMW7cODFs2LCoYAkhxPPPPy8Fqocgl3iS7eawww5D0zQA0tPTKSgoYOTIkVHndV5eHnl5eaxfv573338fh8PBGWecEb1fURTOO+881q5dy2+//caHH36IEIKzzjor5jnjxo3bITvnz58PwKmnnhpt69OnD3vssQeLFi0iEAjw+eefc8YZZ6CqWz4KO/pcSechneSS7cZut8f8bLPZYj7gAG63m1AoRHl5OTk5OdhstpjrxcXFAFRXV1NWVgZAYWFhTJ9txUBti9LSUoLBYJy9LdeqqqowDKPTnyvpPKRASbqUzMxMysrKMAwjZvt//fr1AOTm5pKWlgZAZWUleXl50T7BYHCHnu31enG73Xz++edx17Kzs/F6vdHntmZHnyvpPOQST9KlTJw4kXA4zIsvvhhtE0Lw7LPPMnToUPr378+YMWMAeP3112P6vP/++zv07DFjxuDz+fD5fIwYMSLmX3FxMampqYwYMYI33ngD0So1f8vSUNL9yBmUpEs56qijOOqoo7j00kvx+/3svffePPLIIyxdupR33nkHgP3224+TTjqJv/zlL9hsNvbaay/+/ve/U19fv0PPHjFiBJMnT2bSpEnccMMNHHjggYTDYZYvX85VV10FwE033cRpp53Gueeey/Tp0/n+++956KGHorMrSfciZ1CSLkVRFN566y3+8Ic/cMstt3D00UezefNm3n//fY499thov9dee43f//73/OUvf+F3v/sdGRkZ3H333QCEQqEOP/+FF15g+vTpPPDAA4wfP55TTz2V999/n4aGBgB+97vf8cYbb7B48WKOPvpoXn31Vd58801cLtcOPVfSOShCyLJTEomkZyJnUBKJpMciBUoikfRYpEBJJJIeixQoiUTSY5ECJZFIeixSoCQSSY+lVwZqXnPNNRiGET1CIZFIeg51dXXous599923zb69cgZlGAaWZXW3GRKJJAGWZWEYRrv69soZVMvMadasWd1siUQi2ZqZM2e2u2+PEKjy8nIuu+yyhNfmzp1Ldnb2TrZIIpH0BHqEQGVnZ/Pss8/GtP33v/9l6dKlUpwkkt2YHiFQqqri8XiiP5um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A+eddx59+/bF5XIxZswYvv/++06137Is7rzzTvr27Yvb7eboo49m3bp10evnn38+Z511Fm+++Sb77rsvKSkpHHvssWzatClmnG+//ZaxY8ficrkYNmwY77zzDgcccAA333wza9euRVEU1q1bx9NPP42iKJx//vkx97/55psMGzaMtLQ0zjzzTOrr6zv1de7u9L6j4pJ2MWnSJAYPHsy8efNYs2YNRUVFCCE47rjjaGpq4qWXXqKpqYmLL74Yy7J44oknovc+/vjjpKam8vjjj9PU1MSVV17JhAkTWLFiBV6vF7/fT//+/aMf6htvvJFTTjkleqq9M7juuut44okneOSRRygqKuKGG27guOOOY8WKFdFnfPbZZ/z444/cfffdVFVVMWPGDM4991wWLlwIRPJrjR8/nr322ou3336biooKLrvsMurq6rDb7RQWFvLVV19x0kkncfDBB3PjjTfG1K37/vvvmTNnDg899BA//PADf/nLX0hNTeWpp57qlNcoAYRkt6OiokIA4q233oppf/3114XNZhPr1q2Ltr3wwgvC5XKJhoYGIYQQ/fr1ExkZGaKuri7aZ+nSpQIQDz30UMLnLV68WADiiy++EEII8dFHHwlAfPrpp+22uV+/fmLatGlCCCE2btwoHA6HePnll6PX169fLxRFER988IEQQoipU6cKVVXFqlWron1mz54tALFx40YhhBAXXXSRyMjIEPX19dE+8+fPF4C47bbbEj67hXHjxom0tLSY38MFF1wgUlJShGEY7X5dkraRS7zdkKysLPbee2+uvvpq3nrrLURzxp3//Oc/jBkzhr59+0b7TpgwAb/fz5IlS6JtxxxzTEx12IMOOohBgwbx6aefRtsWLlzIKaecQmFhIWPGjAEiS7/OYP78+UDEed1Cnz592GOPPVi0aFG0raioiIEDB0Z/bsle2bIUnD9/PieffHJM5ZjtCR3Yf//9Y34PBxxwAI2NjVRVVW3X65EkRwrUboiiKMyfP5+99tqLU045hZEjR/Lrr79SWlrKwoULURQl+q8lN3hpaWn0/kRJ1vLz86mrqwPgn//8J0ceeSROp5M5c+bw+uuvA7HpX3eE0tJSgsEgdrs9xtZffvklxk673R5zX0tppxY7SkpK4lIob08l462Xqy3ZPzvrdUqkD2q3pU+fPsybN49vvvmGU045halTp1JQUMDYsWMTlnvq379/9P8Nw4i7vnHjRkaPHg3ArFmzOP3006Nb82vWrOlU271eL263m88//zzuWmsfUXvGqaysjGmT4tKzkAK1m7P//vtz+umn89RTT3H66adz11130a9fPzIyMpLes3DhQgKBQHRGsnjxYtatW8ctt9wCRGYmrVMTf/PNN51q85gxY/D5fPh8Pg4//PAOjzNu3Dj+85//8MADD0RfS6KAzJSUFHw+X4efI+k4UqB2Q0pLS7nllls4+eSTCYVCvPLKK4wbN45p06YxZ84cjjjiCP785z8zaNAgampqqK6u5uyzz47eX1VVxYknnsi1115LTU0NM2bMYOjQoUyZMgWAww47jOeff57Ro0dTX1/Prbfe2qn2jxgxgsmTJzNp0iRuuOEGDjzwQMLhMMuXL+eqq65q9zg33XQTBx98MCeccAJ//etfKSkpYfbs2UBkGdzC3nvvzYcffsiCBQsYNGhQjF9L0rVIH9RuSENDQ7RCzLRp0xg1ahRPPvkkqampfPrppwwfPpwrrriC0aNHc+GFF7Js2bKY+08//XRGjhzJlClTuOCCCxg1ahQffPABDocDgKeeeoqBAwcyefJkHnjgAZ5//nlSUlJoaGjotNfwwgsvMH36dB544AHGjx/Pqaeeyvvvv79dzxg5ciTvv/8+paWlHH/88Tz88MM8/fTTALhcrmi/W265hZycHE488UT+85//dNprkGwbWTRBsl3079+fI488kr///e+dMp5lWViWlfS6oijb5bjeUSoqKsjNzeXll1/mzDPP3GnPlSRGzqAk3cqtt96KzWZL+u+AAw7Yqfa88sorqKrKqFGjdupzJYmRPihJt3LxxRczadKkpNdbL7U6m8rKSs477zzOPfdcioqKWLJkCbfccgvnnXceffr06bLnStqPFChJt5Kfn09+fn63PDscDuN0Orn66qupra1lwIABXHPNNdx4443dYo8kHumDkkgkPRbpg5JIJD0WKVASiaTHIgVKIpH0WKRASSSSHosUKIlE0mORAiWRSHosUqAkEkmPRQqURCLpsUiBkkgkPZb/Dwzh+22OY/ncAAAAAElFTkSuQmCC" />
    


#### Documentation
[`roux.viz.io`](https://github.com/rraadd88/roux#module-roux.viz.io)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_viz_line.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for line plots.
<details><summary>Expand</summary>

### Step plot

#### Demo data


```python
import pandas as pd
data=pd.DataFrame({
    'sample size':[100,30,60,50,30,20,25],
    })
data['step name']='step#'+(data.index+1).astype(str)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample size</th>
      <th>step name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
      <td>step#1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>step#2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>step#3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50</td>
      <td>step#4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>step#5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20</td>
      <td>step#6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>25</td>
      <td>step#7</td>
    </tr>
  </tbody>
</table>
</div>



#### Plot


```python
from roux.viz.line import plot_steps
ax=plot_steps(
    data,
    col_step_name='step name',
    col_step_size='sample size',
    )
```


    
<img src="data:image/png;base64,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ljFmzBgiIiJISUkhIiKC2tpa9uzZw7Bhw9i7dy9z584lLi6Ompoa/P39iYmJwd3d3VRvTEwMPXv2ZOjQodTX1zN16lRefvlls7akpKQQGRmJs7Mz+/btIyAgoMVe8Zw5c7h48SJPP/00lZWVDBo0iC+++IKQkBCLxyyEaN+setd7165dpKam8thjj131ps71KisrY82aNURHRzNy5MhWq/dy8fHxVFVVsX379jb7DiHEzcOqU+/hw4djb29Penp6qzYmPT0de3t7hg0b1qr1CiHEjbAqKB0dHenWrRsZGRkUFxe3SkOKi4vJzMykW7du7er1RSGEsHqYNYPBwNKlS9FoNDz88MM39KpefX09a9eupba2loSEBBnxXAjRrlg9YoNarSYuLo6zZ8+yadMm6uvrraqnaTzKqqoq4uLiJCSFEO1OuxjhvKqqynQXWwgh2ptWmTMnLy+PxMREdDodYWFhREVF4efn12L5srIy0tPTyczMlDlzhBDtXqvPwvjjjz/S2NiIt7c3gYGB+Pj4mGZhrKiooKioiDNnzsgsjEIIm9Em83qnpKRw9OhRampq0Ov1V8zr3atXL4YNGyZ3t4UQNqHVg/JSOp2OxYsXAzBv3jwJRiGETfrdnfNmZWWxdetWcnNzqaurIyAggClTpjBixAizcikpKaxfv57S0lICAgJ45JFHuO22236jVgshfku/u6A8cuQIISEhTJs2DU9PTw4cOMCiRYtwdXUlOjraVOa1115j1qxZREdHs3v3bubPn8/atWvp1q3bb3wEQohfm00E5Zw5cwgJCcHR0ZHPPvsMtVrN2LFjiY+Pv+66pk+fbrZ833338d1337Fv3z5TUG7bto2oqCimTJkCwIMPPsjBgwdJSkrimWeeueHjEULYFpsISoCdO3cyadIkVq9eTU5ODkuWLKF///5EREQwd+5cDh8+3OK+fn5+bNy4scXtdXV1ZsOm5eTkMHHiRLMykZGRMgeOEL9TNhOUISEhzJgxA4CAgACSkpLIzMwkIiKChISEq74ZdLXHj1JSUsjNzeXZZ581rausrMTT09OsnFarpbKy8gaPQghhi2wmKLt37262rNVqqaqqArimqXCbc+jQIZYsWUJCQgLBwcE32EIhxM3KZoLy8l6hSqUyTeplzal3dnY2L7zwArNnz75i7EutVnvFBGaVlZVotdobOAIh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" />
    


### Documentation
[`roux.viz.line`](https://github.com/rraadd88/roux#module-roux.viz.line)

</details>

---

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_viz_line.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for scatter plots.
<details><summary>Expand</summary>

### Volcano plot
#### Demo data


```python
import pandas as pd
data = pd.read_csv('https://git.io/volcano_data1.csv')
data['P']=data['P'].replace(data['P'].min(),0) # to show P=0 as a triangle 
data=pd.concat([
    data.query(expr='P>=0.001'),
    data.query(expr='P<0.001').assign(
        **{'categories':lambda df: pd.qcut(df['BP'],3, labels=['low','med','high'])}, # to annotate
        ),
    ],axis=0)
data.head(1)
```

    The history saving thread hit an unexpected error (DatabaseError('database disk image is malformed')).History will not be written to the database.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CHR</th>
      <th>BP</th>
      <th>P</th>
      <th>SNP</th>
      <th>ZSCORE</th>
      <th>EFFECTSIZE</th>
      <th>GENE</th>
      <th>DISTANCE</th>
      <th>categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>937641</td>
      <td>0.335344</td>
      <td>rs9697358</td>
      <td>0.9634</td>
      <td>-0.0946</td>
      <td>ISG15</td>
      <td>1068</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### Plot


```python
from roux.viz.scatter import plot_volcano
ax=plot_volcano(
    data,
    colx='EFFECTSIZE',
    coly='P',
    colindex='SNP',
    show_labels=3, # show top n 
    collabel='SNP',
    text_increase='n',
    text_decrease='n',
    # palette=sns.color_palette()[:3], # increase, decrease, ns
    )
```

    WARNING:root:transforming the coly ("P") values.
    WARNING:root:zeros found, replaced with min 3.41101e-09
    /mnt/d/Documents/code/roux/roux/stat/transform.py:67: RuntimeWarning: divide by zero encountered in log10
      return -1*(np.log10(x))



    
<img src="data:image/png;base64,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" />
    


#### With highlighted points


```python
import seaborn as sns # required to set the palette of the outlines
ax=plot_volcano(
    data=data.query(expr="P<0.05"),
    colx='EFFECTSIZE',
    coly='P',
    colindex='SNP',
    # show_labels=3, # show top n 
    # collabel='SNP',
    show_outlines='categories',
    outline_colors=sns.color_palette()[:3],
    text_increase='n',
    text_decrease='n',
    palette=sns.color_palette('pastel')[:3], # increase, decrease, ns
    legend=True,
    )
```

    WARNING:root:transforming the coly ("P") values.
    WARNING:root:zeros found, replaced with min 3.41101e-09
    /mnt/d/Documents/code/roux/roux/stat/transform.py:67: RuntimeWarning: divide by zero encountered in log10
      return -1*(np.log10(x))



    
<img src="data:image/png;base64,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" />
    


### Documentation
[`roux.viz.scatter`](https://github.com/rraadd88/roux#module-roux.viz.scatter)

</details>

# API
<details><summary>Expand</summary>
<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.global_imports`
For the use in jupyter notebook for example. 



<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.df`
For processing individual pandas DataFrames/Series 


---

## <kbd>function</kbd> `get_name`

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

## <kbd>function</kbd> `get_groupby_columns`

```python
get_groupby_columns(df_)
```

Get the columns supplied to `groupby`. 



**Parameters:**
 
 - <b>`df_`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`columns`</b> (list):  list of columns. 


---

## <kbd>function</kbd> `get_constants`

```python
get_constants(df1)
```

Get the columns with a single unique value. 



**Parameters:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`columns`</b> (list):  list of columns. 


---

## <kbd>function</kbd> `drop_unnamedcol`

```python
drop_unnamedcol(df)
```

Deletes the columns with "Unnamed" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

## <kbd>function</kbd> `drop_unnamedcol`

```python
drop_unnamedcol(df)
```

Deletes the columns with "Unnamed" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

## <kbd>function</kbd> `drop_levelcol`

```python
drop_levelcol(df)
```

Deletes the potentially temporary columns names with "level" prefix. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

## <kbd>function</kbd> `drop_constants`

```python
drop_constants(df)
```

Deletes columns with a single unique value. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

## <kbd>function</kbd> `dropby_patterns`

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

## <kbd>function</kbd> `flatten_columns`

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

## <kbd>function</kbd> `lower_columns`

```python
lower_columns(df)
```

Column names of the dataframe to lower-case letters. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 



**Returns:**
 
 - <b>`df`</b> (DataFrame):  output dataframe. 


---

## <kbd>function</kbd> `renameby_replace`

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

## <kbd>function</kbd> `clean_columns`

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

## <kbd>function</kbd> `clean`

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

## <kbd>function</kbd> `compress`

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

## <kbd>function</kbd> `clean_compress`

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

## <kbd>function</kbd> `check_na`

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

## <kbd>function</kbd> `validate_no_na`

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

## <kbd>function</kbd> `assert_no_na`

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

## <kbd>function</kbd> `check_nunique`

```python
check_nunique(
    df: DataFrame,
    subset: list = None,
    groupby: str = None,
    perc: bool = False
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

## <kbd>function</kbd> `check_inflation`

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

## <kbd>function</kbd> `check_dups`

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

## <kbd>function</kbd> `check_duplicated`

```python
check_duplicated(df, subset=None, perc=False)
```

Check duplicates (alias of `check_dups`)      




---

## <kbd>function</kbd> `validate_no_dups`

```python
validate_no_dups(df, subset=None)
```

Validate that no duplicates. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 


---

## <kbd>function</kbd> `validate_no_duplicates`

```python
validate_no_duplicates(df, subset=None)
```

Validate that no duplicates (alias of `validate_no_dups`)  




---

## <kbd>function</kbd> `assert_no_dups`

```python
assert_no_dups(df, subset=None)
```

Assert that no duplicates  




---

## <kbd>function</kbd> `validate_dense`

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

## <kbd>function</kbd> `assert_dense`

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

## <kbd>function</kbd> `classify_mappings`

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

## <kbd>function</kbd> `check_mappings`

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

## <kbd>function</kbd> `validate_1_1_mappings`

```python
validate_1_1_mappings(df: DataFrame, subset: list = None) → DataFrame
```

Validate that the papping between items in two columns is 1:1. 



**Parameters:**
 
 - <b>`df`</b> (DataFrame):  input dataframe. 
 - <b>`subset`</b> (list):  list of columns. 
 - <b>`out`</b> (str):  format of the output. 




---

## <kbd>function</kbd> `get_mappings`

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

## <kbd>function</kbd> `groupby_filter_fast`

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

## <kbd>function</kbd> `to_map_binary`

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

## <kbd>function</kbd> `check_intersections`

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

## <kbd>function</kbd> `get_totals`

```python
get_totals(ds1)
```

Get totals from the output of `check_intersections`. 



**Parameters:**
 
 - <b>`ds1`</b> (Series):  input Series. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

## <kbd>function</kbd> `filter_rows`

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

## <kbd>function</kbd> `get_bools`

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

## <kbd>function</kbd> `agg_bools`

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

## <kbd>function</kbd> `melt_paired`

```python
melt_paired(
    df: DataFrame,
    cols_index: list = None,
    suffixes: list = None,
    cols_value: list = None
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

## <kbd>function</kbd> `get_chunks`

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

## <kbd>function</kbd> `get_group`

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

## <kbd>function</kbd> `infer_index`

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

## <kbd>function</kbd> `to_multiindex_columns`

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

## <kbd>function</kbd> `to_ranges`

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

## <kbd>function</kbd> `to_boolean`

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

## <kbd>function</kbd> `to_cat`

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

## <kbd>function</kbd> `sort_valuesby_list`

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

## <kbd>function</kbd> `agg_by_order`

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

## <kbd>function</kbd> `agg_by_order_counts`

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

## <kbd>function</kbd> `groupby_sort_values`

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

## <kbd>function</kbd> `groupby_sort_values`

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

## <kbd>function</kbd> `swap_paired_cols`

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

## <kbd>function</kbd> `sort_columns_by_values`

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

## <kbd>function</kbd> `make_ids`

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

## <kbd>function</kbd> `make_ids_sorted`

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

## <kbd>function</kbd> `get_alt_id`

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

## <kbd>function</kbd> `split_ids`

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

## <kbd>function</kbd> `dict2df`

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

## <kbd>function</kbd> `log_shape_change`

```python
log_shape_change(d1, fun='')
```

Report the changes in the shapes of a DataFrame. 



**Parameters:**
 
 - <b>`d1`</b> (dic):  dictionary containing the shapes. 
 - <b>`fun`</b> (str):  name of the function. 


---

## <kbd>function</kbd> `log_apply`

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

## <kbd>class</kbd> `log`
Report (log) the changes in the shapes of the dataframe before and after an operation/s.  



**TODO:**
  Create the attribures (`attr`) using strings e.g. setattr.  import inspect  fun=inspect.currentframe().f_code.co_name 

### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```








---

### <kbd>method</kbd> `check_dups`

```python
check_dups(**kws)
```





---

### <kbd>method</kbd> `check_na`

```python
check_na(**kws)
```





---

### <kbd>method</kbd> `check_nunique`

```python
check_nunique(**kws)
```





---

### <kbd>method</kbd> `clean`

```python
clean(**kws)
```





---

### <kbd>method</kbd> `drop`

```python
drop(**kws)
```





---

### <kbd>method</kbd> `drop_duplicates`

```python
drop_duplicates(**kws)
```





---

### <kbd>method</kbd> `dropna`

```python
dropna(**kws)
```





---

### <kbd>method</kbd> `explode`

```python
explode(**kws)
```





---

### <kbd>method</kbd> `filter_`

```python
filter_(**kws)
```





---

### <kbd>method</kbd> `filter_rows`

```python
filter_rows(**kws)
```





---

### <kbd>method</kbd> `groupby`

```python
groupby(**kws)
```





---

### <kbd>method</kbd> `join`

```python
join(**kws)
```





---

### <kbd>method</kbd> `melt`

```python
melt(**kws)
```





---

### <kbd>method</kbd> `melt_paired`

```python
melt_paired(**kws)
```





---

### <kbd>method</kbd> `merge`

```python
merge(**kws)
```





---

### <kbd>method</kbd> `pivot`

```python
pivot(**kws)
```





---

### <kbd>method</kbd> `pivot_table`

```python
pivot_table(**kws)
```





---

### <kbd>method</kbd> `query`

```python
query(**kws)
```





---

### <kbd>method</kbd> `stack`

```python
stack(**kws)
```





---

### <kbd>method</kbd> `unstack`

```python
unstack(**kws)
```






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.dfs`
For processing multiple pandas DataFrames/Series 


---

## <kbd>function</kbd> `filter_dfs`

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

## <kbd>function</kbd> `merge_with_many_columns`

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

## <kbd>function</kbd> `merge_paired`

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

## <kbd>function</kbd> `merge_dfs`

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

## <kbd>function</kbd> `compare_rows`

```python
compare_rows(df1, df2, test=False, **kws)
```






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.dict`
For processing dictionaries. 


---

## <kbd>function</kbd> `head_dict`

```python
head_dict(d, lines=5)
```






---

## <kbd>function</kbd> `sort_dict`

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

## <kbd>function</kbd> `merge_dicts`

```python
merge_dicts(l: list) → dict
```

Merge dictionaries. 



**Parameters:**
 
 - <b>`l`</b> (list):  list containing the dictionaries. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 

TODOs: in python>=3.9, `merged = d1 | d2` 


---

## <kbd>function</kbd> `merge_dict_values`

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

## <kbd>function</kbd> `flip_dict`

```python
flip_dict(d)
```

switch values with keys and vice versa. 



**Parameters:**
 
 - <b>`d`</b> (dict):  input dictionary. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.google`
Processing files form google-cloud services. 


---

## <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object.  

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

## <kbd>function</kbd> `get_service`

```python
get_service(service_name='drive', access_limit=True, client_config=None)
```

Creates a google service object.  

:param service_name: name of the service e.g. drive :param access_limit: True is access limited else False :param client_config: custom client config ... :return: google service object 

Ref: https://developers.google.com/drive/api/v3/about-auth 


---

## <kbd>function</kbd> `list_files_in_folder`

```python
list_files_in_folder(service, folderid, filetype=None, fileext=None, test=False)
```

Lists files in a google drive folder. 

:param service: service object e.g. drive :param folderid: folder id from google drive :param filetype: specify file type :param fileext: specify file extension :param test: True if verbose else False ... :return: list of files in the folder      


---

## <kbd>function</kbd> `get_file_id`

```python
get_file_id(p)
```






---

## <kbd>function</kbd> `download_file`

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

## <kbd>function</kbd> `upload_file`

```python
upload_file(service, filep, folder_id, test=False)
```

Uploads a local file onto google drive. 

:param service: google service object :param filep: path of the file :param folder_id: id of the folder on google drive where the file will be uploaded  :param test: True is verbose else False ... :return: id of the uploaded file  


---

## <kbd>function</kbd> `upload_files`

```python
upload_files(service, ps, folder_id, **kws)
```






---

## <kbd>function</kbd> `download_drawings`

```python
download_drawings(folderid, outd, service=None, test=False)
```

Download specific files: drawings 

TODOs: 1. use download_file 


---

## <kbd>function</kbd> `get_comments`

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

## <kbd>function</kbd> `search`

```python
search(query, results=1, service=None, **kws_search)
```

Google search. 

:param query: exact terms ... :return: dict 


---

## <kbd>function</kbd> `get_search_strings`

```python
get_search_strings(text, num=5, test=False)
```

Google search. 

:param text: string :param num: number of results :param test: True if verbose else False ... :return lines: list 


---

## <kbd>function</kbd> `get_metadata_of_paper`

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

## <kbd>function</kbd> `share`

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

## <kbd>class</kbd> `slides`







---

### <kbd>method</kbd> `create_image`

```python
create_image(service, presentation_id, page_id, image_id)
```

image less than 1.5 Mb 

---

### <kbd>method</kbd> `get_page_ids`

```python
get_page_ids(service, presentation_id)
```






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.io`
For input/output of data files. 


---

## <kbd>function</kbd> `to_zip`

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

## <kbd>function</kbd> `read_zip`

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

 fun_read=lambda x: pd.read_csv(io.StringIO(x.decode('utf-8')),sep='        ',header=None), 


---

## <kbd>function</kbd> `get_version`

```python
get_version(suffix: str = '') → str
```

Get the time-based version string. 



**Parameters:**
 
 - <b>`suffix`</b> (string):  suffix. 



**Returns:**
 
 - <b>`version`</b> (string):  version. 


---

## <kbd>function</kbd> `version`

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

## <kbd>function</kbd> `backup`

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

## <kbd>function</kbd> `read_url`

```python
read_url(url)
```

Read text from an URL. 



**Parameters:**
 
 - <b>`url`</b> (str):  URL link. 



**Returns:**
 
 - <b>`s`</b> (string):  text content of the URL. 


---

## <kbd>function</kbd> `download`

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

## <kbd>function</kbd> `read_text`

```python
read_text(p)
```

Read a file.  To be called by other functions   



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  contents. 


---

## <kbd>function</kbd> `to_list`

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

## <kbd>function</kbd> `read_list`

```python
read_list(p)
```

Read the lines in the file. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `read_list`

```python
read_list(p)
```

Read the lines in the file. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `is_dict`

```python
is_dict(p)
```






---

## <kbd>function</kbd> `read_dict`

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

## <kbd>function</kbd> `to_dict`

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

## <kbd>function</kbd> `post_read_table`

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

## <kbd>function</kbd> `read_table`

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

## <kbd>function</kbd> `get_logp`

```python
get_logp(ps: list) → str
```

Infer the path of the log file. 



**Parameters:**
 
 - <b>`ps`</b> (list):  list of paths.      



**Returns:**
 
 - <b>`p`</b> (str):  path of the output file.      


---

## <kbd>function</kbd> `apply_on_paths`

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

## <kbd>function</kbd> `read_tables`

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

## <kbd>function</kbd> `to_table`

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

## <kbd>function</kbd> `to_manytables`

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

## <kbd>function</kbd> `to_table_pqt`

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

## <kbd>function</kbd> `tsv2pqt`

```python
tsv2pqt(p: str) → str
```

Convert tab-separated file to Apache parquet.  



**Parameters:**
 
 - <b>`p`</b> (str):  path of the input. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output. 


---

## <kbd>function</kbd> `pqt2tsv`

```python
pqt2tsv(p: str) → str
```

Convert Apache parquet file to tab-separated.  



**Parameters:**
 
 - <b>`p`</b> (str):  path of the input. 



**Returns:**
 
 - <b>`p`</b> (str):  path of the output. 


---

## <kbd>function</kbd> `read_excel`

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

## <kbd>function</kbd> `to_excel_commented`

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

## <kbd>function</kbd> `to_excel`

```python
to_excel(
    sheetname2df: dict,
    outp: str,
    comments: dict = None,
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

Keyword parameters:  
 - <b>`kws`</b>:  parameters provided to the excel writer. 


---

## <kbd>function</kbd> `check_chunks`

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

# <kbd>module</kbd> `roux.lib`




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

## <kbd>function</kbd> `to_class`

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

## <kbd>function</kbd> `decorator`

```python
decorator(func)
```






---

## <kbd>class</kbd> `rd`
`roux-dataframe` (`.rd`) extension.  



### <kbd>method</kbd> `__init__`

```python
__init__(pandas_obj)
```









<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.lib.seq`
For processing biological sequence data. 

**Global Variables**
---------------
- **bed_colns**

---

## <kbd>function</kbd> `reverse_complement`

```python
reverse_complement(s)
```

Reverse complement. 



**Args:**
 
 - <b>`s`</b> (str):  sequence 



**Returns:**
 
 - <b>`s`</b> (str):  reverse complemented sequence 


---

## <kbd>function</kbd> `fa2df`

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

## <kbd>function</kbd> `to_genomeocoords`

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

## <kbd>function</kbd> `to_bed`

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

## <kbd>function</kbd> `read_fasta`

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

## <kbd>function</kbd> `to_fasta`

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

# <kbd>module</kbd> `roux.lib.set`
For processing list-like sets. 


---

## <kbd>function</kbd> `union`

```python
union(l)
```

Union of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `union`

```python
union(l)
```

Union of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `intersection`

```python
intersection(l)
```

Intersections of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `intersection`

```python
intersection(l)
```

Intersections of lists. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `nunion`

```python
nunion(l)
```

Count the items in union. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`i`</b> (int):  count.     


---

## <kbd>function</kbd> `nintersection`

```python
nintersection(l)
```

Count the items in intersetion. 



**Parameters:**
 
 - <b>`l`</b> (list):  list of lists. 



**Returns:**
 
 - <b>`i`</b> (int):  count.     


---

## <kbd>function</kbd> `dropna`

```python
dropna(x)
```

Drop `np.nan` items from a list. 



**Parameters:**
 
 - <b>`x`</b> (list):  list. 



**Returns:**
 
 - <b>`x`</b> (list):  list. 


---

## <kbd>function</kbd> `unique`

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

## <kbd>function</kbd> `list2str`

```python
list2str(x, ignore=False)
```

Returns string if single item in a list. 



**Parameters:**
 
 - <b>`x`</b> (list):  list 



**Returns:**
 
 - <b>`s`</b> (str):  string.         


---

## <kbd>function</kbd> `unique_str`

```python
unique_str(l, **kws)
```

Unique single item from a list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  list. 


---

## <kbd>function</kbd> `nunique`

```python
nunique(l, **kws)
```

Count unique items in a list 



**Parameters:**
 
 - <b>`l`</b> (list):  list 



**Returns:**
 
 - <b>`i`</b> (int):  count. 


---

## <kbd>function</kbd> `flatten`

```python
flatten(l)
```

List of lists to list. 



**Parameters:**
 
 - <b>`l`</b> (list):  input list. 



**Returns:**
 
 - <b>`l`</b> (list):  output list. 


---

## <kbd>function</kbd> `get_alt`

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

## <kbd>function</kbd> `jaccard_index`

```python
jaccard_index(l1, l2)
```






---

## <kbd>function</kbd> `intersections`

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

## <kbd>function</kbd> `range_overlap`

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

## <kbd>function</kbd> `get_windows`

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

## <kbd>function</kbd> `bools2intervals`

```python
bools2intervals(v)
```

Convert bools to intervals. 



**Parameters:**
 
 - <b>`v`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`l`</b> (list):  intervals. 


---

## <kbd>function</kbd> `list2ranges`

```python
list2ranges(l)
```






---

## <kbd>function</kbd> `get_pairs`

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

# <kbd>module</kbd> `roux.lib.str`
For processing strings. 


---

## <kbd>function</kbd> `substitution`

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

## <kbd>function</kbd> `substitution`

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

## <kbd>function</kbd> `replace_many`

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

## <kbd>function</kbd> `replace_many`

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

## <kbd>function</kbd> `tuple2str`

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

## <kbd>function</kbd> `linebreaker`

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

## <kbd>function</kbd> `findall`

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

## <kbd>function</kbd> `get_marked_substrings`

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

## <kbd>function</kbd> `get_marked_substrings`

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

## <kbd>function</kbd> `mark_substrings`

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

## <kbd>function</kbd> `get_bracket`

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

## <kbd>function</kbd> `align`

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

## <kbd>function</kbd> `get_prefix`

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

## <kbd>function</kbd> `get_suffix`

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

## <kbd>function</kbd> `get_fix`

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

## <kbd>function</kbd> `removesuffix`

```python
removesuffix(s1: str, suffix: str) → str
```

Remove suffix. 

Paramters:  s1 (str): input string.  suffix (str): suffix.  



**Returns:**
 
 - <b>`s1`</b> (str):  string without the suffix. 

TODOs:  1. Deprecate in py>39 use .removesuffix() instead. 


---

## <kbd>function</kbd> `str2dict`

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

## <kbd>function</kbd> `dict2str`

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

## <kbd>function</kbd> `str2num`

```python
str2num(s: str) → float
```

String to number. 



**Parameters:**
 
 - <b>`s`</b> (str):  string. 



**Returns:**
 
 - <b>`i`</b> (int):  number. 


---

## <kbd>function</kbd> `num2str`

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

## <kbd>function</kbd> `encode`

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

## <kbd>function</kbd> `decode`

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

# <kbd>module</kbd> `roux.lib.sys`
For processing file paths for example. 


---

## <kbd>function</kbd> `basenamenoext`

```python
basenamenoext(p)
```

Basename without the extension. 



**Args:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`s`</b> (str):  output. 


---

## <kbd>function</kbd> `remove_exts`

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

## <kbd>function</kbd> `read_ps`

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

## <kbd>function</kbd> `to_path`

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

## <kbd>function</kbd> `to_path`

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

## <kbd>function</kbd> `makedirs`

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

## <kbd>function</kbd> `to_output_path`

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

## <kbd>function</kbd> `to_output_paths`

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

## <kbd>function</kbd> `get_encoding`

```python
get_encoding(p)
```

Get encoding of a file. 



**Parameters:**
 
 - <b>`p`</b> (str):  file path 



**Returns:**
 
 - <b>`s`</b> (string):  encoding. 


---

## <kbd>function</kbd> `get_all_subpaths`

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

## <kbd>function</kbd> `get_env`

```python
get_env(env_name: str, return_path: bool = False)
```

Get the virtual environment as a dictionary. 



**Args:**
 
 - <b>`env_name`</b> (str):  name of the environment. 



**Returns:**
 
 - <b>`d`</b> (dict):  parameters of the virtual environment. 


---

## <kbd>function</kbd> `runbash`

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

## <kbd>function</kbd> `runbash_tmp`

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

## <kbd>function</kbd> `create_symlink`

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

## <kbd>function</kbd> `input_binary`

```python
input_binary(q: str)
```

Get input in binary format. 



**Args:**
 
 - <b>`q`</b> (str):  question. 



**Returns:**
 
 - <b>`b`</b> (bool):  response. 


---

## <kbd>function</kbd> `is_interactive`

```python
is_interactive()
```

Check if the UI is interactive e.g. jupyter or command line.   




---

## <kbd>function</kbd> `is_interactive_notebook`

```python
is_interactive_notebook()
```

Check if the UI is interactive e.g. jupyter or command line.      



**Notes:**

> 
>Reference: 


---

## <kbd>function</kbd> `get_excecution_location`

```python
get_excecution_location(depth=1)
```

Get the location of the function being executed. 



**Args:**
 
 - <b>`depth`</b> (int, optional):  Depth of the location. Defaults to 1. 



**Returns:**
 
 - <b>`tuple`</b> (tuple):  filename and line number. 


---

## <kbd>function</kbd> `get_datetime`

```python
get_datetime(outstr=True)
```

Get the date and time. 



**Args:**
 
 - <b>`outstr`</b> (bool, optional):  string output. Defaults to True. 



**Returns:**
 
 - <b>`s `</b>:  date and time. 


---

## <kbd>function</kbd> `p2time`

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

## <kbd>function</kbd> `ps2time`

```python
ps2time(ps: list, **kws_p2time)
```

Get the times for a list of files.  



**Args:**
 
 - <b>`ps`</b> (list):  list of paths. 



**Returns:**
 
 - <b>`ds`</b> (Series):  paths mapped to corresponding times. 


---

## <kbd>function</kbd> `get_logger`

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

# <kbd>module</kbd> `roux.lib.text`
For processing text files. 


---

## <kbd>function</kbd> `get_header`

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

## <kbd>function</kbd> `cat`

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

# <kbd>module</kbd> `roux.run`
For access to a few functions from the terminal. 



<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.stat.binary`
For processing binary data. 


---

## <kbd>function</kbd> `compare_bools_jaccard`

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

## <kbd>function</kbd> `compare_bools_jaccard_df`

```python
compare_bools_jaccard_df(df: DataFrame) → DataFrame
```

Pairwise compare bools in terms of the jaccard index. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  dataframe with boolean columns. 



**Returns:**
 
 - <b>`DataFrame`</b>:  matrix with comparisons between the columns. 


---

## <kbd>function</kbd> `classify_bools`

```python
classify_bools(l: list) → str
```

Classify bools. 



**Args:**
 
 - <b>`l`</b> (list):  list of bools 



**Returns:**
 
 - <b>`str`</b>:  classification. 


---

## <kbd>function</kbd> `frac`

```python
frac(x: list) → float
```

Fraction. 



**Args:**
 
 - <b>`x`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`float`</b>:  fraction of True values. 


---

## <kbd>function</kbd> `perc`

```python
perc(x: list) → float
```

Percentage. 



**Args:**
 
 - <b>`x`</b> (list):  list of bools. 



**Returns:**
 
 - <b>`float`</b>:  Percentage of the True values 


---

## <kbd>function</kbd> `get_stats_confusion_matrix`

```python
get_stats_confusion_matrix(df_: DataFrame) → DataFrame
```

Get stats confusion matrix. 



**Args:**
 
 - <b>`df_`</b> (DataFrame):  Confusion matrix. 



**Returns:**
 
 - <b>`DataFrame`</b>:  stats. 


---

## <kbd>function</kbd> `get_cutoff`

```python
get_cutoff(
    y_true,
    y_score,
    method,
    show_diagonal=True,
    show_area=True,
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

# <kbd>module</kbd> `roux.stat.classify`
For classification. 


---

## <kbd>function</kbd> `drop_low_complexity`

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

## <kbd>function</kbd> `get_Xy_for_classification`

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

## <kbd>function</kbd> `get_cvsplits`

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

## <kbd>function</kbd> `get_grid_search`

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

## <kbd>function</kbd> `get_estimatorn2grid_search`

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

## <kbd>function</kbd> `get_test_scores`

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

## <kbd>function</kbd> `plot_metrics`

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

## <kbd>function</kbd> `get_probability`

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

## <kbd>function</kbd> `run_grid_search`

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

## <kbd>function</kbd> `plot_feature_predictive_power`

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

## <kbd>function</kbd> `get_feature_predictive_power`

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

## <kbd>function</kbd> `get_feature_importances`

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

## <kbd>function</kbd> `get_partial_dependence`

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

# <kbd>module</kbd> `roux.stat.cluster`
For clustering data. 


---

## <kbd>function</kbd> `check_clusters`

```python
check_clusters(df: DataFrame)
```

Check clusters. 



**Args:**
 
 - <b>`df`</b> (DataFrame):  dataframe. 


---

## <kbd>function</kbd> `get_clusters`

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

## <kbd>function</kbd> `get_n_clusters_optimum`

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

## <kbd>function</kbd> `plot_silhouette`

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

## <kbd>function</kbd> `get_clusters_optimum`

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

## <kbd>function</kbd> `get_gmm_params`

```python
get_gmm_params(g, x, n_clusters=2, test=False)
```

Intersection point of the two peak Gaussian mixture Models (GMMs). 



**Args:**
 
 - <b>`out`</b> (str):  `coff` only or `params` for all the parameters. 




---

## <kbd>function</kbd> `get_gmm_intersection`

```python
get_gmm_intersection(x, two_pdfs, means, weights, test=False)
```






---

## <kbd>function</kbd> `cluster_1d`

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

## <kbd>function</kbd> `get_pos_umap`

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

# <kbd>module</kbd> `roux.stat.compare`
For comparison related stats. 


---

## <kbd>function</kbd> `get_cols_x_for_comparison`

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

## <kbd>function</kbd> `to_filteredby_samples`

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

## <kbd>function</kbd> `to_preprocessed_data`

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

## <kbd>function</kbd> `get_comparison`

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

## <kbd>function</kbd> `compare_strings`

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

# <kbd>module</kbd> `roux.stat.corr`
For correlation stats. 


---

## <kbd>function</kbd> `get_spearmanr`

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

## <kbd>function</kbd> `get_pearsonr`

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

## <kbd>function</kbd> `get_corr_resampled`

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

## <kbd>function</kbd> `corr_to_str`

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

## <kbd>function</kbd> `get_corr`

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

## <kbd>function</kbd> `get_corrs`

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

## <kbd>function</kbd> `get_partial_corrs`

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

## <kbd>function</kbd> `check_collinearity`

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

## <kbd>function</kbd> `pairwise_chi2`

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

# <kbd>module</kbd> `roux.stat.diff`
For difference related stats. 


---

## <kbd>function</kbd> `get_demo_data`

```python
get_demo_data() → DataFrame
```

Demo data to test the differences. 


---

## <kbd>function</kbd> `compare_classes`

```python
compare_classes(x, y, method=None)
```

 




---

## <kbd>function</kbd> `compare_classes_many`

```python
compare_classes_many(df1: DataFrame, cols_y: list, cols_x: list) → DataFrame
```






---

## <kbd>function</kbd> `get_pval`

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

## <kbd>function</kbd> `get_stat`

```python
get_stat(
    df1: DataFrame,
    colsubset: str,
    colvalue: str,
    colindex: str,
    subsets=None,
    cols_subsets=['subset1', 'subset2'],
    df2=None,
    stats=[<function mean at 0x7f62f8090d40>, <function median at 0x7f62e8eb8d40>, <function var at 0x7f62f8093200>, <built-in function len>],
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

## <kbd>function</kbd> `get_stats`

```python
get_stats(
    df1: DataFrame,
    colsubset: str,
    cols_value: list,
    colindex: str,
    subsets=None,
    df2=None,
    cols_subsets=['subset1', 'subset2'],
    stats=[<function mean at 0x7f62f8090d40>, <function median at 0x7f62e8eb8d40>, <function var at 0x7f62f8093200>, <built-in function len>],
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

## <kbd>function</kbd> `get_significant_changes`

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

## <kbd>function</kbd> `apply_get_significant_changes`

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

## <kbd>function</kbd> `get_stats_groupby`

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

## <kbd>function</kbd> `get_diff`

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

## <kbd>function</kbd> `binby_pvalue_coffs`

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

# <kbd>module</kbd> `roux.stat.enrich`
For enrichment related stats. 


---

## <kbd>function</kbd> `get_enrichment`

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

## <kbd>function</kbd> `get_enrichments`

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

# <kbd>module</kbd> `roux.stat.fit`
For fitting data. 


---

## <kbd>function</kbd> `fit_curve_fit`

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

## <kbd>function</kbd> `fit_gauss_bimodal`

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

## <kbd>function</kbd> `get_grid`

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

## <kbd>function</kbd> `fit_gaussian2d`

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

## <kbd>function</kbd> `fit_2d_distribution_kde`

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

## <kbd>function</kbd> `check_poly_fit`

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

## <kbd>function</kbd> `mlr_2`

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

## <kbd>function</kbd> `get_mlr_2_str`

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

# <kbd>module</kbd> `roux.stat.io`
For input/output of stats. 


---

## <kbd>function</kbd> `perc_label`

```python
perc_label(a, b=None, bracket=True)
```






---

## <kbd>function</kbd> `pval2annot`

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

# <kbd>module</kbd> `roux.stat`




**Global Variables**
---------------
- **binary**
- **io**


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.stat.network`
For network related stats. 


---

## <kbd>function</kbd> `get_subgraphs`

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

# <kbd>module</kbd> `roux.stat.norm`
For normalisation. 


---

## <kbd>function</kbd> `norm_by_quantile`

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

## <kbd>function</kbd> `norm_by_gaussian_kde`

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

## <kbd>function</kbd> `zscore`

```python
zscore(df: DataFrame, cols: list = None) → DataFrame
```

Z-score. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input table. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output table. 


---

## <kbd>function</kbd> `zscore_robust`

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

# <kbd>module</kbd> `roux.stat.paired`
For paired stats. 


---

## <kbd>function</kbd> `get_ratio_sorted`

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

## <kbd>function</kbd> `diff`

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

## <kbd>function</kbd> `get_diff_sorted`

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

## <kbd>function</kbd> `balance`

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

## <kbd>function</kbd> `get_paired_sets_stats`

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

## <kbd>function</kbd> `get_stats_paired`

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

## <kbd>function</kbd> `get_stats_paired_agg`

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

## <kbd>function</kbd> `classify_sharing`

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

# <kbd>module</kbd> `roux.stat.regress`
For regression. 


---

## <kbd>function</kbd> `to_columns_renamed_for_regression`

```python
to_columns_renamed_for_regression(df1: DataFrame, columns: dict) → DataFrame
```

[UNDER DEVELOPMENT] 


---

## <kbd>function</kbd> `check_covariates`

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

## <kbd>function</kbd> `to_input_data_for_regression`

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

## <kbd>function</kbd> `to_formulas`

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

## <kbd>function</kbd> `get_stats_regression`

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

## <kbd>function</kbd> `to_filteredby_variable`

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

## <kbd>function</kbd> `run_lr_test`

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

## <kbd>function</kbd> `plot_residuals_versus_fitted`

```python
plot_residuals_versus_fitted(model: object) → Axes
```

plot Residuals Versus Fitted (RVF). 



**Args:**
 
 - <b>`model`</b> (object):  model. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 


---

## <kbd>function</kbd> `plot_residuals_versus_groups`

```python
plot_residuals_versus_groups(model: object) → Axes
```

plot Residuals Versus groups. 



**Args:**
 
 - <b>`model`</b> (object):  model. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  output. 


---

## <kbd>function</kbd> `plot_model_qcs`

```python
plot_model_qcs(model: object)
```

Plot Quality Checks. 



**Args:**
 
 - <b>`model`</b> (object):  model. 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.stat.set`
For set related stats. 


---

## <kbd>function</kbd> `get_intersection_stats`

```python
get_intersection_stats(df, coltest, colset, background_size=None)
```






---

## <kbd>function</kbd> `get_set_enrichment_stats`

```python
get_set_enrichment_stats(test, sets, background, fdr_correct=True)
```

test:  get_set_enrichment_stats(background=range(120),  test=range(100),  sets={f"set {i}":list(np.unique(np.random.randint(low=100,size=i+1))) for i in range(100)})  # background is int  get_set_enrichment_stats(background=110,  test=unique(range(100)),  sets={f"set {i}":unique(np.random.randint(low=140,size=i+1)) for i in range(0,140,10)})                         


---

## <kbd>function</kbd> `test_set_enrichment`

```python
test_set_enrichment(tests_set2elements, test2_set2elements, background_size)
```






---

## <kbd>function</kbd> `get_paired_sets_stats`

```python
get_paired_sets_stats(l1, l2)
```

overlap, intersection, union, ratio 


---

## <kbd>function</kbd> `get_enrichment`

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

## <kbd>function</kbd> `get_enrichments`

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

# <kbd>module</kbd> `roux.stat.solve`
For solving equations. 


---

## <kbd>function</kbd> `get_intersection_locations`

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

# <kbd>module</kbd> `roux.stat.transform`
For transformations. 


---

## <kbd>function</kbd> `plog`

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

## <kbd>function</kbd> `anti_plog`

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

## <kbd>function</kbd> `log_pval`

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

## <kbd>function</kbd> `get_q`

```python
get_q(ds1: Series, col: str = None, verb: bool = True, test_coff: float = 0.1)
```

To FDR corrected P-value. 


---

## <kbd>function</kbd> `glog`

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

## <kbd>function</kbd> `rescale`

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

## <kbd>function</kbd> `rescale_divergent`

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

# <kbd>module</kbd> `roux.stat.variance`
For variance related stats. 


---

## <kbd>function</kbd> `confidence_interval_95`

```python
confidence_interval_95(x: <built-in function array>) → float
```

95% confidence interval. 



**Args:**
 
 - <b>`x`</b> (np.array):  input vector. 



**Returns:**
 
 - <b>`float`</b>:  output. 


---

## <kbd>function</kbd> `get_ci`

```python
get_ci(rs, ci_type, outstr=False)
```






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.viz.annot`
For annotations. 


---

## <kbd>function</kbd> `set_label`

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

## <kbd>function</kbd> `annot_side`

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

## <kbd>function</kbd> `annot_corners`

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

## <kbd>function</kbd> `confidence_ellipse`

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

## <kbd>function</kbd> `show_box`

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

## <kbd>function</kbd> `annot_confusion_matrix`

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

## <kbd>function</kbd> `get_logo_ax`

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

## <kbd>function</kbd> `set_logo`

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

## <kbd>function</kbd> `color_ax`

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

## <kbd>function</kbd> `annot_n_legend`

```python
annot_n_legend(ax, df1: DataFrame, colid: str, colgroup: str, **kws)
```






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.viz.ax_`
For setting up subplots. 


---

## <kbd>function</kbd> `set_`

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

## <kbd>function</kbd> `set_axes_minimal`

```python
set_axes_minimal(ax, xlabel=None, ylabel=None, off_axes_pad=0) → Axes
```

Set minimal axes labels, at the lower left corner. 


---

## <kbd>function</kbd> `set_ylabel`

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

## <kbd>function</kbd> `rename_labels`

```python
rename_labels(ax, d1)
```






---

## <kbd>function</kbd> `format_labels`

```python
format_labels(ax, fmt='cap1', title_fontsize=15, test=False)
```






---

## <kbd>function</kbd> `rename_ticklabels`

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

## <kbd>function</kbd> `get_ticklabel_position`

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

## <kbd>function</kbd> `get_ticklabel_position`

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

## <kbd>function</kbd> `set_ticklabels_color`

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

## <kbd>function</kbd> `set_ticklabels_color`

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

## <kbd>function</kbd> `format_ticklabels`

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

## <kbd>function</kbd> `set_equallim`

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

## <kbd>function</kbd> `get_axlims`

```python
get_axlims(ax: Axes) → Axes
```

Get axis limits. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `set_axlims`

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

## <kbd>function</kbd> `get_axlimsby_data`

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

## <kbd>function</kbd> `split_ticklabels`

```python
split_ticklabels(
    ax: Axes,
    axis='x',
    grouped=False,
    group_x=0.01,
    group_prefix=None,
    group_loc='left',
    group_colors=None,
    group_alpha=0.2,
    show_group_line=True,
    show_group_span=True,
    sep: str = '-',
    pad_major=6,
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

## <kbd>function</kbd> `set_grids`

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

## <kbd>function</kbd> `rename_legends`

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

## <kbd>function</kbd> `append_legends`

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

## <kbd>function</kbd> `sort_legends`

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

## <kbd>function</kbd> `drop_duplicate_legend`

```python
drop_duplicate_legend(ax, **kws)
```






---

## <kbd>function</kbd> `reset_legend_colors`

```python
reset_legend_colors(ax)
```

Reset legend colors. 



**Args:**
 
 - <b>`ax`</b> (plt.Axes):  `plt.Axes` object. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `set_legends_merged`

```python
set_legends_merged(axs)
```

Reset legend colors. 



**Args:**
 
 - <b>`axs`</b> (list):  list of `plt.Axes` objects. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  first `plt.Axes` object in the list. 


---

## <kbd>function</kbd> `set_legend_custom`

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

## <kbd>function</kbd> `get_line_cap_length`

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

## <kbd>function</kbd> `get_subplot_dimentions`

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

## <kbd>function</kbd> `set_colorbar`

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

## <kbd>function</kbd> `set_colorbar_label`

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

# <kbd>module</kbd> `roux.viz.bar`
For bar plots. 


---

## <kbd>function</kbd> `plot_barh`

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

## <kbd>function</kbd> `plot_value_counts`

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

## <kbd>function</kbd> `plot_barh_stacked_percentage`

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

## <kbd>function</kbd> `plot_bar_serial`

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

## <kbd>function</kbd> `plot_barh_stacked_percentage_intersections`

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

## <kbd>function</kbd> `to_input_data_sankey`

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

## <kbd>function</kbd> `plot_sankey`

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

# <kbd>module</kbd> `roux.viz.colors`
For setting up colors. 


---

## <kbd>function</kbd> `rgbfloat2int`

```python
rgbfloat2int(rgb_float)
```






---

## <kbd>function</kbd> `get_colors_default`

```python
get_colors_default() → list
```

get default colors. 



**Returns:**
 
 - <b>`list`</b>:  colors. 


---

## <kbd>function</kbd> `get_ncolors`

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

## <kbd>function</kbd> `get_val2color`

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

## <kbd>function</kbd> `saturate_color`

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

## <kbd>function</kbd> `mix_colors`

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

## <kbd>function</kbd> `make_cmap`

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

## <kbd>function</kbd> `get_cmap_section`

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

## <kbd>function</kbd> `append_cmap`

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

# <kbd>module</kbd> `roux.viz.compare`
For comparative plots. 


---

## <kbd>function</kbd> `plot_comparisons`

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

# <kbd>module</kbd> `roux.viz.dist`
For distribution plots. 


---

## <kbd>function</kbd> `hist_annot`

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

## <kbd>function</kbd> `plot_gmm`

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

## <kbd>function</kbd> `plot_normal`

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

## <kbd>function</kbd> `plot_dists`

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

## <kbd>function</kbd> `pointplot_groupbyedgecolor`

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

# <kbd>module</kbd> `roux.viz.figure`
For setting up figures. 


---

## <kbd>function</kbd> `get_subplots`

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

## <kbd>function</kbd> `labelplots`

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

# <kbd>module</kbd> `roux.viz.heatmap`
For heatmaps. 


---

## <kbd>function</kbd> `plot_table`

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

## <kbd>function</kbd> `plot_crosstab`

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

# <kbd>module</kbd> `roux.viz.image`
For visualization of images. 


---

## <kbd>function</kbd> `plot_image`

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

# <kbd>module</kbd> `roux.viz.io`
For input/output of plots. 


---

## <kbd>function</kbd> `to_plotp`

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

## <kbd>function</kbd> `savefig`

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

## <kbd>function</kbd> `savelegend`

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

## <kbd>function</kbd> `update_kws_plot`

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

## <kbd>function</kbd> `get_plot_inputs`

```python
get_plot_inputs(plotp: str, df1: DataFrame, kws_plot: dict, outd: str) → tuple
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

## <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code.  




---

## <kbd>function</kbd> `log_code`

```python
log_code()
```

Log the code.  




---

## <kbd>function</kbd> `get_lines`

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

## <kbd>function</kbd> `to_script`

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

## <kbd>function</kbd> `to_plot`

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

## <kbd>function</kbd> `read_plot`

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

## <kbd>function</kbd> `to_concat`

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

## <kbd>function</kbd> `to_montage`

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

## <kbd>function</kbd> `to_gif`

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

## <kbd>function</kbd> `to_data`

```python
to_data(path: str) → str
```

Convert to base64 string. 



**Args:**
 
 - <b>`path`</b> (str):  path of the input. 



**Returns:**
 base64 string. 


---

## <kbd>function</kbd> `to_convert`

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

## <kbd>function</kbd> `to_raster`

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

## <kbd>function</kbd> `to_rasters`

```python
to_rasters(plotd, ext='svg')
```

Convert many images to raster. Uses inkscape. 



**Args:**
 
 - <b>`plotd`</b> (str):  directory. 
 - <b>`ext`</b> (str, optional):  extension of the output. Defaults to 'svg'. 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.viz.line`
For line plots. 


---

## <kbd>function</kbd> `plot_range`

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

## <kbd>function</kbd> `plot_connections`

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

## <kbd>function</kbd> `plot_kinetics`

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

## <kbd>function</kbd> `plot_steps`

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

# <kbd>module</kbd> `roux.viz`




**Global Variables**
---------------
- **colors**
- **figure**
- **io**
- **ax_**
- **annot**


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.viz.scatter`
For scatter plots. 


---

## <kbd>function</kbd> `plot_trendline`

```python
plot_trendline(
    dplot: DataFrame,
    colx: str,
    coly: str,
    params_plot: dict = {'color': 'r', 'lw': 2},
    poly: bool = False,
    lowess: bool = True,
    linestyle: str = 'solid',
    params_poly: dict = {'deg': 1},
    params_lowess: dict = {'frac': 0.7, 'it': 5},
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

## <kbd>function</kbd> `plot_scatter`

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
    params_plot: dict = {},
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
 - <b>`params_plot`</b> (dict, optional):  parameters provided to the `plot` function. Defaults to {}. 
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

## <kbd>function</kbd> `plot_qq`

```python
plot_qq(x: Series) → Axes
```

plot QQ. 



**Args:**
 
 - <b>`x`</b> (pd.Series):  input vector. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `plot_ranks`

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

## <kbd>function</kbd> `plot_volcano`

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

# <kbd>module</kbd> `roux.viz.sets`
For plotting sets. 


---

## <kbd>function</kbd> `plot_venn`

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

## <kbd>function</kbd> `plot_intersections`

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

## <kbd>function</kbd> `plot_enrichment`

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

# <kbd>module</kbd> `roux.vizi`






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.vizi.scatter`






<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.workflow.df`
For management of tables. 


---

## <kbd>function</kbd> `exclude_items`

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

# <kbd>module</kbd> `roux.workflow.function`
For function management. 


---

## <kbd>function</kbd> `get_quoted_path`

```python
get_quoted_path(s1: str) → str
```

Quoted paths. 



**Args:**
 
 - <b>`s1`</b> (str):  path. 



**Returns:**
 
 - <b>`str`</b>:  quoted path. 


---

## <kbd>function</kbd> `get_path`

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

## <kbd>function</kbd> `remove_dirs_from_outputs`

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

## <kbd>function</kbd> `get_ios`

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

## <kbd>function</kbd> `get_name`

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

## <kbd>function</kbd> `get_step`

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

## <kbd>function</kbd> `to_task`

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

# <kbd>module</kbd> `roux.workflow.io`
For input/output of workflow. 


---

## <kbd>function</kbd> `clear_variables`

```python
clear_variables(dtype=None, variables=None)
```

Clear dataframes from the workspace. 


---

## <kbd>function</kbd> `clear_dataframes`

```python
clear_dataframes()
```






---

## <kbd>function</kbd> `get_lines`

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

## <kbd>function</kbd> `to_py`

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

## <kbd>function</kbd> `import_from_file`

```python
import_from_file(pyp: str)
```

Import functions from python (`.py`) file. 



**Args:**
 
 - <b>`pyp`</b> (str):  python file (`.py`). 


---

## <kbd>function</kbd> `to_parameters`

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

## <kbd>function</kbd> `read_nb_md`

```python
read_nb_md(p: str) → list
```

Read notebook's documentation in the markdown cells. 



**Args:**
 
 - <b>`p`</b> (str):  path of the notebook. 



**Returns:**
 
 - <b>`list`</b>:  lines of the strings. 


---

## <kbd>function</kbd> `read_config`

```python
read_config(p: str, config_base=None, convert_dtype: bool = True)
```

Read configuration. 



**Parameters:**
 
 - <b>`p`</b> (str):  input path.  


---

## <kbd>function</kbd> `read_metadata`

```python
read_metadata(
    p: str = './metadata.yaml',
    ind: str = './metadata/',
    max_paths: int = 30,
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

## <kbd>function</kbd> `to_info`

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

## <kbd>function</kbd> `make_symlinks`

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

## <kbd>function</kbd> `to_workflow`

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

## <kbd>function</kbd> `create_workflow_report`

```python
create_workflow_report(workflowp: str, env: str) → int
```

Create report for the workflow run. 



**Parameters:**
 
 - <b>`workflowp`</b> (str):  path of the workflow file (`snakemake`). 
 - <b>`env`</b> (str):  name of the conda virtual environment where required the workflow dependency is available i.e. `snakemake`. 


---

## <kbd>function</kbd> `to_diff_notebooks`

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

# <kbd>module</kbd> `roux.workflow.knit`
For workflow set up. 


---

## <kbd>function</kbd> `nb_to_py`

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

## <kbd>function</kbd> `sort_stepns`

```python
sort_stepns(l: list) → list
```

Sort steps (functions) of a task (script). 



**Args:**
 
 - <b>`l`</b> (list):  list of steps. 



**Returns:**
 
 - <b>`list`</b>:  sorted list of steps. 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.workflow`




**Global Variables**
---------------
- **io**
- **df**


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.workflow.monitor`
For workflow monitors. 


---

## <kbd>function</kbd> `plot_workflow_log`

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

# <kbd>module</kbd> `roux.workflow.task`
For task management. 


---

## <kbd>function</kbd> `run_experiment`

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

## <kbd>function</kbd> `run_experiments`

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

# <kbd>module</kbd> `roux.workflow.version`
For version control. 


---

## <kbd>function</kbd> `git_commit`

```python
git_commit(repop: str, suffix_message: str = '')
```

Version control. 



**Args:**
 
 - <b>`repop`</b> (str):  path to the repository. 
 - <b>`suffix_message`</b> (str, optional):  add suffix to the version (commit) message. Defaults to ''. 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.workflow.workflow`
For workflow management. 


---

## <kbd>function</kbd> `get_scripts`

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

## <kbd>function</kbd> `to_scripts`

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


</details>
