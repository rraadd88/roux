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




```python
from roux.lib.io import to_dict
to_dict(d,'tests/output/data/dict.json')
```




    'data/dict.json'




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




    'data/table.tsv'




```python
from roux.lib.io import read_table
read_table('tests/output/data/table.tsv')
```

    WARNING:root:dropped columns: Unnamed: 0
    INFO:root:shape = (150, 5)





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

<a href="https://github.com/rraadd88/roux/blob/master/examples/roux_query.ipynb"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## Helper functions for querying data from Biomart 
<details><summary>Expand</summary>

### Requirements


```python
# installing the required roux subpackage
!pip install roux[query]
```

### A wrapper around `pybiomart`.


```python
from roux.query.biomart import query
df01=query(
      species='homo sapiens',
      release=100,
      attributes=['ensembl_gene_id','entrezgene_id',
                  'percentage_gene_gc_content','hgnc_symbol','transcript_count','transcript_length'],
      filters={'biotype':['protein_coding'],},
)
```

    INFO:root:hsapiens_gene_ensembl version: 100 is used



```python
df01.head(1)
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
      <th>NCBI gene (formerly Entrezgene) ID</th>
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
      <td>4535.0</td>
      <td>MT-ND1</td>
      <td>47.7</td>
      <td>1</td>
      <td>956</td>
    </tr>
  </tbody>
</table>
</div>




```python
from roux.lib.io import to_table
to_table(df01,'tests/output/data/biomart/00_raw.tsv')
```




    'tests/output/data/biomart/00_raw.tsv'



#### Documentation
[`roux.query.biomart`](https://github.com/rraadd88/roux#module-roux.query.biomart)

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
!pip install roux[viz]
```

### Example of annotated scatter plot


```python
# demo data
import seaborn as sns
df1=sns.load_dataset('iris')
# plot
from roux.viz.scatter import plot_scatter
ax=plot_scatter(df1,colx='sepal_length',coly='petal_width')
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

    WARNING:root:overwritting: plot/scatter_annotated.png



    
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
from roux.viz.dist import hist_annot
ax=hist_annot(df1,colx='sepal_length',colssubsets=['species'],bins=10,
          params_scatter=dict(marker='|',alpha=1))
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/hist_annotated.png')
```


    
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
ax=sns.heatmap(df1,
            cmap='Blues',
           cbar_kws=dict(label='mean value'))
from roux.viz.annot import show_box
ax=show_box(ax=ax,xy=[1,2],width=2,height=1,ec='red',lw=2)
from roux.viz.io import to_plot
_=to_plot('tests/output/plot/heatmap_annotated.png')
```


    
<img src="data:image/png;base64,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" />
    


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


    
<img src="data:image/png;base64,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" />
    


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
df1=sns.load_dataset('iris')
```


```python
# import helper functions
from roux.viz.io import *

## parameters
kws_plot=dict(y='sepal_width')
## log the code from this cell of the notebook
log_code()
# plot
fig,ax=plt.subplots(figsize=[3,3])
sns.scatterplot(data=df1,x='sepal_length',y=kws_plot['y'],hue='species',
                ax=ax,)
## save the plot
to_plot('tests/output/plot/plot_saved.png',# filename
       df1=df1, #source data
       kws_plot=kws_plot,# plotting parameters
       )
assert exists('tests/output/plot/plot_saved.png')
```

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
    numpy     : 1.18.1
    sys       : 3.7.13 (default, Mar 29 2022, 02:18:16) 
    [GCC 7.5.0]
    seaborn   : 0.11.2
    tqdm      : 4.64.1
    json      : 2.0.9
    matplotlib: 3.5.1
    re        : 2.2.1
    logging   : 0.5.1.2
    scipy     : 1.7.3
    yaml      : 6.0
    pandas    : 1.3.5
    
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




    
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASgAAAFCCAYAAABCaEAlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAA9hAAAPYQGoP6dpAABzI0lEQVR4nO2dd3xUVfr/37dMTya9h44iogh2pQp2fyquomLDFV3UteDqrmVVxK64VsS2ll17Wdd1UVlRxIog+kVFxUIv6T2Zeu89vz8mGTLMTAghISGc9+vFS3Puuec+k8x85pznPOd5FCGEQCKRSHogancbIJFIJMmQAiWRSHosUqAkEkmPRQqURCLpsUiBkkgkPRYpUBKJpMciBUoikfRYpEBJJJIeixQoiUTSY5ECJemx3H777SiK0mafBQsWMHz4cFwuF3fccUenPHfjxo0oisJzzz3XKeNJOo4UKMkuw7XXXkv//v3x+XwAbN68mZNOOok999yTDz74gDPPPJNDDz2UyZMnd7Olks5C724DJJL2YlkWNpstOqtauHAhgUCAhx56iKKiIgCEEOi6fFv3FuRfUrLLMHv2bGbPnh39ecOGDaiqGhUngCVLlnSHaZIuQi7xJNtF//79efTRR7npppvIzc0lNzeXxx57jHA4zJ/+9Ceys7MpLi7m6aefjt4TDoe5+eab6du3L06nk5EjR/Lmm2/GjGsYBjfccAN5eXmkpqby+9//nsbGxpg+F154If379wfg/PPP54YbbsCyLBRFibYPHjyY888/P3qPZVnceeed9O3bF7fbzdFHH826detixv3f//7HiBEjcDqdHHjggSxdurTzfmGSHUNIJNtBv379RL9+/cQ555wjPvzwQzF58mQBiGOOOUZMnz5dfPjhh+LEE08UmqaJH374QQghxGmnnSbcbrd44oknxCeffCLOPfdcoSiKePXVV6PjXnTRRcJms4n7779fLFq0SEyfPl04nU7R+i06bdo00a9fPyGEEGvWrBEXXXSR0DRNfPXVV+K7774TQggxaNAgMXXq1Og9f/7zn4XX6xX/+Mc/xAcffCAOPvhgMXToUGGaphBCiI8//lhomiZOPvlk8dFHH4mnnnpKFBQUCEA8++yzXfvLlGwTKVCS7aJfv35i8ODBwjAMIYQQNTU1wmaziX322UdYliWEEKK0tFSoqipuu+02sWjRIgGI559/PjqGZVlizJgxon///kIIIVavXi0URRF33313zLMmTpyYVKCEEGLmzJlC07SYe1oL1MaNG4XD4RAvv/xy9Pr69euFoijigw8+EEIIMW7cODFs2LCoYAkhxPPPPy8Fqocgl3iS7eawww5D0zQA0tPTKSgoYOTIkVHndV5eHnl5eaxfv573338fh8PBGWecEb1fURTOO+881q5dy2+//caHH36IEIKzzjor5jnjxo3bITvnz58PwKmnnhpt69OnD3vssQeLFi0iEAjw+eefc8YZZ6CqWz4KO/pcSechneSS7cZut8f8bLPZYj7gAG63m1AoRHl5OTk5OdhstpjrxcXFAFRXV1NWVgZAYWFhTJ9txUBti9LSUoLBYJy9LdeqqqowDKPTnyvpPKRASbqUzMxMysrKMAwjZvt//fr1AOTm5pKWlgZAZWUleXl50T7BYHCHnu31enG73Xz++edx17Kzs/F6vdHntmZHnyvpPOQST9KlTJw4kXA4zIsvvhhtE0Lw7LPPMnToUPr378+YMWMAeP3112P6vP/++zv07DFjxuDz+fD5fIwYMSLmX3FxMampqYwYMYI33ngD0So1f8vSUNL9yBmUpEs56qijOOqoo7j00kvx+/3svffePPLIIyxdupR33nkHgP3224+TTjqJv/zlL9hsNvbaay/+/ve/U19fv0PPHjFiBJMnT2bSpEnccMMNHHjggYTDYZYvX85VV10FwE033cRpp53Gueeey/Tp0/n+++956KGHorMrSfciZ1CSLkVRFN566y3+8Ic/cMstt3D00UezefNm3n//fY499thov9dee43f//73/OUvf+F3v/sdGRkZ3H333QCEQqEOP/+FF15g+vTpPPDAA4wfP55TTz2V999/n4aGBgB+97vf8cYbb7B48WKOPvpoXn31Vd58801cLtcOPVfSOShCyLJTEomkZyJnUBKJpMciBUoikfRYpEBJJJIeixQoiUTSY5ECJZFIeixSoCQSSY+lVwZqXnPNNRiGET1CIZFIeg51dXXous599923zb69cgZlGAaWZXW3GRKJJAGWZWEYRrv69soZVMvMadasWd1siUQi2ZqZM2e2u2+PEKjy8nIuu+yyhNfmzp1Ldnb2TrZIIpH0BHqEQGVnZ/Pss8/GtP33v/9l6dKlUpwkkt2YHiFQqqri8XiiP5umyccff8ykSZO6zyiJRNLt9Egn+ZIlS/D5fDL1qkSym9MjZlBb89577zF+/HicTmfSPsl8Vi1kZWV1tlkSyW6DZdSA1QSKiqJloaiObrGjx82gVq9ezS+//MIxxxzT3aZIJLsdwgpg+Jfj2/QHGtceTePa4wiU344V3twt9vS4GdS7777L8OHD4xLZb82cOXOSXtuebUyJRLIFM/grvg3nAM1xhCJMuP5fmP6vcRc/i2rLa/P+zqZHzaBqa2v54osvOO6447rbFIlkt8My6whWziYqTq2vhddiBlfudJt6lEAtWLCArKwsRowY0d2mSCS7H5YP0/910stG40c70ZgIPUagDMNgwYIFHHPMMXE11iQSyc5ARVGTn19V9NydaEuEHqMEX3zxBX6/nyOOOKK7TZFIdksUPRtb+jlJr9tSj016ravoMU7ysWPHMnbs2O42QyLZbVEUDXv6aZj+5aj2QjTnPmAFCTe+j817MspOdpBDD5pBSSSSHoDqxZl7DcKsJ1Axm2DNM+juUeiug1FVz7bv72R6zAxKIpF0PyK0lqYNZ4EIRH62GghWPYjR9BHOgvvQbEU71R45g5JIJACYRhXBqgej4hRzLfAtVmj1TrdJzqAkkt0UYfqxzCqEsRHQUPQcjKbPk/Y3GhZg84zZeQYiBUoi2S2xzDrC9W8TrLwPRBgAd9/XQXGA8CW+qRt8UHKJJ5HshljBnwlW3BUVJwCjaTE278lJ77F5T9gZpsUgBUoi2c2wzAaCVY/FtYeq5mJPn4Jq6x93zZ5xAYq28wM15RJPItndEAGs8MYEFwL4N1+Jq+ABzNCvGI0LUFQvtrRTUfRCNJsUKIlE0tWobjTnUIzGTXGXrPAawg3zcGTPQE85FlXtXomQSzyJZDdDVT04Mi8GlPiLih1b2mkoitbt4gRSoCSS3RLV3h9X4VwULSfaptj6NOd8ajsX29ZYZhNmaD1m4AfM0Foss77T7Ox+iZRIJDsdRXWje8bi6fsawqwFRUHRMlD1nG3e2xorXE6g8kGMhv8CJgCaexyuvJmotvwdtlPOoCSS3RRFUVBteWjOIWiOPbdfnKwmAlUPYTS8RYs4AZi+j/GXXhvJa76DSIGSSHZRhLAQVqj7nm9UYdS/nfCa6f8KYVbt8DPkEk8i2cWwzEaEsZlQ7asIYxO6ezR6yhGoO/kgL1YDrWdOWyPMSmDwDj1CCpREsgshTD9G4/sEym6MthlNn6BUzcXd53k0x6CdZ4yaQmQnUCS8rGgZO/6IHR5BIpHsNCyzkkBZfNUiYdUSKL8Ny6zbabYoWia6J3EGXNWxF4qWvcPPkAIlkexCWIHvSbasMv1LIztyOwlVS8WZeyOa67DYdsdQ3IUPo+o7XjxXLvEkkl0IkSBXUyzxJaO6EtWWj6vwbwijCmFWoWjpKFpWp4gTSIGSSHYpNOd+Sa+p9kEoqncnWtP8XC0dtHSg8/1fcoknkexCKHo2trQzElzRcObe3Gkzl56CnEFJJD0UIUyEUYrhW4YV/AXNtQ+acz8cWVeiuQ4iVP0kwqxobrsCRc/HDK3BaPwIYVaje8ai2gdsfwCmUYEVWhvZHdTS0T0TUGy5smiCRCKJIITADP6Eb8P5WzJc1oKipuPu80/s3uPR3YdGEs6pHlAUjPr3CJTfHB0jVPMMqmM47qKHUPX2lYyywmX4S67CDCyPtgUr/4Yz92b01P+HqqV03otsBz1uiffZZ58xc+ZMzjnnHJYvX97d5kgk3YIwyvFv+mNc+l1h1eIv+ROWUYWqZ6La8lC1FES4LEacWrCC3xGqeRnRKnNm0mcKg1DdazHi1EKg/FaEUdbh19NReswMyrIsHn74YVasWMFZZ53FjBkz8Hp3vsNPsmtjGZUgQrQUAVCUHvcd3C6EWdm8fBuOzfs7FC0NYVYSqvsXVnAlwqyGVv6mcOOCpGOF6l7Cnn4myjYO7wqjilDtS0mvhxveRXNcvv0vZgfoMQL1zjvv8P3333PbbbdRWLh96R4kEsusx/R/Q7DyPqzQahQtE3vGNGzeE1H1HQ8Y3NkIy48z568I4SNYPRdhlKPYinGkT0VYdc0i3Kq/WZl8MKuRZNHeW3VsPr6SxCZjx8/WbS894uslHA7z1ltvceqpp0pxkmw3QlgYTYvwb740WrtNmNUEK2cTqJi9U6OrO4uIw/s3gpUPIIxyAER4I4GKOxDCAi12t073jEs6luY6EFT3th+qutFcByW9rKdOaJ/xnUiPEKiVK1fS0NCA2+3m+uuv55JLLmHOnDk0NCRXc4mkBWGUE6y4N+E1o+G/3fLNv+OYhOveSHglVPMcijBi2jT7nqj2PRP01nBk/xlVS9vmE1UtDWfOn0m0sFLtg9EcQ9pjeKfSI5Z4GzduRNM0PvnkEy688EKampp4+umneeihh7jxxhsT3nPZZZe1OWZWVu+KB5EkR1j1EZ9MEqzQb2iOgV3ybMtsRJhVmIHvUVBRnfug6lko27klbxkVWOHNWKFVrbISJMkUIHyRZR7F0SbVlou76HGCNc9GhE340ZwjceRch+bYo912qPaBuPu8RLDibszAN6A4sXlPxZF5Qbt3AjuTHiFQfr8fXde5+uqr8Xgif9jzzjuPe+65h+rqajIzM7vZQkmPRrG3fVnrms0Wy6glVPsCoerH2OLjUXFkX4Mt7Xeo7XyuFd6Eb9MlWKHfmlsUXIWPtH1Tgtes2vJxZv8JR8bvAQtFdaNo6e18Nc3Dqk501z6ohXOadxBVFC0TRW37d9xV9AiBSklJweFwRMUJiPqiampqEgrUnDlzko43c2b8aW9J70VRM9BcB2L6l8VfVD2otr5d8lwr+BOh6rlbtxKsvBfNNQLVNWLbY5j1+MtmtRInAIEw61G0HIRZEXePah+MoiX+0lZUB4raCal29XQgfYfH2VF6hA9qyJAh1NfXU1JSEm0rKSlBVVUKCgq60TLJroCqp+HMuxVF36pum2LDXfhofHsrLLMGK1yOZTZu8zlCWJGlmFGOZdQSrH4qad9Q9bMIa1sHe0GYNZi+z+Lvr3kGZ95MFNsA7JnTcebehD3jAlRbf1wFf+t1R1qS0SNmUP369WPkyJHMnTuXCy+8ECEEL7zwAhMmTMDtbsfug2S3R7P3x9PnZczgDxi+ZWj2AWjuw1H1fBQl/m1uGdWY/mUEq59AGOVojn1xZF+GahuAorni+4dLCTe8Q6juVRAhnDk3IczypPYIswwhgig42zbc8iduDv2GFdqAM/daQlWPEQ5vRLUPjgixtuMzpF2FHiFQAFdeeSXPPfccN998M4qiMH78eM4555zuNkuyC6HaClBtBdhSjmyzn2U2EKp+ilDtP6Jthm8RxvpPcBc/he6OzW9kGWX4Nl2KFVoZbQs3zkdzDIuGNWyN5joYpR1b+4qWCooLRKxQ6SlHgWjEv+niaJvpr8K3cQnO/NnYUo9JKLy9jR7zCt1uN5deeimXXnppd5si6eUIsypGnLZg4S+7BU+fF2IO2Jr+71qJkw7oGI0LcBfOIdw4P3IerjWKK1IuXLFt0xZFy8Ge8fs4X5bNezL+kj8lvCdQfju6a38UW+93f/QYgZJIdhZW4Kek10R4A8Ksh2aBElaYcP1b6Km/w5FxFsJqRAgTVc/ECKzElX8/wapHsEK/AKA69saVdyuKlo0Z2ogwy1EUG4qWg6Lnxh29UVQ79vQpKKoz4tOyGkBxA2pctPiWF1CHMGtACpRE0gvZ1pZ5axFRFLSUY1E1D00bp4HVHJWu2LFnXgrOPFyFc4EgoKCoaaAohGr/SajqcSAyu1K0TFwFD6K59oubWal6FvaM87GlHo+w/CiqM3KmsE0bte16ybsqPWIXTyLZmaj2IUDi5ZfqHB5TjURRdHTnEPybZ2wRJwARIlT1IAg/qi0/4pS390fVMzB9SwlVPUKLOEHk6I1v04WI8Jad6tYoio5qK0RzDEK1FUWWmEmyYyp6IYq64xVTdgWkQEl6NEIILLMOy2xqV3/LDGCGy7HMhq3am5rbfah6Ns782+JvVlNx5d0aSWHbcp9lEar7N2DE9wdC1U/EzHYso4pgVZIYPRFqM+tAaxQ9F1fBfcQtchQHroLZqLbkoRO9CbnEk/RYrHAJ4cYPMBreAcWFPeMcNOd+CbMTWFYAEd5IqPYlzMAKVFsB9oypKHohmFUEa/6JFVqNZh+MPeM8NPdoPP3eIlT3GiK8Ec11CHrqUaj6VsUvRQARXtOGjRuhdSEDEcYKr0/a3wyuTHqtNYqio7sOxNP/P4Tr3sQK/oLqHI7d+/9QdnaBzm5ECpSkR2KFN9O04TyEsTna5vcvQfdMwJl3S5xIWYEV+DZdGHUsW8EVGI0LcORcj+n/GqPx/eb2Hwg3/DfiD/KMw5lzPQgDFDuKosQbojhRHcOg6ZOEdqr2QZEwgWh/O6p9EFYwsSNed45o9+9AUZ1o9gGo2VdFdgqT2diLkUs8SY9DWGFCtS/GiFMLRtPCuNgjM7QpUswywa5XsOI+bN5JW7VaBMpnIowSFEWLHA9J8sFXVRVbylGgOBJet2dMQ9W3+INUPRNH9pWJX5jiRktJnhYlGYqitmljb0bOoCQ9DmFWE67/b9LrobrX0VwHbtmyt+qxki3DFCJ+m8K5IIKg2DCaPiFc92+EWY0ZssDyo6ipkQycaqwQCcsg1DAfV/49BCru3SKaqhdH1mUY/mWo9gHNB2uVyLlA5344c28hUDkbrIjvTLH1bT6i0vtDAzoTKVCSHkobBShFO4tTKnZc+bMJ172J0TifSPoSHT31WFwF94IQNK09vrmvA3vGVOzp58adcxOhVQSbPsSR+QcUPbP5+Sah2lexwutQbX0IlkcOqCu2Ylx5d6GnnkSKZ3QkXkmxoWgZ211dRSIFStIDUbQMbKknEKp9PuF1e9ppsQGPaiqqrR9WeF1sv/RzCde/hdH0UatWA6NhHogQjqwrtjSLIKHqJ0Gx48i4CEWNhCEoqo4tbTL+zRcTKL8lzhZb+lmYTYu2DBPeiG/TBXj6/RvNPhBsMkPsjiB9UJIeh6LasWech6LFzzg09+GozQnYRPNMSrMX48idydaxTZrrwK3EaQtG4wKEWRvXHqp+Ji7FieYciuaMT4Wr6PnYPOMwmj6OvSDChGpfQVjhqI2SjiFnUJIeiWorwtP3ZUL1/8VoeBdFdWFLPydykFeECNW/g9G4AEXPw572O1T7Hnj6vkaw5lms4A8oej6KGp+VYAsiYa6lSLbK2FJPqp6Dq3A2hu9LQrUvghXElno8mnNf/GV/JW45qnrR3YdiNH1MuOFdFM0bmfXpxc15liTtRQqUpMei2gpxZF6IPW0yKBqqloYV3kjThvNjdvjCtc/jyPkrNu/JOHNvRFj1KIoLYW4jF7mSKBWKDSVBu6rnYveehO4eC1gIxUGg5GqEUbp1T1z5dxGsfBAr9OsWG+tew54xrXnXL73dv4PdHbnEk/RoFEWLFKjU0hCWj0DlwwnDD4IVdyDMSlQtBc1WiKpnoGiZqM59E46rOffHCv4Q127znoTSRjI4VU9H1TPRNA+OrOlx13XPaEzf4hhxaiFU83RC2yXJkQIl2WUQZi1Gw/yk142mT2N+VvVM3Pl/az5716rdMQxn/u2EGt6Padc8Y3FkXb6NpWGrceyDcObeBK0O/+opRxNueCfpPaG6N9s1tiSCXOJJdh2ERbIzcZHLCc7rKTqOnL9EjqyYlRHHu+IAxYOn+AlEuBzLqkXVC1D07JhzeNtC1bzYvKegu8dgGRsAFUXPbzPVr7C2nVpYsgUpUJJdBy0FzTkcM/Bdwss296iYny2zhmDVXML1bwBapHil5QNMbGln4ci6HM21LzuSuERRnSj2YlR7pASUZfnQPWMxGv+X2EbvCTvwtN0PucST7DKoWjqOnBtJ9L2quceibBVzJIxqwvUtSyqzuax3pNZcuO61bTvRO2Kj6saRdXns+byWa45h3VL8cldGzqAkuxSaY088fV8nWPkQhn8pipaOI+N89NRj4iPAzWqSRaRrzuGASaj+v1jhDWjOEZFcTEmKU1pmA8KoxPB9ClYA3TMaRc9H1ePLP6n2vnj6vkGwei5G0yIU1YMtbQp27yTUNirMSOKRAiXZpVBUO5pzCM6Ce8FqBEVD0bITH6RN4uzWnMOxZ5xD0/rTI+fzWsa29cFT/DSqrTimv2XWE657jWDl/dG2YNWD6ClH48y9MS6zgqLoaI4BuPJmIcwGQEHRs1B2kyyYnYlc4kl2SVQtBdWWj6rnJD3lr6gZqLb+ce32jAvwl90UI04QyUfuL7sjrkaeCG+IEacWjMb343YOY5/vRrXlodpypTh1EClQki7BNEOYoXXN/7akuRVWGCtcGvnXSggsyyJsGIQNA8vq+PEQy6iOjG2Uo9oKcRU8gKJtmeEoahpC+KNZBuLs9n3SvDRstleYhGpfSfq8UM0zWEZiX5ZlNja/1jKEFU7YR9I2cokn6XSs8CbCdf8iXPcqwqxDc43EkXUNip5FuPZFQvVvgBVE94zBkX0VplpMeU0ddU0R0UjzeMjNyMRu23bZpugzzQaswAoClfdhBVei6Pk4Mi9CTzkKd58XmgthrkJ17Jsg+rs1InZmJYzER2JaLpu1kYR3rdtEGCu0lkDF/ZGqwaoLe9rp2NPPQbXtPkU3OwMpUJJOxQxtJFD2V0z/V1va/F8jjE0Eym6IydtkNH2E4VuMvegV6pt0hBAA1DY20uDzM6ioqF0iJYSF6fs8po6cMEoIlN+KLfATjpyrsaVMACZE7An8nHQsRctCUVO2/Kw60D0TMZJk1NRcB0eKb7bCCq2laf3kLQn0rEZCNc9gNH2Ku/jJpI54STw9RqBKS0u54oorYtoOP/xwZsyY0T0GSTqEMEpjxAlAtfVDWLWJk8qJAGbtE6R7ZlDnj4QAmKaJaZnUNjaQk56xzUySwignUH5n5Ifm3EvCbAThI1z/OvbM80FrVSFFdaK5R2H6Po8by575B8RWWRF0zygULSd+JqXYcGRdElNB2DKbCFY9nDC7pxX6FTPwE2qKFKj20mMEqqKigj59+nDrrbdG23S9x5gnSULEx7IZYZShOvfD8H0R10d1DsPwLU06hun7jJziy0i3N59fsw2iqlGnvqmJTK8XXWv7fSCseoTw48i5FtVWjBXehKplITAJVT2GFfwNzT5gy/MCP2BPOxXTPoBQ/b/BakKxFePIuCBSBMGqBbb4rVRbIe4+zxOsfBCjcQFgojn3x5F7A6qt31a/kAaMpnjhayHc8C62lPFtvh7JFnZIAV5++WVefPFF1q5dSygU+42hKAo//5x8Kr01FRUV5Obm4vF4dsQkyU7EDP6Kb9NFCKMcAN1zDJprv/iOVhBFS0s6jqKmIPyLMSpub27RyMr8EwHHUSi0Iw93c+bMUPXjMVHmip6LK+82hBL7nlJUD/7Nf0T3jMWVOxMUW3M59FexQj9jS5sca75l0RDKwG+7Cm/hZYDAF9Jp8qWSaVNjP0SKGnk9ZuLjLq1r7km2TYcFatasWcyaNYtBgwax//7743a7t31TG1RWVpKTI1Oi7ipY4TJ8m/4QFScAo2kBjqzpBFEAsaXdtxhX/l2torpjsXlPJNzY+hCwiVE9m9Si4WhaccJ7WqOoKYTr34w7AiOMcvxlN+Eu/mdMu+YYDIodo2kRRqtsmNDiU0qPaQsZBhvLI6+zKlpuzwBqcDjspHla+ay0LGxpZxKqTlwbzx5XwEHSFh0WqMcee4xLL72UOXOSFCncTsrLy1myZAlffvklxcXFnHjiiYwcOTJp/8suu6zN8bKykqfMkHQMy2yMnGVT3VhGOcIo27oHYd9inLk3ESjfslRH+LBC67BnXkSo+qmYOzTnfmiOvSPpdrciXPsPVOcQFOEHtIRR2wDC8mE0Lkx8zShHmJVA32ibouXgKngA/+bLaX34WNFyIiWtWs32hBBU19cnHBugoqYGj9OFrkXinBRFw552KkbTx1jB72P62rMu361q2nUGHRYoTdOYPHnytju2k9NOO43f/e531NTU8MEHH3D33Xdz/fXXM2LEiE57hqRjWGYjVmgVwaq5ka16Wz/smRdiSzuTcF1sjFCocjaOnBvw9P034YZ3EWYVumcUqmNvFC0dW+rxhOvnIawGbCnHYJk1+EtviHumauuHPeMcQlWPYTQtQFGc2NPPQvdMiK+qKwK0VWRBGLHObUW1o7sPwdP/vxiNH2CF1qG5D0V3jUS1xVZdEUIQCiePYQobZnT3MTq+nouz4GHM4G+YTe+D6sWWejyKXoCqJS5nLklMhwXq0ksv5Y033mDcuO2v85WI/PxIfEhhYSHDhg2jsrKSefPmJRWotmZuM2fO7BSbJCCEgdH0CYHSa6JtprEZ/6bFOLKvQXePiZxPa0Ww8kFs/f+DM+eq+AG1NLScyIFZYYUIV9zbXLKpNTacuTfiL7kqUhWFyIIxUH4rmnMersIHYiukqB5Q3AnGiaDY+sa3qU40ez+0zGltvn5VVfG4nDT6E4/tcthRt9plDBkGqzf7UJQiXI6LsUyBrzSA29FEcW4KNrn5027a/Zu68847Y34WQvDmm2+ycuVKxo0bh6rGBqUrisL111/fYcP22GMPli9f3uH7JZ2DMCoIlM9KeC1YNRdX/l1xAmXP+D2Ktu1DsZHiCGcTrn89Ujm3GT31yObZV03cPWbgG8zgzzECJYQDe/oUQjVPx/XXnCNi4po6QponhYqaWqwEBRDyMrPQtC3HWCzLoqKmGrM5Gr7Bt0XYmgIBguGQFKjtoN2/qRtvvDFhe0lJCR9++GFc+44K1Lp16ygu3raDVNK1CLO6OU1Joou+5oq7OmCA4saeeQH2tDOiZZtiuguBYZoYZmRZpGsamlaEu89rCGMTiEgclGIrIliZfIYcrvsXqn1IpDKw6kDBgWofgD3jgsixFOEDNPSUidi8kzADv6HZ+3T4d2DTdQYWFrKxopxA8261TdcpzM7BZrMRMsIYhomiRGZcbS0JaxsaSHHt2IbS7kS7BWpHzkdti0AgwIcffsjw4cNxOBy8//77rFy5knvvvbfLnilpL20f11T0DFL6z0OIAKgeVC03oThZQuAPBthQVoZhNguRotAnJw2X2IC/7Eawmp3Rqhdn9tUomgcjUfpcxUaw4h6Mxnebf+yLM/cGROBHXPm3RWxWNIymz/GXXIUz/8Ed+QWgKApOh4P+BYWYpokANFVFVVUafE1srqiMzq40VSU/Kwvd56O+Kf683+5YvnxH6PBc85///CcnnngiGRnxcR01NTX8+uuvHHzwwe0aq7GxkWXLlvHmm28SCAQYPHgwt99+O0VFcseju1G0TBQtu3knbKtrahqqltOu82Vhw2BtSUmMQ1kB7JTgL7mS1mEJWPUEym/BVTgHo/GjON+SzTOOQPnt0Z9FeD2Bkmtx5s3CXzJjKyNtaI5B7Xmp20TXtOhuHYAvGIiGH7RgWhabKirol59Pg88X50DPSJVO8u2hwwL1+9//nq+++iqhQP34448cf/zx1NXVtWus7Oxs6djuoSh6Lq6Ce/Ft/AOx+cA1nAX3oGyVCykZdY0NcR9Wr8eGWf9PYsQpiiBc/za21KMJ178VbdVTjsQyyhBW7HtLWHUIsxpFL4ypnOLImQVa54WcWJYVtbayJt5H1kJtYyNet4e6pi0ZG9JTUrHbbAghsCwLRVHifLeSWLZLoH766Sf+7//+D4j4E+bPn8/KlStj+vh8Pl544QUZEd5LUBQFzbk/nv7/IVz7OmbwB1THEOxpZ6DailGUbb+FhBD4g/Fn0xyaifD/lvQ+K7QWR84NWOHS5owAZ2KFVxOsuC9xf6MCe8YfMBrfQdH7Yk8/C/QiNK19VVraImwYBEJBqurqEUKQnpJCeqqXpkAg6hBvTSgcJi8jE8M0UVWFrLQ0HDY7lmVRU19Pg9+HTdPIbG5vPTOTbGG7BGrDhg2cc845QOSNm8xxnpuby6OPPrrj1kl6BIpqR7MPQM35E8IKRhzT7RCm6P2KgsvhoMEX65MJGhpu22AIrkx8n30wNYFiTOcsLKHiFSZU/5WWvOJbI/Q9qBcHYk8fj2GqrC1roCATvJqFtgMzFcMw2FxZsdWOnB+7zUZRTi7ry+LTtzjtdtxOJ33z81GIOM8DoRBrNm+KEbS6piZyMzLI8qbF7AZKImyXQE2cOJE1a9YghGDgwIG8/fbb7LtvbGHElJQUGcXdS1EUHWUbB3eTkZaSQkVtTcwyr84XJjvnPMzGd4hf5ilo3qlUVvixmu8Jhu3kp1+EUXVXvG1qOqY+hPLy2KXf5soKPC7XDglUIBSKEacWQuEwTQE/HpeLJr8/5lpWWlrM8s00TUqqKhPOtsprakjzpEiBSsB2vds0TaNfv8jp7WeffZZRo0Yl9EFJJFtj13X6FxTE7OIBhCnEVfAwgbIbt/iV1DTsObOoC2RQkJ2KqqgoikIwFCKkjseeXoFR+w8gsp2v2vqh597P+mqNrSPKBRAMhRLmlTJNE8OyEJaFqqrouo6qKBim2bxbJ9BUjeqG5Edd6puayE5LjwqUpqoU5+Ri02OfZ1pWnIi1ptHvx2G3t/Eb3D3psJO8X79+/PTTTwmvKYqCy+WisLCQ3FxZxUISeU+4HU4GFRXHxEHpmoaijMPjfDOaaleoGdQ02rHb7ZRVV0cFzWm3k5eZSVA5F2//yWDWgOrEUtJYXRogbCSOPxIJnPChcJiSqsrozEhVFHIzMkl1u9lUWYEvEMlGkJGaGufcjxlbCFJcLgYVFoGitHpNseEEyUdIbqNkBwRqwoQJ7YrpGDx4MDfddFPUdyXZfVEUBZuuJ4ikVlFsBdB8Ds40TZzOIOtKS2J6BUIhNpSXM7CgEM2eA0SCLy3DADYlfa7T7oj52TAM1peVRoMuIRKnZdN11pZsJtxqhtfo95OTnp5wiQeRpatN17e5G6epKi6HA38wmPC6DN5MTIcFatmyZZxyyimcffbZHHnkkWiaRigU4oMPPuCll17iwQcfRAjByy+/zNSpU9F1nTPPPLMzbZf0UgRQWZt4C9+yLBoDfpyOLaJj03WKcnJYW1IS1z83IyPOtxMyjBhxArBpOqZlxogTRHbvWpz8W4uLrmlkedPaFSqgaxqF2Tms3rwpQWxUqtzFS0KHBerPf/4zl19+Oddcc01M+1FHHUVWVhavvPIKr7/+OqeddhqXXHIJDz/8sBSoBAgrhDBKCTctxAr+guY6EN19CIpe2KVRx2HDwB8MUu9rQlc10lNTsWlaQketEIKwYdDo8+ELBnA5nKS63WiqimGZ1Dc1EQyF8LhceJyu7Sp2AJHlVsgwqGuMHKnJ9KbhD8WHJbTQ5Pfjdjiorq9H122kp6TgtNkZXFRMeW0N/mAQm66Tm56B027HNE1qGxrwBwN4nC4S5cCz2/Q40WqhpLKSwpwcTNOiuj4SZpCW4iEj1btdr9Vpj9hYWVtLU8CPpmnkpKXjdjqlQCWhwwL15ZdfMmtW4kOko0aN4rbbbov+PHnyZF55JXnpnt0VIQzMwP/h2/SH6GHZcP1bKGoa7j7/RHPs0SXPDRkG60o2E2x1Zqyyrpb8rCwyUr1xO14t2+NWq6IGpdUK/fILKKmsJBgORds1VWVAYRHOdjp8t/YFAdh1GzZNJ2glFgybrlNeU0Njs9O5sraGwuwc0lJSKM7JxWwOgtRUFX8wyJqSzdFZS73PR2FWfHCpYVqkJDnEawlBSWUlexT3IS0lJeo/294vEEVRcNjtFGRnR22UwtQ2Hd577du3L//4xz8SXnv11VdJS9uS9KuxsVHmF0+AMCrwbb4y5iQ/RKKi/aXXYhnVSe7sOJHT9jUx4tRCaVUVYSO2hFK42V9jbbUsEUKwsbyM7PTYVL6Rox7lMTt1beEPBuL8O7WNDWR6kx8JSXW7o+LUwubKCgzDQFVVbLqOrmkYpsn6stKYJZVlWaAQJwzBcAin3Z5UdLLT0tGaHeA2Xd+h2W1rGyVts0Mpf6dMmcKGDRs4++yzKS4uprKykldffZV///vfPPTQQ9G+8+fPZ9iwYZ1icG/CCm/eckB262vBlZF0I0mySHYUw4xUS0mGPxhAU1UsIVCIzB62Fq3WY2lq/IfMHwximuY2P4Bhw0iYrTIYDiOEICPVS02rLX5FUSjIyqamIbH9/lAItcV2BYQlEgpleU0NRTk5bKqoiLle39REv/x8ahsbSW12WrcUE81ITcUSArP5d6GqqhSYnUCHBer000/Hbrdz7bXXMnXqVBRFQQhBbm4uDz/8MH/84x+jfY8//njOPvvsTjG4NyFE8riYSIeuqUabbNs8Oy0dBKzevImwYaAqChleL31y89hYUZ7wvmRjtbU137pPosBFgNLqKnIzMhhc3IdgOISqqNh1nc1VlQnjiXLS07Esi9WbNhE2I7ZnetMozs1lU3l5zCZ+ZFlZRf+CAkzLwjBMHDYbuq5hWQIEbCwvQwAOm42CrMiSrKyygvrm2Z7L4aAwOweHPT5hnaTz2KF116RJk5g0aRJr1qyhpKQEr9fL0KFD4xyt/+///b8dMrK3otr6Elllx39IFTUtLnl/pzxTVXE7nPiCsVVH3A4nuq6xqXJLelxLCKrq6gi4QuSmZ1BWE7vkVEicPiSS56l9O1upbndS57QQAruuR/1ZhmkmTPvjcbpQVZXNW9leWVdLqttNdnoGFVvtCkYEWMXpdMS0rSstiVn+BsNh1paW0Cc3L8Zx7w8GWb1pI4OK+7Tb3ybZfjrlKPWAAQM4/PDD2WeffWS4/nagaJnY089NeM2Rcx2K3vlVbnRNoyA73kmc4fVSUVub8J6m5ijnraUoKz2dusbGuP75WVnbrGUHEbFMT0m8xa5pGukpqTFb+LqmUdDs4FYVBafdjl3XyfCmUpnE9gafD7fTga5pOO326PszLyMz3g8VCiX0zQFU1NbE+cUEUN4qe6ak8+mwQJmmye23386QIUNwOp1ozVvULf+kU3zbqFoK9swLcebdiWLrA2iojr1wFT2FnnIEitI1Yu+w2RhUVIzHFTnlrzU7bc02HNuGaeB2OgGw22wU5+aRmerF43JFAy+ddjv9CwpIdbnb7UR22O0MKCgkLSUFRVFQFIW0lBQGFhQmPPrhsNsZXNyH4tw8Ut1uMr1pOO32mLiorTEti/ysLFLdbvIzsxhUVEx6ampc/FKyvOMQ2cm06/EhBU2BQJcmc9zd6bCKXHHFFTz++OMcf/zxnH766Tib37yS7UPVs7CnTUL3jIqkvFXsScsrddozm6Oa++bmNWeCVLb5IbPpEVESCBSUqChl2mykut0IBKrSMcexw26nICub3IwMEER3yxJhNe8Stg6aVKqhMCcHBeJ29wAQkV2+lp1IXdMYUFAY94y2vlTVZh/r1uiq2p7SopIO0mGBeuWVV5g1a1bSlCuS7UPtguXcttA0DY3Ih9Q0TVJdbhoSzCJURcFhsyVN9t8ZRQC2zlaZCMuyqKytjYvoFsDmigr65OXHCZSmRg4atw6TiIQflDGgoCBGlFLdbkqrqhI+Oz01NSb5XAvZ6elytdCFdHiJ5/F4GD16dGfaIulGtGbf1NaR0YoSCci06TqGaRIyDAwzNuwgFA5HMg20USyghZao9LAR3m7fjWGaSTMLCCKxTI5W9quKQmFOTpyDnOa+xlbP1zWdPrl5cX1dDgcZqalx8Vpej0eeoetiOiz906dPZ+7cuYwdO1amLe0l2G02BhQUEgyH8AWC2G06bocTVVFo9Psoq64mGA5jt9nIy8jAYXfQ5PdTWVdL2DAi2QYyMnHY7QlnVZG4pzqqGxoQlkWK20NeRgZ2m63dPqu2whcsS1CUk0uj34/dpuO0O9jUqhJLfP9YgdJUlRS3mz369KXJ78cwTVKafWyqqrBHcR8a/X4sYZHicstgy51AhwUqOzubZcuWMXToUA477DDsWzk0FUXhiSee2GEDJTuXlmwDLTMDy7KoaWigpGpL0YRgKITZvNxqHUjpDwZZW1pCcU4uXo8n5ourpWhCy7EYgPqmRhp9TQwqKm5XLqSWpWaynTaPy4nb6Yw684PhUNLsAQrx0eQQESlNVWNmYtFrdk3mbNrJdFig7r77bgBCoRAff/xx3HVZXqd3YJgmZdXxfpnI7CQSd9SS0rZlyVZaXYXb6cTeSqD8wWCMOLVgCUF5bQ1F2TnbnInruk5+VnZcGhYAh82OwxYrHrqmk+n1JoxWz0xLk7OfXYAOC9SaNWs60w5JD8UwzbhzeHabjWAoUiE3Jz0DTVObj7boBEMhymtrMEyTsGlgmhYOm63Ns3kNPh+GZcUIWjLcTgf98vIpqaokZBgoRHIy5WZkxi0rNVUlNz0DXdOprKvFsiK5ybPT0hOGGUh6Hju8/eDz+ViyZAmlpaWccMIJeNs45CnZ9Ug0E7aEQNNUCrNz2FxZEXNWz+10Rh3NazZvKf/k9XgozsllY0V53HiqorR7q15TNVI9HpwOB5awUIhkBEgmNrquk5MeESRhWSiqiq0DmQgk3UOHv0KCwSCXXHIJubm5TJw4kXPOOYe1a9cC8NJLL3H11Vd3lo2SbqQliLM1hmFg021x4gTgCwQSHkaub2oiEAqR4oovAZXl3f7llk3Xcdjs2G22bc6EFEXBrus4miPPpTjtOnRYoK666ir+9a9/8eSTT7JpU2yWwMzMTF577bUOGzV37lxOP/10ysvjv20lXY9pmoTCYcKGgRCCgqzsmA+1oiiEm68nor6pCcuyyE5LJzcjA29zjcTqhnrSU1Jj+jrtdtJTI22GYWA0P7MFISIZCdqbvkXSu+jwEu/ll1/mmWee4ZRTTom7lpGRQVlZWYfGbV0cVLJzMU2TsGFQURcJhrRpOtnpaYQNg755efgCEUe3y27fZgyToij4AgEMy8TjdNI/v4DS6mqcdjsZqV4syyI9NSWaL7y6vj66I5iekoo3xQMCahoaqG9qRFVVMr1p0W1/ye5Bh//SqqqSkpKS8NqqVatiEta1F8MweOqppzjhhBN46aWXOmqapIMEQiHWlmyOpiYJhcM0lfrJSksjEAoSaHaM1/t85KQnLzemNJduasmYEAqHqWtspE9ePpqmUZSTgxAiMhMzDNZuld2ztLoKp8POxvLYxHebKsrxOJ0U5+ZJkdpN6PASb9KkSVx99dVs2rSlmoaiKNTV1TF79mxOOOGE7R5z3rx52O12DjvssI6aJekgoXCYzZUVCYsfVdXV4fV4IjnMm5rwB4OEjDCuJAd0M1K9ccdCLCGorK2NLhVb/tvga4qLa/J6PNQ1NiZc1jUFAgTbyFcu6V10+Gvovvvu49hjj2XQoEGMGTMGiBwgXrlyJSkpKdxzzz3bNV55eTn//ve/mTlzZrucmJdddlmb13e36sYRX42BYVpAS8259juETctKGgAJkUBLm6YTbj7mEgiFKM7No7RVPnFFUchM9eKw29lcWRc3RlPAH4nebnaIG6aZMDtmistFWXXiqi4Q8WV5XC7p7N4N6LBAZWRksGTJEt544w3efvttjjzySLxeL9dddx3Tpk1LuvxLxjPPPMMRRxzBwIEDpXN8O7Esi6ZAICYXuKaqFOXkkOJyd0q8j9J8rs1qTvYfCodRFIXinFwMy8SyRLRIwYbyjvkfWxBAW9rT/qAEya7ODi/kTzvtNE477bQdGmPp0qWsW7eOGTNmtPueOXPmJL02c+bMHbJnVyPUnAmyNaZlsb6sjEFFxUmXYq3RVBWn3Z703JquaXHPaPD76JObFxPB3ZbzfGux1DWNTK83GpHeQqPPh9fjSRgBDpHkenL2tHvQboHqiNP6rLPOale///3vf9TU1DBt2jRgy4HQGTNmcOqpp3Lqqadu97N3FyzLoqquNun1ippqinPztjmLsttsFGbnxJRoaiE3I4PahvhUIy0HalvHMFlCkJnqjcs6oKkqWWlpcWOnuNw47Q4CoS1n5hp8PvoXFNDg88WFMqS63Ti3s+6eZNel3QK1vaXLFUVpt0BdcsklBFsd6qyurua2227j+uuvp1+/ftv13N0NS4iksx6AQDiMZVntWua1ZNqsqa/HFwxg03Sy0tIIhcOUNyWv9NuaJp8PXdcozsmlrini6HY7naS43JRWVdI3vyCmv03X6ZefT5PfHxW1zFQvDlsk02aDz0dtYwOqqpLtTcPpcMj8S7sR7f5Ld+XZu+ytcmTbmr8h8/Ly5NGZbRA54W9PemrfrttQ2umDChsGq0s242nOCGCaFuvKSslNzyDV7aHB1xT//K3GdtjtbCgvQ1M1vB43dpuNQChEVV0dmqqiJliZ2XSd9NRUUt2RDAqt89pner2kp6RAcyFOye5FuwWqIzMZIQRXX301f/7znykoKNj2DZLtRlVVstPSkta6y83IaNcH2zBNNlVWYFlWXGK20uoq+ublxwmU2xFfstvlcKCpKqYVv0OX03xwNxmJCm4oiiILcezGdOlXkmVZPPTQQ1Rs5QTdFrm5ubz22mvk5uZ2kWW9C7vNRt+8WD+TqigU5eTgsLfPX2OaJv5gkBSXi+KcXPrk5tE3L4+M1NTmwEsjRoxcDgfFeblxAmXTdfoXFMYFUmZ5vdHCCBJJe+nyxXx7CjhKdgxVVUl1exhc5IiGGbTk+G5viIEA8jIzEUJQUlUZ3Y3zejz0yc3DsiwGFBRGqgk3j53ogK+iKLgcDgYWFjWnarHQtUjmSblEk2wv0tvYS1AUBbvNFpdTvL1oqooQgvKaWGd4fVMTYcOgOCcXh93OtgMWIrRk5pRIdgT5lSYBIjPdyrr46G+IZMOUxSkl3YH8ipMAkXCFtmrjBUMhwkaYpkAQl8OOxxnJKiB9SpKuRAqUBIg41dtEUdhcucU3pSoKAwoKccmCrZIuRC7xJEBkiz/Nk/j8ZKT4ZewxFksI1peVJk1aJ5F0Bl06g1IUhXHjxm33wWFJBKs5myRCoCpKl0ZQa6pKXlYWISMcE/QZOXScS1l1ddw9YdOMiJYRRojI31s6xiWdSZe+m1RV5aOPPurKR/RawoZBVV0t1fX1WELgsNkpyMrC5XR22Xa9Xdfpl5dP2DQjyek0DV3X2VBWlrBkVFFOLvVNjVTV1WFaFnbdRl5WJh6nq0MlnUzTJNyO6sSSnovNZuvUwNp2C5RtO6q/QuTbNJjk+IWkbQzDYEN5Gb5AINoWDIdYW1pCv/yC6JGQrkDXdXRdj2ZACIXDhIx40chOT6fB10R905bo8pARZkNZGUXZOaQ3B3i2ByEEpaWl1NXVybi5XRxFUUhLSyM/P79TNlDaLVB//etf5Y7NTiJkGDHi1JqSqkqc9vhI7a5C1zRy0zMoq4ld4nmcTipraxPeU1pdRYrb3W4b6+rqqK2tJScnB4/HI99nuyhCCJqamqioqMDlcpGenr7DY7b7XX7LLbfs8MMk7aMll3ciQuEwlth5MUmqqpLh9WLTdcpqqgkbBk67HdNMboNpWZim2S6BEkJQXl6O1+uNOzQu2fVwuVwEg0HKy8tJS0vb4S+bTvkarqysJLRVyo+GhgaGDBnSGcPvdmhq8jW8ws7PKKlrGumpqXhcrmixg0Q+qda0O9WwaWKapsxa0Yvwer3U19dHqk3v4Ey/w3fX19dz1VVX8corrxBIshwxZS2zDuFxOlEgYQEDb0pKhxzQnUHrGZEQAlVVEwZ3up3OdjtKjeYwBZnjqffQ8rc0DGOH/64d3g665JJLeO+997j55ptRFIWbbrqJp59+mksvvZSsrCzee++9HTJsd8NorknXkqGyT14+TrudvMxMCrOzyfR6cdnt5GVkdkqO8R3F1rzjt/VMSdc0inLisxxsC+l36j105t+yw/L23//+l6effprJkydz7733MmrUKI466ijOP/98+vbty5NPPsnRRx/daYb2VgzTxBfwU15TQ9gwcDSLktNuJ9ObRlVdLYZp4nI4KezAB7+rUBQFt9PJHsV9aAr4CYZCuJ0unHZ7hw8sSyRb0+GvYofDEU1CN3z4cBYtWhS9dvDBB/O///1vh43r7ZiWRXVdHevLygiEQpiWhS8QoMnvp7S6is2VFQTDYUzLotHvY9WmjW2m993ZtGRQyEj1kp+VjdfjkeIk6VQ6LFD7778/3333HRDJV37//ffz1FNPsWDBAmbNmkVxcXGnGdlbMUyT8trY9CaKouC0O2Lii1qzubIiYUFLya6N3+9n4MCBXHPNNd1tSo+iw0u8W265JVq/7oILLuCLL75g+vTpAKSnp/PKK690joW9mFCCqGmHzZY0vzjQPNMye8xST9I5tASoys2CWDr822hdnlxRFJ5++mlmzZpFSUkJQ4cOlefv2kGiDAKi+dxdW8jClb0Pt9vN6tWru9uMHkeHl3gTJkzg119/jWkrLi7moIMOYuPGjfzpT3/aYeN6OzZdjxOjYDiM02FPcgekutwydW4X8uOPPzJx4kRSU1MZMGAAzz33HIsWLUJRFFatWsVpp52G1+slJyeH6667Lhom0cKTTz7Jnnvuicvl4vDDD+fbb7+NuV5aWspFF11Efn4+LpeLYcOGRV0luq7HBET7/X6uueYa8vLySE1N5fTTT6e61aHtjRs3MmnSJNLT0ykqKuKuu+7qul9MN9HhGdSiRYtobIwv5gjwyy+/8Pjjj3P//fd32LDdAV3XKc7NY31ZaUx7XWMjBVnZlFRVxvbXNPKzs2WVky5k0qRJDB48mHnz5rFmzRqKiopirh133HHMmzePBQsWcMcddxAIBHjwwQcBmDt3LjNmzOC+++5jxIgR3HPPPRx99NH88ssvpKWlUVFRwaGHHoqiKDz44IMUFxfz7bff0rdv34S2TJ06lS+++ILHHnsMp9PJlVdeyZQpU6IbUBdccAE1NTW8+eabVFRU9MpQje0SqHnz5vHwww9Hf7700ktJTU2N6ePz+fjmm28YOXJk51jYi1EVBY/LxR7FfahpaCAYDuF2OknzpKCqKm6nk5r6esKmQarbTYrLLXfJupDKykp+/fVXZs+ezbhx4xg3bhxAdIf6kEMO4d577wVg7NixVFdXM3fuXG666SZcLhezZs3i1ltv5YorrgDgwAMPpKCggDfeeINp06Yxa9Ysqqqq+OWXX6I74KNHj05oy7Jly3j99ddZvHgxhx56KABpaWmMHj2aVatWMWjQIL744gvuuOMOJkyY0JW/lm5lu9YKe+yxB+FwmHA4jKIo0f9v/S8jI4PLL7+cf/3rX11lc69CU1Ucdjv5WVn0yc0jJz0Du82Grmm4HA4KsrPpk5dPpjdNilMXk5WVxd57783VV1/NW2+9FZdZYfLkyTE/T5kyhXA4zJdffsnixYspLy/n7LPPjl53u90ceuihUYF7++23+d3vfteuGpH/+c9/GDhwYFScAA4//HAcDkd0vDFjxnDnnXfy3HPP9do0Nds1gxoyZEg0v5Oqqjz55JPsv//+nWLIvHnz+PjjjyktLSU7O5vJkydz+OGHd8rYuwqJIsQVRbrEdxaKojB//nwuueQSTjnlFPbbbz9ef/316HWPxxPTPz8/H9iSjQFIuFw78sgjASgpKaFPnz7tsqW0tJTVq1cnXLaVlkZcAi+//DKXX345F154Ibfeeisvv/wyhxxySLvG31XosA+q9Rks0zQpLy8nb6vikdtDfX095513HkVFRSxYsICHH36Yvn37yniqBBiGgWFZCCHQVFUWL+hE+vTpw7x58/jmm2845ZRTmDp1KnfeeSdAnEN848aNQKTQrN/vByLLwbS0tJh+LQehMzIy2LBhQ7vs8Hq9DB48OEYgWygsLAQi4TzPP/88M2fO5LTTTuPUU09lw4YNveq9sEPbQa+++irjxo3D6XTSp08fVqxYAcDHH3/MCy+8sF1jnXXWWey7775kZmYyefJkHA4H33///Y6Y1+sQQuAPBllTspnfNm5g1aaN/LZpI7WNDfJgdiez//77c/rpp/Pjjz9G2959992YPi+//DIej4dDDz2UQw89FJvNxqZNmxgxYkTMv4EDBwJwzDHH8NZbb0XjB9tizJgxrF+/nrS0tLjxtq64PXjwYC666CI2bdpEXZLSYbsqHZ5BzZkzhxkzZjBt2jQuvvjimLV3eXk5d9xxB+ecc06HxjZNE8Mw4qbUuzthw2BNyeaY2atlWWyqqEDP17s00+buQGlpKbfccgsnn3wyoVCIV155JeooB3j88cdJTU3liCOOYP78+Tz11FPMmjWLlJQUUlJSmDFjBtOnT2f16tWMGTMGVVVZvnw5U6dOxev1cvvtt/PWW28xevRobr75Zvr27cu3337LmWeeSU5OTowtJ554IgcccAATJ07k+uuvZ9iwYfh8PlatWsX06dMJh8NMnz6dk08+GZfLxRNPPMHw4cM7JUlcj0J0kOLiYvHAAw9Ef1YURXz77bdCCCGWLFkinE5nh8ZtamoSjz76qLjiiiuE3+/v0Bg333yzuPnmmzt0b0+mqrZWfL/qt4T/ft2wXoQNo7tN3G78fr/48ccfO/y37kx++eUXMW7cOJGamiqysrLEGWecIUpLS8VHH30kAPHcc8+JiRMnCqfTKfr06SPuueceYVlW9H7LssT9998v9txzT6HrukhNTRVjx44Va9eujfZZvny5OPbYY0VKSopITU0VBxxwgFi5cqUQQghN08TMmTOjfevq6sQf//hHkZ+fLzRNEzk5OeL0008XQghRUVEhjj/+eJGeni7S0tLEcccdJ3799ded84vaBtv6m27P57PDM6jq6moOPPDAhNcqKio6FLJ/88038/PPP+N2u7nhhhtwtlFz7bLLLmtzrKysrO1+fk+nKUneLYgU1hQyn/cOsccee8Qcem/hp59+AmDQoEF88MEHSe9XFIWrrrqKq666Kmmf/fbbL2kqoq19XF6vlzlz5jBnzpy4vtnZ2bzzzjtJn9Nb6LAPavTo0TExUbAlD8wzzzwTcxSmvcyYMYM777yTiRMnMmvWrGiErSRCWxHmNhmCIOmFdHgG9be//Y2xY8ey7777cuaZZwLwz3/+kxUrVrBo0SI+/fTT7R4zMzOTzMxMBg0aRGVlJW+++SbDhw9P2DfRt0oLM2fO3O5n7wp4PSmUV1cnzLSZm54ha9JJeh0dnkHts88+rFixgvHjx/PEE0+g6zrPPfccLpeLL774Iunyr7243W65M7UVNl2nX0Fh3Fm87PR0UtyubrJKIuk6dugr1+v1MmTIEA499FDq6urIy8vjyCOPZJ999tmucfx+P/PmzeOAAw4gLS2N77//no8//phzzz13R8zrdaiKgsfpZFBRMYZpYgkLmx6JOpcHiLuO8ePHS/9eN9FhgVq1ahUTJ05kw4YNDBw4kPz8fD799FNeeOEFZs+ezYIFC6KRttuiZfv0gw8+wOfzUVhYyEUXXcT48eM7al6vpSWLpTz2Itkd6LBAXXrppRiGwZIlS2KWc4sXL2by5MlcddVVvPzyy+0aKysri+uuu66jpkgkkl5Kh9cFn376KQ8//HCcr+mwww7jgQceiIu6lUgkku2lwwKVm5ubNGtmZmYmDoejw0ZJJBIJ7IBAXXbZZcyaNYumrZL7h0Ih7r33Xi644IIdNk4ikezedNgH5XK5qK+vp1+/fhxzzDF4PB4Mw2DhwoVUVVVRXFzMH/7wh2h/RVF44oknOsVoiUSye9BhgbrvvvsASE1N5Ysvvoi2K4pCdnY2CxcujOnfm1JASCSdic/nIxAIkJmZ2d2m9Dg6LFBr1qzpTDskkt2WzMxM/v73v3c4+0dvRp6NkPQ6TCH4qSJATcAgw6kzNMeJ1oNn8ME26iDu7sjwY0mvYvHGRqa/s46bPt7M/UvKuenjzUx/Zx2LNyauQNTZvPrqqwwePBiPx8PYsWMxDINAIMCMGTPIzs4mNTWVc889l8bGRtauXRt1fZx77rkoihLNpvDLL79w0kknkZKSQkZGBlOnTo0pObVw4UL23Xdf3G43+++/PxUVFQA8/fTTDB8+HLfbTf/+/fnHP/6xU153VyEFStJrWLyxkXsXl1Hljz3DWeU3uXdxWZeLVFlZGeeddx5XXXUVK1eu5Prrr0fXda688koWL17MBx98wJdffsm6deu47rrr6Nu3LzU1NQA88cQT1NTUMHr0aMrLyxk7dixOp5MVK1bw0UcfsWzZsuihfNM0mTx5Mv/v//0/fv31V2655ZZowju3283cuXNZuXIlv//977ngggt2aXeMXOJJegWmEDy9vLLNPs8sr+TgIk+XLfdKSkoIhULsvffe9OnThz59+rB582aeeeYZli9fzrBhwwC48847Oe6443j44YejGTDdbnf0/x9//HEUReGFF17Abo+k2Hn++ec54IAD+PTTT9lnn32orq5mzz33pKioKKZ235QpU6L/f+ONNzJ79mz+97//cfHFF3fJa+5q5AxK0iv4qSIQN3Pamkq/yU8VyZP+7Sj77bcfJ598MieccAKXX345paWlfPfddxiGwahRo0hPTyc9PZ3jjz+exsZGysrKEo6zdOlSRo8eHRUniORIT0tL46uvviIjI4MrrriCiy66iHPPPZdVq1ZF+5WWlvKXv/yFAw44gKKiInw+H1VVVV32mrsaKVCSXkFNwNh2p+3o1xEUReHf//43L7/8MosWLeLAAw+MVnv5/PPPWb58OcuXL+e7775jzZo15OXlJRzHNM2E2RMsy4pWlX7ooYdYsGABa9euZf/99+e3336jvr6eQw45hKVLlzJz5kw++ugjCgsLd+lMDHKJJ+kVZDjb91Zub7+OoigKJ598MqNHjyY7OztaFWbTpk0cffTRCe/RdT0m99n+++/PM888QyAQiKa9Xrx4MQ0NDRx00EHRfkcccQQLFy4kPz+f//znP/Tv35/169fzww8/kJKSghBil98hlDMoSa9gaI6TLJfWZp9sl8bQnOR57neU3377jX/9619s3ryZ+fPnA3DCCSdw8sknc/HFFzN//nwqKir46aef+Oqrr6L39evXj/fff5+ysjLC4TBXXXUVoVCI8847j59//pnPPvuMqVOncvTRR3P44YdTU1PDc889x4YNG/j4449paGhg8ODB0fRGb7/9Nps2beKaa67Z5ctQSYGS9Ao0RWHaiOw2+1wwIrtL46HKy8uZOXMmgwYN4sYbb2Tu3LmMGDGCF198kZNOOolzzz2XgoICjjrqKD777LPofffccw8ffPABgwYNoqSkhOzsbL744guqq6sZMWIE55xzDsceeyz/+te/gEhRkkcffZQhQ4YwdepUrr/+ek4++WRGjRrFddddx2WXXcbIkSPJyclhwoQJNDbunBCLrkARu/ICNQktOclnzZrVzZZItkUgEGDNmjUMGDCgzSo+7WXxxkaeXl4Z4zDPdmlcMCKbw4oTZ9+QdC7b+ptuz+dT+qAkvYrDilM4uMizS0WSS5IjBUrS69AUhX1yZRGJ3oD0QUkkkh6LFCiJRNJjkQIlkUh6LFKgJBJJj0UKlEQi6bFIgZJIJD2WHhFmYFkWCxcu5J133qGsrIy0tDTOOOMMWVlYItnN6RECBbBkyRKmTZtGYWEhy5Yt47HHHiM7O5t99tmnu02TSCTdRI9Y4qmqyl//+lf22WcfMjMzOfrooxk6dCiffPJJd5smkfQoDjroIB588MFOGat///7cfvvtnTJWV9EjBCoRKSkp+Hy+7jZDIulRDBw4kMLCwu42Y6fRY5Z4rRFCsGbNGsaOHZu0z2WXXdbmGFlZWZ1t1g5jCUG13yBkCmyaQqZTR1O3/4yYP2xRHzKxBLh1hbQuznG0qyGEoCkQwDANdE3H43TuMnUZQ6FQTCbNrXn11Vd3ojXJ2ZadnUWPnEF9/PHHVFdXM3HixO42pdOoCxi891sd13ywiT/O38CM9zfyxk81253hsawxzJxl5Vz63noufW89Mz8p4ccKP0HD6iLLdy3qmhr5ef061pZsZmN5OWtLNvPz+nXUNXVtypHf//73HHLIITFtzz77LB6Ph8bGRt58802GDRuG0+lk5MiRfPnll9F+/fv355lnnmHs2LHRL+VE1WEABg8ezC233BK9d/PmzZx33nlkZ2fjdrs57LDDsKzIe+Grr75iwoQJuFwu8vLymDFjBoFA8pTH26okM378eGbPns3JJ59M//79d/RX1i563FdvaWkpzz33HJMmTSI7O3l+nzlz5iS91pLOoacQMi3mr6rnlR9rom2+sMUrP9ZQ6TM4f78sPPa2k60BVPoMbvp4MxW+LaK2ri7ETYs2c8/EIgZndl0ytl2BuqZGNiTI822YZqQ9D9I8XZNyZcqUKRx77LFs3LiR4uJiAN58801OOeUUvv32W6ZOncoTTzzBqFGjeOmllzjxxBNZv349LlfkUPMNN9zAPffcw4knnhitDnP//fdz0kknsWLFCnQ9/qMaDAaZMGEChYWFvPPOOxQVFbFq1SpUVeWnn35i/PjxTJs2jeeff57Vq1dz5pln0tTUxFNPPRU3VkslmbFjx7JixQpqa2s5++yzOfPMM3n//fej/WbPns2f/vQnnnzyyS75PW5Nj5pBNTU1cc899zBo0CAmT57c3eZ0GjUBkzdX1ia89uHaBuqCbSf7b2FllT9GnFqwgH9+V0VjqH3j9EaEEJRUtl3VpbSyssvyc0+cOJGcnBzefPNNABobG1mwYAHnnnsud9xxBxdffDFnnXUW/fr14/rrryctLY158+ZF7z/00EOZOnUqmZmZcdVhjjvuuITPfPTRR6mqquLtt9/mkEMOobi4mHHjxgERIRkyZAgPP/wwRUVFjBkzhkceeYRnnnmGdevWxY3VupJM//79GTFiBM8//zwLFizg008/jfYrLCzkuuuuS5pPvbPpMQIVCoW499570XWdq6++GlXtMabtMI0hi5CV+IMhgOptVCNp4avNyTcNfqoKEDB6Xe7BdhPxObX9ewybJk1tLHF2BE3TOP3006NZL9955x0yMjI48sgj+eabb3j00UejVV3S09NZu3Ytv/32W/T+kSNHRv8/UXWYRCxatIixY8eSkhI/K1y6dGmci2TChAlYlsXXX3+dsH9blWQS2bkz6BEqYJomDz74IJWVlcyYMSPi5GxqoqmpqbtN6xTsWtsOWpetfX+GHHfyFXmaQ6MD/vZeg2G2z5fX3n4d4ayzzuKzzz6jvLycf//735x11llomobf7+faa6+NVnVZvnw5v/32G3/84x+j97ZUa4HE1WFaCny2JhwOJ50RJqoM01KYofWz2uoPsZVkkt3blfQIH9TixYtZtmwZAFdddVXMtddee607TOpUvHaNQRkOVtXEV9jIcetkONv3Rx/XN5V/JVkqnrRnOuntHKc3omvteyu3t19HOOyww+jbty9vvPEG77zzTnRpNHToUH7++eftcixvXR1m4cKFnHrqqTF99ttvP5599lkaGxvjZlH7778/CxcujGl7//33UVWVAw44IO557a0ks7PpETOo0aNH89prryX81xtIc2r86ZDcuKojqXaVG0blk+lq34cm261z+UE5cX+0gwrcjOmTgrqdW+m1AYPSxjCljWHCVsd3AWv8BpU+g/p2+tK6Ao/Tib6Nb3ebpuHphLznbTFlyhTuvPPOqB8H4LrrruOVV17h7rvvZv369WzevJl33nkn6RiJqsMMGjQort/VV19NOBzm1FNP5ZtvvmHTpk18+OGHANx000388MMPXHHFFaxevZp3332Xq666igsvvDDqxG/NtirJdBc9Yga1O1CYaufuCcVsrA+xpi5Icaqd/ml2sttYtm2Ny6YyPMfFXROKWFHhxx+22DvHSZ7HRqqj/bMnf9hkU0OYF76v5odKPx6bxjEDU5k4wEuux9buceoCBl+V+Hjjp8hu5IB0B+cNz2RQhgO3befO5hRFoSA7O+EuXgv52dldHg81ZcoU7rrrLq644opo26RJk3j99de56aabuOmmm3C5XBx00EEce+yxCZdMLdVhVq1aRX5+frQ6zNbk5OSwZMkSrrjiCsaMGYPNZmPEiBFMmDCBvfbai08++YQrrrgi6myfPn06N954Y0K7WyrJ/PGPf2TEiBHk5eVx0kknceedd3ba76YjyKouuxDVfoPbPy1hTV2I4lQbdk1hU0MYIeC+I4vpk9a+wLmfqwLc+NEmtvapD8ly8KdD8tolUk0hkxdXVPPeqvq4a38+NI9Diz3tmtF1dlWXuqZGSiorYxzmNk0jPzu7y0IMJLHIqi67KWtqg6ypCwGRnT+bphAyBQJ48Ycqrjgod5szlypfmH98WxUnTgA/VwUpbQy3S6DqgmZCcQL4+/JKhmQ5yHK3fzbWWaR5UvC6PbtsJLkkFilQXUB90KQ2YLK5IUSaUyPHbSPLpVHhM6gJmJQ1hsn26GQ5dfJSEn+Iw6ZJhc+kwmdQFzAp9trJdumMzHFy1OA0AoYgaFrkuHW+KfXxweoG/GHBtjQhaAp+qkq+1f5ViY/hee5tvsYN9aGk12oCJo0hi6xtD9MlKIpCiktWdekNSIHqZKr9Bo8uK+ebUn+0rcCj89fRBdy/pIzVtVs+2AUpNm4YlU+xN3ZpFjZNVtWEuevzEupDW5zXk/dK5/gh6fzty7JozJMCTOifyrQRWbRnkqAAdlVJGpeV0s6QB6fedj99d455kHQaPWIXr7cQMi3e+KkmRpwARvVN4YlvKmPECaCkMcw9X5RS1hiOaS9vMrnt01hxUoC9sl3c/XlpTECmIBKNbopILNS2SHNqjO2b3BdzSJFnm2NARFydSeK79shwbJfTXiJJhpxB7SBh06IxZKGpEDAEH65piOuzR6aTN36qTXj/xoYwvrBJbUAhaFh4bCqra4P4tjr8u1e2k+8r/JhJtjTe/qWWQ4o8GM0zowyXhq05Gt+0BA0hE0VRSHNonDo0nZVVAfqm2RmU4cBvWHyxoYkjB6S2O5Yq06Xzl8PzueOzkhibUu0qlx+ci1cKlKQTkALVQYQQlDUZvPtbHV+X+HDZVC4amZ1w6WQkUxXgpD3TqAtaPPddOeVNYQ4t8iScfaQ5NCoTnMNrodJnUOELM3dZJZYQjO6TwhH9UwCFD9c28PmGRmyqwrGDvBxY4Oa6w/OZ91sdH6xpIMWmcvreGeyV5SS9nalbdFVhYLqd+44s5vMNjZQ1GeyZ5WBEnptcj3xbSToH+U7qIJsbw/zlw034wltmOiWNYVLtKg2h2NmPriloCnGzn8OLPbh1lVmflkTbPlzbwAUj4rM4lDSGGds3hc82JLanf7qD78r8rG92Xr/2Uw1Ds508tLSc6sCWLffHv6lkz0wHxw3yMr/VLtyDS8s5pNDNhSOzyW7H7pvfsPhwTQMvrKhmrywnaU6NBasbePbbKm4ZU8C+7XC0SyTbQvqgOoA/bPHSiuoYcQL4aG0D/2+PtLj+yzY3Mb5falz7+H6pvPFT7BmrhpCFJQQFW+3urasLUey1k2pP/Cc7dWh6jDAeWOBm8aamGHFq4ZfqIEFTxJ3tW7LZlzBbQiLqAiYv/lCNIHJQ+ctNTayvD2EJePTrCqr9XXfmTbL7IGdQrWjJeFkfjIhEmkMj0xWf9bIpbLJkUxPn7pPJyAI3TWELm6qAEJT5DKYMy+Ctn2vxGwKFyFGQaSOzGZHvwqWrBE2BQ1NItaukOzWOGuhlYLqDkCWwqwq/VAf44wE5VAUMnLqKYUX6qwhuGFXA35dXRs/1pTs0Tt87g29LfeyZtWVrfd9cF2/9XJv0tS4r8bFvrouFa7f4zFQiS8XSxhANIQunpuJ1qqQ54t8mG5rFKBFlTQaNIbPdR3gkkmTId1AzIdNiZWWAB5aUU9t8psxtU5k+MpsDC91bBUAq/HVUPr9UB7nuw01Rv1OeR2fGwbl4bQoPHt0HX9jCoSl4nRqNIYtP1zfy1WYfLZ/rfXKc3DSmgGeXV/HyDzXNI8MxA704bQr/+r/a6JJNU+C4QV76pzs4pNDD5KEZmEIQNAT/W1XHhvowe2Rtidq1BG1GcmuKgtXqEIEKXHZQLsvLfDy4tDwqPgPT7VxzWB4FKfat7m/79ykDIyWdgVziNVPRZHDrpyVRcYJI1ssHlpazsT42DCDNrlITNHnlx5oYp3hZk8Ftn5aS47GT67HRP91BQaodwxT849sqlrYSJ4AVFQHmfl3BvnlbZj4C2DvHyR2flUbFCSL+q3m/1VPaFOaHSj93f1HK7MVlPPxVOT9XBzms2MPXJVvyRX1T4uPw4uQhAwcXufm2bEs4xKHFHn6tDrBwbWPMzGh1bYhbPymJW7IVee0kC4UqTrUlXYrujjzyyCPsv//+OzTG119/TXZ2Nps2bWpX/82bN5OdnR2Ty2lXRL6LiGzD/291PaaAwRl2Lj0gm8sOzGFEs3C88mM1/rBJ0LAImRY1QTNp2IDPsPi23I8lBP6whWEJ6oORJWEifq4K0rdVoGaaQyNkCmoS+I4A/reqnnF9Y/1ZOW6dI/qlxjzju3I/++e7KUwQqX5ggRvDIuYZo4pT+HBtAwqwb46LYwd5GVXswaEplDYZlDfFinS6U+OSA3LixrZrClccnNvu3cCuQAgTw7eUcP07GL6lCNG9mUazs7MZOnToDo3h8XgYMmQIbnf7Nh8cDgd77rkn6enpO/Tc7kYu8YCgabGxLshdRxRSEzBZuLYBw4ps1Z+7bybVfoNlm30sXNeAriqcvU8mJVsFV7ZmTU2QlZV+Xvmhlmx3xMfUVjKTxmYfVtgSZLm0NsduCFn08do4IN+N37A4tMhDvzQ7NQGDC0dm8+XGJiwBBxVGfGMXjMiixm+yaH0DNlXhhMFpDMp04A9bnL1PJt+U+kixqeR4dIpSbZyzbxbfl/lZVRMk06Xxp0PyWFbSRGmjwV6tNhcdmsqhRSn0T3Mw79daShsNhmY7OHJAGjndGGYQblhAoOJOhLElq4Gi5+HMuQFb6lHdYtOUKVOYMmVKwmvtrY6y11578fnnn7f7mVlZWXzxxRft7t9TkTMoIh+28/bL5rUfa7h3cRnLSnwsL/MzZ1kFj3xVgaoo3L+0nOVlfpaV+FhR4Se/jQ9h3zQ77/xaz/cVfj5a14hDU2nLI5NiUwk3r6tqAmbMjGprPDaV+qDJyXumcc4+GWQ4VRaubeBvS8r5z8+1FKba6JNm4/3V9dy7uIz3V9czuk8KN4zK59rD8zmw0EPQENy0aDPflPoYlO4gzalRHzA5e59M7ltcylu/1PJ9hZ+P1zdy1xelFKTY6JsWPxNz21QGZji45IAc/jo6n7P2yaIg1dZtx1zCDQvwl8yIEScAYZTjL5lBuGFBlz27raout99+e0yyukRVXPx+P3/84x9JT08nNzeX2267jZNOOil68v+zzz5DURTWrl0LwPnnn88111zDzJkzKSgoIDMzk5tvvjn6jI0bN6IoCosWLYq2LVmyhCOPPBKPx0N6ejpXXnklAG+//TYHH3wwKSkpFBQUcM8993TBb6hjSIECNFWhtDHM/5X5466trQvxS3WQIVmOaNuHaxo4IUE4AYBTV9g318UXrZZbJY0hDipMPDXfI9MRc/C2LmDSN81OepJI7OMGean0Gdz8SQk3LCrhka8qGd03BRUobTJ4b1U97/5Wz8aGyCzstKEZOG0qbpuGU1cJGhYv/1BNdcDkp8pANFjTbwhe+aEGf4I0By+uqMbRxtk7m6bisWsdqvHXWQhhEqi4E0i0tRhpC1Tc1WXLvSlTpvDVV1+xcePGaFtLVZdE3HDDDUybNo13330XgOnTp/Puu+/y1ltvsWTJEr7//nsWLlyIzZY8Ju3pp5+msbGRpUuX8sADD3D77bfzyiuvJOy7cuVKxo8fz7Bhw1i2bBlff/015513HhBJ43vHHXewcuVKbr31Vq677roeU9VbChSR3EYLEhxRaeHzDY0cWLDF4by2LkTIFPxur/QYR3GWS+Om0QX8Z6vt/ceWVXL2Ppnsnx8rUkOyHMw4OJflpVuc20VeG79UBbj84NwY/5GqwMT+qeR6bDH+HQH4QhbXHp4Xc9DXpStcdXAuRamxs7H6kMnnG+JrxNk0+C1BSmKIOOjX1yXPXpCIprBJSUOYdXVBKn0GZhenHTP9X8fNnGIRCKMU0x9fMKAzaKuqSyJaV3HZuHEjL774Io888gjjx49nwIABPP3009H6dsnIyMjgvvvuo0+fPkydOpXRo0fz9ttvJ+z7l7/8hfHjx/PQQw8xdOhQBg0aFE39e8IJJ3DUUUdRXFzMRRddxF577RUVzu5G+qCIfMjNZEE9gGGJuIIEz39fzZS9M3jwqD7UBk3sqoLHrvL0/1XwTVlsOpMmw+KvizZz27hCzhueSWPIwm1TSbWrZLttXH5wLg0hC1/YItOp8XWJj8e/rmDSkHSy3TqGKbBrCl+V+Hjs6woePLoP90woImwJ0p0a6U4Np6Zy/9EO6gImgkh8VIZTR98qHkAIEsYvtfHyo7+D9lLWGOaJ/6tgeakfAXjtKucNz+KQIg8p7aj/1xGEUdGp/baX1lVdrrjiipiqLol20lpXR/nqq6+wLCumknZqaiq5ubltPnPgwIEx4Rz9+/dvswLMHXfckfBafX09Dz74IO+99x7r1q2jvLycqqqqNp+9s5ACBaTYNcb0TeHb8vglHsCBhR5+qIjPoVSQaqPIa6eo+ee6gEFVIPG3XmPIQleVuNQqANluG9mtJlfDTMHT31bx1P/F13nbI8NBulPF64gfJ8dtI2cbx1RS7Coj811xGReq/SaFKTY2J3DQK8CgDEdceyKqfAa3fLKZ0qYtYQn1IYs5yypw6gqj+sRH1HcGih6/o7gj/TrCWWedxdy5c+OquiSidXtLUtutneXbm+xWVdWks65kFWBM0+SII47AMAxuuOEG9tlnH6ZNm9Zl9QO3F7nEa2ZYjot+CVLmZrk0Dixw83+lsTXp+qfZ2Ts7NilamlPnkgNy0BO4Yk4Y7G13pgAFEqZE0VWYvHdG0owG7cFt0zh/eHZcqpR3f6vjvOGZCUtXTRrS/ooxGxpCMeLUmn9+V91lR2A01wEoeh4k3Y5QUPR8NFd8RZPOYuuqLsmWd1szbNgwgJhdt9ra2qSzoY6w33778b///S+uffny5XzzzTe89NJLnHHGGQwbNoxwOPku8s5mt5xBNYVNqnwmn21opCFockiRh/5pdq47PJ/PNjTy0doGTCE4rMjD0QO92DSF0/ZK55MNjWiKwjGDvIwqTiErQcGDAel2/nZUH974qYafKgNkODVOHZrBXlmOdi9vviv3s2emk72ynJHKwwGTIVkOJg7w8q+VNVx6QA4ZO5C+u9Br429HFfOfn2v5ptRPil3l5CHp7Jnh4P6jinntxxp+qYqEGZw2NIMhWc52F0H4pY1sneU+g9COqGsbKIqGM+cG/CUziIhU6+dERMuZcz2K0rVpYBJVddkWQ4YM4fjjj+eSSy7h2WefJT8/n+uuuw5FUTqtgO3tt9/OUUcdxTXXXMP06dNRVZXKykoKCgpQVZV58+aRmZnJSy+9xPfff7/TC3QmY7cTqKawyUdrG3h6+ZY19vzV9QxMs3PD6AImDUljbN8UhBBkunRsWuQNcsawTI4bnIaigNehJT1GYtdU+qbZufTAHPzhyLJue5O35bh17l1cRmGKjdF9U/DYVNbVhbj7i1IUiJz72wE0RaEw1c60kdmcEbLQVCWa7C4DuPygXHzNsVnba3t+G/nM3bqaNPq8M4jEOT2YJA7q+p0SB5Woqkt7eOmll/jDH/7AxIkTo2EGn3/+ecKqwR3hyCOPZP78+Vx77bU88sgjpKenc/7553PPPffw0EMPcfvtt3PXXXdxwQUXMHXqVBoakm8a7Ux6VFWXH3/8kWeffZYNGzYk3S5tD21VjdhQF+KK9xPnLDllSBpnDcuKcyzvbMqbwsx4f0PCLf8T90jj3H0zo8LZ0yhvCnP5/zYknCmdtlc6ZwzLjImT6uyqLhAJOYjs6lWg6DmR5V8Xz5w6m+rqarKzs1mwYEFcCfOeTmdWdekx7/LvvvuOe+65p8tD8z/bGL/F3sL/VjVQF9w+H0l9MFIEoaIpTMjoePHL1mS5dW4ZWxiXH3z/fBeThqT3WHGCSKbNW8YW4N5qqnRwoZvjB6ftlCBORdHQ3Qdj856A7j54lxCnjz76iI8++ojNmzfz3XffMWXKFPbee2/Gjx/f3aZ1Kz1miTdw4EAefPBBvv32W77//vsue05b1W/9hpUwzC8RQcNiTW2Qvy+vYlVNEJuqMK5fCqfvnbHNnbRtoSkKgzId3H90MSWNYeoDJn28djJceo9PpaurCntmOnnwmGI2N4RpCJr0TXOQ7tR6vO3dyfr167nzzjtZv349WVlZjBs3jmeffTbpLuDuQo8RqM5aa2+LQ4o8MZkkWzM8L5KvqT2srwvx1482R8/YhS3BB2sa+KkywKxxhWTtYC4kTVHaFTbQE9HUXdf27mLq1KlMnTq1u83ocfTctUIX0ddrZ1B6fDiBrsDU4Vl42rHT1hA0efbbqoQHgDc1hFlbmzgiWyKRbB89Zga1vVx22WVtXs/KykrYnunSuW5UAe/9Vsv81Q0Ewhb75rmYOjyLotT2feMHTIuVbWynLyvxcUBB+8o3SSSS5OyyArUjZLt1puyTxfF7pCOEwKWr7Zo5taAqCh6bSmM4sVM8s51BjZIt9KDNZMkO0pl/y11WoObMmZP0Wss2ZlvoqtJhP1G6Q+P4PdJ47ceahNcPK945/rTeQMtpfZ/Ph0uWK+8V+HyRUxdtZWJoL7usQHUnmqpwzEAvK8p9/Fi5xd+kAJcdmJMwwlySGE3TSE9Pp7y8HAC32y3zme+iCCHw+XyUl5eTnp7eKTuQ8pPUQTJdOn8+LJ+yxjDLy/ykOjRG5LnIdOk4uzJcuheSn58PEBUpya5Nenp69G+6o0iB2gHSnTrpTp0h2XJpsiMoikJBQQG5ubk96qCqZPux2WydGrvV4wRq/Pjxu3307O6Kpmm7fWCiJBa5FpFIJD0WKVASiaTHIgVKIpH0WHqcD6ozqKurw7KsdsVDSSSSnUtNTU27E/H1yhmUruudlomwu6mqquoxCey7Gvlaex+JXqeqquh6++ZGPSphnSSeljOHbUXO9xbka+197Ojr7B3TDIlE0iuRAiWRSHosUqAkEkmPRQqURCLpsUiBkkgkPRYpUBKJpMciwwwkEkmPRc6gJBJJj0UKlEQi6bFIgZJIJD0WKVASiaTHIgVKIpH0WHplupXeQmlpKVdccUVM2+GHH86MGTO6x6CdwGeffcaCBQtYtWoV11xzDSNGjOhukzqV8vLypEVn586dS3Z29k62qOsIBAK89NJLLFmyhHA4zPDhw7ngggvwer3tHkOGGfRgvv/+e5577jluvfXWaJuu6zgcjm60qmuwLIuHH36YFStWcNZZZzFy5Ei8Xm+vy1FuWRZ+vz+m7b///S9Lly7l/vvv7yaruobHH3+cTZs2cfHFFxMKhXjooYcoLi7mmmuuafcYcgbVg6moqCA3NxePp/eXUX/nnXf4/vvvue222ygsLOxuc7oMVVVj/p6mafLxxx8zadKk7jOqi1i6dCmXXnopRUVFABx33HG8+OKL2zWG9EH1YCorK8nJyeluM7qccDjMW2+9xamnntqrxSkRS5YswefzMW7cuO42pdNxOp2UlJREf66oqCAvL2+7xpAzqB5MeXk5S5Ys4csvv6S4uJgTTzyRkSNHdrdZnc7KlStpaGjA7XZz/fXXU1tby7Bhw5g6dSqpqandbV6X8t577zF+/HicTmd3m9LpTJkyhSeffJLq6mo8Hg+ffvopV1999XaNIWdQPZjTTjuNe+65hyuvvJK0tDTuvvtuli9f3t1mdTobN25E0zQ++eQTLrzwQi655BJ+/fVXHnrooe42rUtZvXo1v/zyC8ccc0x3m9Il7LvvvvTt25evvvqKf//73+Tn55Oenr5dY0iB6sHk5+dTWFjIsGHDuPLKK9lzzz2ZN29ed5vV6fj9fnRd5+qrr2bQoEEMHz6c8847j++++47q6uruNq/LePfddxk+fHivXNYahsGsWbMYOHAgDz/8MI899hh2u51Zs2YRDAbbPY4UqF2IPfbYo1d+YFNSUnA4HDHO45YPbU1NTXeZ1aXU1tbyxRdfcNxxx3W3KV3CihUr2LRpE1OmTEFVVbxeLxdffDEVFRX88MMP7R5HCtQuxLp16yguLu5uMzqdIUOGUF9fH+NQLSkpQVVVCgoKutGyrmPBggVkZWX1ujivFkzTBCAUCkXbWkJGtqfikhSoHkogEOCdd95hw4YNlJeX88ILL7By5UrOOOOM7jat0+nXrx8jR45k7ty5rFu3jrVr1/LCCy8wYcIE3G53d5vX6RiGwYIFCzjmmGN6TXm0rdlrr73IyMiIxkJt3ryZJ598kr59+7LPPvu0exwZqNlDqays5NFHH2X9+vUEAgEGDx7M+eefz4ABA7rbtC7B5/Px3HPPsWTJEhRFYfz48Zxzzjntrp+2K/HJJ5/w1FNP8fjjj/fqGLeNGzfywgsv8Msvv6CqKsOHD+ecc84hMzOz3WNIgZJIJD2W3jm/lEgkvQIpUBKJpMciBUoikfRYpEBJJJIeixQoiUTSY5ECJZFIeixSoCQSSY9FCpRkp7No0SIUReGzzz5r9z39+/fnwgsv7EKrto/x48dz5JFHdrcZvR4pUBLJNvj666/58MMPu9uM3RIpUBLJNjj11FN5+eWXu9uM3RIpUBKJpMciBWo35ccff2TixImkpqYyYMAAnnvuOQCqq6uZNm0amZmZpKenc8kllxAIBKL39e/fnzlz5jB79mz69euHy+Xi6KOP5rfffov22bBhA+eddx59+/bF5XIxZswYvv/++06137Is7rzzTvr27Yvb7eboo49m3bp10evnn38+Z511Fm+++Sb77rsvKSkpHHvssWzatClmnG+//ZaxY8ficrkYNmwY77zzDgcccAA333wza9euRVEU1q1bx9NPP42iKJx//vkx97/55psMGzaMtLQ0zjzzTOrr6zv1de7u9L6j4pJ2MWnSJAYPHsy8efNYs2YNRUVFCCE47rjjaGpq4qWXXqKpqYmLL74Yy7J44oknovc+/vjjpKam8vjjj9PU1MSVV17JhAkTWLFiBV6vF7/fT//+/aMf6htvvJFTTjkleqq9M7juuut44okneOSRRygqKuKGG27guOOOY8WKFdFnfPbZZ/z444/cfffdVFVVMWPGDM4991wWLlwIRPJrjR8/nr322ou3336biooKLrvsMurq6rDb7RQWFvLVV19x0kkncfDBB3PjjTfG1K37/vvvmTNnDg899BA//PADf/nLX0hNTeWpp57qlNcoAYRkt6OiokIA4q233oppf/3114XNZhPr1q2Ltr3wwgvC5XKJhoYGIYQQ/fr1ExkZGaKuri7aZ+nSpQIQDz30UMLnLV68WADiiy++EEII8dFHHwlAfPrpp+22uV+/fmLatGlCCCE2btwoHA6HePnll6PX169fLxRFER988IEQQoipU6cKVVXFqlWron1mz54tALFx40YhhBAXXXSRyMjIEPX19dE+8+fPF4C47bbbEj67hXHjxom0tLSY38MFF1wgUlJShGEY7X5dkraRS7zdkKysLPbee2+uvvpq3nrrLURzxp3//Oc/jBkzhr59+0b7TpgwAb/fz5IlS6JtxxxzTEx12IMOOohBgwbx6aefRtsWLlzIKaecQmFhIWPGjAEiS7/OYP78+UDEed1Cnz592GOPPVi0aFG0raioiIEDB0Z/bsle2bIUnD9/PieffHJM5ZjtCR3Yf//9Y34PBxxwAI2NjVRVVW3X65EkRwrUboiiKMyfP5+99tqLU045hZEjR/Lrr79SWlrKwoULURQl+q8lN3hpaWn0/kRJ1vLz86mrqwPgn//8J0ceeSROp5M5c+bw+uuvA7HpX3eE0tJSgsEgdrs9xtZffvklxk673R5zX0tppxY7SkpK4lIob08l462Xqy3ZPzvrdUqkD2q3pU+fPsybN49vvvmGU045halTp1JQUMDYsWMTlnvq379/9P8Nw4i7vnHjRkaPHg3ArFmzOP3006Nb82vWrOlU271eL263m88//zzuWmsfUXvGqaysjGmT4tKzkAK1m7P//vtz+umn89RTT3H66adz11130a9fPzIyMpLes3DhQgKBQHRGsnjxYtatW8ctt9wCRGYmrVMTf/PNN51q85gxY/D5fPh8Pg4//PAOjzNu3Dj+85//8MADD0RfS6KAzJSUFHw+X4efI+k4UqB2Q0pLS7nllls4+eSTCYVCvPLKK4wbN45p06YxZ84cjjjiCP785z8zaNAgampqqK6u5uyzz47eX1VVxYknnsi1115LTU0NM2bMYOjQoUyZMgWAww47jOeff57Ro0dTX1/Prbfe2qn2jxgxgsmTJzNp0iRuuOEGDjzwQMLhMMuXL+eqq65q9zg33XQTBx98MCeccAJ//etfKSkpYfbs2UBkGdzC3nvvzYcffsiCBQsYNGhQjF9L0rVIH9RuSENDQ7RCzLRp0xg1ahRPPvkkqampfPrppwwfPpwrrriC0aNHc+GFF7Js2bKY+08//XRGjhzJlClTuOCCCxg1ahQffPABDocDgKeeeoqBAwcyefJkHnjgAZ5//nlSUlJoaGjotNfwwgsvMH36dB544AHGjx/Pqaeeyvvvv79dzxg5ciTvv/8+paWlHH/88Tz88MM8/fTTALhcrmi/W265hZycHE488UT+85//dNprkGwbWTRBsl3079+fI488kr///e+dMp5lWViWlfS6oijb5bjeUSoqKsjNzeXll1/mzDPP3GnPlSRGzqAk3cqtt96KzWZL+u+AAw7Yqfa88sorqKrKqFGjdupzJYmRPihJt3LxxRczadKkpNdbL7U6m8rKSs477zzOPfdcioqKWLJkCbfccgvnnXceffr06bLnStqPFChJt5Kfn09+fn63PDscDuN0Orn66qupra1lwIABXHPNNdx4443dYo8kHumDkkgkPRbpg5JIJD0WKVASiaTHIgVKIpH0WKRASSSSHosUKIlE0mORAiWRSHosUqAkEkmPRQqURCLpsUiBkkgkPZb/Dwzh+22OY/ncAAAAAElFTkSuQmCC" />
    


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
    WARNING:root:zeros found, replaced with min 0.001
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
    WARNING:root:zeros found, replaced with min 0.001
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
to_zip(p, outp=None, fmt='zip')
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
get_version(suffix='')
```

Get the time-based version string. 



**Parameters:**
 
 - <b>`suffix`</b> (string):  suffix. 



**Returns:**
 
 - <b>`version`</b> (string):  version. 


---

## <kbd>function</kbd> `version`

```python
version(p, outd=None, **kws)
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
    p,
    outd,
    versioned=False,
    suffix='',
    zipped=False,
    move_only=False,
    test=True,
    no_test=False
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
 - <b>`no_test`</b> (bool):  no testing (False).  



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

## <kbd>function</kbd> `read_yaml`

```python
read_yaml(p)
```

Read `.yaml` file. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

## <kbd>function</kbd> `to_yaml`

```python
to_yaml(d, p, **kws)
```

Save `.yaml` file. 



**Parameters:**
 
 - <b>`d`</b> (dict):  input dictionary. 
 - <b>`p`</b> (str):  path. 

Keyword Arguments: 
 - <b>`kws`</b> (d):  parameters provided to `yaml.safe_dump`. 



**Returns:**
 
 - <b>`p`</b> (str):  path. 


---

## <kbd>function</kbd> `read_json`

```python
read_json(path_to_file, encoding=None)
```

Read `.json` file. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


---

## <kbd>function</kbd> `to_json`

```python
to_json(data, p)
```

Save `.json` file. 



**Parameters:**
 
 - <b>`d`</b> (dict):  input dictionary. 
 - <b>`p`</b> (str):  path.  



**Returns:**
 
 - <b>`p`</b> (str):  path. 


---

## <kbd>function</kbd> `read_pickle`

```python
read_pickle(p)
```

Read `.pickle` file. 



**Parameters:**
 
 - <b>`p`</b> (str):  path. 



**Returns:**
 
 - <b>`d`</b> (dict):  output dictionary. 


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
post_read_table(df1, clean, tables, verbose=True, **kws_clean)
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
    p,
    ext=None,
    clean=True,
    filterby_time=None,
    params={},
    kws_clean={},
    kws_cloud={},
    check_paths=True,
    tables=1,
    test=False,
    verbose=True,
    **kws_read_tables
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
get_logp(ps)
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
    ps,
    func,
    replaces_outp=None,
    replaces_index=None,
    drop_index=True,
    colindex='path',
    filter_rows=None,
    fast=False,
    progress_bar=True,
    params={},
    dbug=False,
    test1=False,
    verbose=True,
    kws_read_table={},
    **kws
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
    ps,
    fast=False,
    filterby_time=None,
    to_dict=False,
    params={},
    tables=None,
    **kws_apply_on_paths
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
to_table(df, p, colgroupby=None, test=False, **kws)
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
to_manytables(df, p, colgroupby, fmt='', ignore=False, **kws_get_chunks)
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
to_table_pqt(df, p, engine='fastparquet', compression='gzip', **kws_pqt)
```






---

## <kbd>function</kbd> `tsv2pqt`

```python
tsv2pqt(p)
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

# <kbd>module</kbd> `roux.query.biomart`
For querying BioMart database. 

**Global Variables**
---------------
- **release2prefix**

---

## <kbd>function</kbd> `get_ensembl_dataset_name`

```python
get_ensembl_dataset_name(x: str) → str
```

Get the name of the Ensembl dataset. 



**Args:**
 
 - <b>`x`</b> (str):  species name. 



**Returns:**
 
 - <b>`str`</b>:  output. 


---

## <kbd>function</kbd> `query`

```python
query(
    species: str,
    release: int,
    attributes: list = None,
    filters: list = None,
    databasep: str = 'external/biomart/',
    dataset_name: str = None,
    force: bool = False,
    **kws_query
) → DataFrame
```

Query the biomart database. 



**Args:**
 
 - <b>`species`</b> (str):  species name. 
 - <b>`release`</b> (int):  Ensembl release.  
 - <b>`attributes`</b> (list, optional):  list of attributes. Defaults to None. 
 - <b>`filters`</b> (list, optional):  list of filters. Defaults to None. 
 - <b>`databasep`</b> (str, optional):  path to the local database folder. Defaults to 'data/database'. 
 - <b>`dataset_name`</b> (str, optional):  dataset name. Defaults to None. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output 



**Examples:**
 1. Setting filters for the human data:  filters={  # REMOVE: mitochondria.py# REMOVE: non protein coding  'biotype':['protein_coding'],  } 


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.query.ensembl`
For querying Ensembl databases. 

**Global Variables**
---------------
- **release2prefix**

---

## <kbd>function</kbd> `to_gene_name`

```python
to_gene_name(k: str, ensembl: object) → str
```

Gene id to gene name. 



**Args:**
 
 - <b>`k`</b> (str):  gene id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  gene name. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_protein_id`

```python
to_protein_id(k: str, ensembl: object) → str
```

Transcript id to protein id. 



**Args:**
 
 - <b>`x`</b> (str):  transcript id. 
 - <b>`ensembl`</b> (str):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  protein id. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_gene_id`

```python
to_gene_id(k: str, ensembl: object) → str
```

Transcript id to gene id. 



**Args:**
 
 - <b>`k`</b> (str):  transcript id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  gene id. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_transcript_id`

```python
to_transcript_id(k: str, ensembl: object) → str
```

Protein id to transcript id. 



**Args:**
 
 - <b>`k`</b> (str):  protein id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  transcript id. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_dnaseq`

```python
to_dnaseq(k: str, ensembl: object) → str
```

Gene id to DNA sequence. 



**Args:**
 
 - <b>`k`</b> (str):  gene id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  DNA sequence. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_protein_id_longest`

```python
to_protein_id_longest(k: str, ensembl: object) → str
```

Gene id to protein id of the longest protein. 



**Args:**
 
 - <b>`k`</b> (str):  gene id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  protein id. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_protein_seq`

```python
to_protein_seq(k: str, ensembl: object, transcript: bool = False) → str
```

Protein/transcript id to protein sequence. 



**Args:**
 
 - <b>`k`</b> (str):  protein id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  protein sequence. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `to_cdsseq`

```python
to_cdsseq(k: str, ensembl: object) → str
```

Transcript id to coding sequence (CDS). 



**Args:**
 
 - <b>`k`</b> (str):  transcript id. 
 - <b>`ensembl`</b> (object):  ensembl object. 



**Returns:**
 
 - <b>`str`</b>:  CDS sequence. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `get_utr_sequence`

```python
get_utr_sequence(k: str, ensembl: object, loc: str = 'five') → str
```

Protein id to UTR sequence. 



**Args:**
 
 - <b>`k`</b> (str):  transcript id. 
 - <b>`ensembl`</b> (object):  ensembl object. 
 - <b>`loc`</b> (str):  location of the UTR. 



**Returns:**
 
 - <b>`str`</b>:  UTR sequence. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `is_protein_coding`

```python
is_protein_coding(k: str, ensembl: object, geneid: bool = True) → bool
```

A gene or protein is protein coding or not. 



**Args:**
 
 - <b>`k`</b> (str):  protein/gene id. 
 - <b>`ensembl`</b> (object):  ensembl object. 
 - <b>`geneid`</b> (bool):  if gene id is provided. 



**Returns:**
 
 - <b>`bool`</b>:  is protein coding. 



**Notes:**

> 1. `ensembl` object. from pyensembl import EnsemblRelease ensembl EnsemblRelease(release=100) 


---

## <kbd>function</kbd> `rest`

```python
rest(
    ids: list,
    function: str = 'lookup',
    target_taxon: str = '9606',
    release: str = '100',
    format_: str = 'full',
    test: bool = False,
    **kws
)
```

Query Ensembl database using REST API. 



**Args:**
 
 - <b>`ids`</b> (list):  ids. 
 - <b>`function`</b> (str, optional):  query function. Defaults to 'lookup'. 
 - <b>`target_taxon`</b> (str, optional):  taxonomic id of the species. Defaults to '9606'. 
 - <b>`release`</b> (str, optional):  ensembl release. Defaults to '100'. 
 - <b>`format_`</b> (str, optional):  format of the output. Defaults to 'full'. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 

Keyword Args: 
 - <b>`kws`</b>:  additional queries. 



**Raises:**
 
 - <b>`ValueError`</b>:  ids should be str or list. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

## <kbd>function</kbd> `to_homology`

```python
to_homology(
    x: str,
    release: int = 100,
    homologytype: str = 'orthologues',
    outd: str = 'data/database',
    force: bool = False
) → dict
```

Query homology of a gene using Ensembl REST API.  



**Args:**
 
 - <b>`x`</b> (str):  gene id. 
 - <b>`release`</b> (int, optional):  Ensembl release number. Defaults to 100. 
 - <b>`homologytype`</b> (str, optional):  type of the homology. Defaults to 'orthologues'. 
 - <b>`outd`</b> (str, optional):  path of the output folder. Defaults to 'data/database'. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  output. 

References: 
 - <b>`1. Documentation`</b>:  https://e{release}.rest.ensembl.org/documentation/info/homology_ensemblgene 


---

## <kbd>function</kbd> `to_domains`

```python
to_domains(
    x: str,
    release: int,
    species: str = 'homo_sapiens',
    outd: str = 'data/database',
    force: bool = False
) → DataFrame
```

Protein id to domains.  



**Args:**
 
 - <b>`x`</b> (str):  protein id. 
 - <b>`release`</b> (int):  Ensembl release. 
 - <b>`species`</b> (str, optional):  species name. Defaults to 'homo_sapiens'. 
 - <b>`outd`</b> (str, optional):  path of the output directory. Defaults to 'data/database'. 
 - <b>`force`</b> (bool, optional):  overwrite output. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


---

## <kbd>function</kbd> `to_species_name`

```python
to_species_name(k: str) → str
```

Convert to species name. 



**Args:**
 
 - <b>`k`</b> (_type_):  taxonomic id. 



**Returns:**
 
 - <b>`str`</b>:  species name. 


---

## <kbd>function</kbd> `to_taxid`

```python
to_taxid(k: str) → str
```

Convert to taxonomic ids.   



**Args:**
 
 - <b>`k`</b> (str):  species name. 



**Returns:**
 
 - <b>`str`</b>:  taxonomic id. 


---

## <kbd>function</kbd> `convert_coords_human_assemblies`

```python
convert_coords_human_assemblies(
    release: int,
    chrom: str,
    start: int,
    end: int,
    frm: int = 38,
    to: int = 37,
    test: bool = False,
    force: bool = False
) → dict
```

Convert coordinates between human assemblies. 



**Args:**
 
 - <b>`release`</b> (int):  Ensembl release. 
 - <b>`chrom`</b> (str):  chromosome name. 
 - <b>`start`</b> (int):  start position. 
 - <b>`end`</b> (int):  end position. 
 - <b>`frm`</b> (int, optional):  assembly to convert from. Defaults to 38. 
 - <b>`to`</b> (int, optional):  assembly to convert to. Defaults to 37. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`force`</b> (bool, optional):  overwrite outputs. Defaults to False. 



**Returns:**
 
 - <b>`dict`</b>:  output. 


---

## <kbd>function</kbd> `map_id`

```python
map_id(
    df1: DataFrame,
    gene_id: str,
    release: str,
    release_to: str,
    out: str = 'df',
    test: bool = False
) → DataFrame
```

Map ids between releases. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input dataframe. 
 - <b>`gene_id`</b> (str):  gene id. 
 - <b>`release`</b> (str):  release to convert from. 
 - <b>`release_to`</b> (str):  release to convert to. 
 - <b>`out`</b> (str, optional):  output type. Defaults to 'df'. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 

Notes:      
 - <b>`1. m`</b>: m mappings are possible. e.g. https://useast.ensembl.org/Homo_sapiens/Gene/Idhistory?db=core;g=ENSG00000276410;r=6:26043227-26043713;t=ENST00000615966 


---

## <kbd>function</kbd> `read_idmapper_output`

```python
read_idmapper_output(outp: str) → DataFrame
```

Read the output of Ensembl's idmapper. 



**Args:**
 
 - <b>`outp`</b> (str):  path to the file. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


---

## <kbd>function</kbd> `map_ids_`

```python
map_ids_(ids: list, df00: DataFrame, release: int, release_to: int) → DataFrame
```

Function for mapping many ids. 



**Args:**
 
 - <b>`ids`</b> (list):  list of ids. 
 - <b>`df00`</b> (pd.DataFrame):  source dataframe. 
 - <b>`release`</b> (str):  release to convert from. 
 - <b>`release_to`</b> (str):  release to convert to. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 


---

## <kbd>function</kbd> `map_ids`

```python
map_ids(
    srcp: str,
    dbp: str,
    ids: list,
    release: int = 75,
    release_to: int = 100,
    species: str = 'human',
    test: bool = False
) → DataFrame
```

Map many ids between Ensembl releases. 



**Args:**
 
 - <b>`srcp`</b> (str):  path to the IDmapper.pl file. 
 - <b>`dbp`</b> (str):  path to the database. 
 - <b>`ids`</b> (list):  list of ids. 
 - <b>`release`</b> (str):  release to convert from. 
 - <b>`release_to`</b> (str):  release to convert to. 
 - <b>`species`</b> (str, optional):  species name. Defaults to 'human'. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  output. 



**Examples:**
 srcp='deps/ensembl-tools/scripts/id_history_converter/IDmapper.pl', dbp='data/database/ensembl_id_history_converter/db.pqt', ids=ensembl.gene_ids(),     


<!-- markdownlint-disable -->

# <kbd>module</kbd> `roux.query`






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

## <kbd>function</kbd> `get_corr_bootstrapped`

```python
get_corr_bootstrapped(
    x: <built-in function array>,
    y: <built-in function array>,
    method='spearman',
    ci_type='max',
    cv: int = 5,
    random_state=1,
    verbose=False
) → tuple
```

Get correlations after bootstraping. 



**Args:**
 
 - <b>`x`</b> (np.array):  x vector. 
 - <b>`y`</b> (np.array):  y vector. 
 - <b>`method`</b> (str, optional):  method name. Defaults to 'spearman'. 
 - <b>`ci_type`</b> (str, optional):  confidence interval type. Defaults to 'max'. 
 - <b>`cv`</b> (int, optional):  number of bootstraps. Defaults to 5. 
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
    fmt='<',
    n=True,
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
    bootstrapped=False,
    ci_type='max',
    magnitide=True,
    outstr=False,
    **kws
)
```

Correlation between vectors (wrapper). 



**Args:**
 
 - <b>`x`</b> (np.array):  x. 
 - <b>`y`</b> (np.array):  y. 
 - <b>`method`</b> (str, optional):  method name. Defaults to 'spearman'. 
 - <b>`bootstrapped`</b> (bool, optional):  bootstraping. Defaults to False. 
 - <b>`ci_type`</b> (str, optional):  confidence interval type. Defaults to 'max'. 
 - <b>`magnitide`</b> (bool, optional):  show magnitude. Defaults to True. 
 - <b>`outstr`</b> (bool, optional):  output as string. Defaults to False. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `get_corr_bootstrapped` function. 


---

## <kbd>function</kbd> `get_corrs`

```python
get_corrs(
    df1: DataFrame,
    method: str,
    cols: list,
    cols_with: list = [],
    coff_inflation_min: float = None,
    test: bool = False,
    **kws
)
```

Correlate columns of a dataframes. 



**Args:**
 
 - <b>`df1`</b> (DataFrame):  input dataframe. 
 - <b>`method`</b> (str):  method of correlation `spearman` or `pearson`.         
 - <b>`cols`</b> (str):  columns. 
 - <b>`cols_with`</b> (str):  columns to correlate with i.e. variable2. 

Keyword arguments: 
 - <b>`kws`</b>:  parameters provided to `get_corr` function. 



**Returns:**
 
 - <b>`DataFrame`</b>:  output dataframe. 

TODOs: 0. use `lib.set.get_pairs` to get the combinations. 1. Provide 2D array to `scipy.stats.spearmanr`? 2. Add parallel processing through `fast` parameter. 


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
    stats=[<function mean at 0x7fb280212d40>, <function median at 0x7fb2800ccd40>, <function var at 0x7fb280216200>, <built-in function len>],
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
    stats=[<function mean at 0x7fb280212d40>, <function median at 0x7fb2800ccd40>, <function var at 0x7fb280216200>, <built-in function len>],
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
check_covariates(df1, covariates, colindex, plot: bool = False)
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
    show_n_ha='left',
    alternative: str = 'two-sided',
    offx_n: float = 0,
    xlim: tuple = None,
    xscale: str = 'linear',
    offx_pval: float = 0.05,
    offy_pval: float = None,
    saturate_color_alpha: float = 1.5,
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
 - <b>`xlim`</b> (tuple, optional):  x-axis limits. Defaults to None. 
 - <b>`offx_pval`</b> (float, optional):  x-offset for the p-value labels. Defaults to 0.05. 
 - <b>`offy_pval`</b> (float, optional):  y-offset for the p-value labels. Defaults to None. 
 - <b>`saturate_color_alpha`</b> (float, optional):  saturation of the color. Defaults to 1.5. 
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

## <kbd>function</kbd> `get_lines`

```python
get_lines(
    logp: str = 'log_notebook.log',
    sep: str = '# plot',
    test: bool = False
) → list
```

Get lines from the log. 



**Args:**
 
 - <b>`logp`</b> (str, optional):  path to the log file. Defaults to 'log_notebook.log'. 
 - <b>`sep`<.py# plot'. 
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
    df1: DataFrame = None,
    kws_plot: dict = {},
    logp: str = 'log_notebook.log',
    sep: str = '# plot',
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
 - <b>`kws_plot`</b> (dict, optional):  parameters for plotting. Defaults to dict(). 
 - <b>`logp`</b> (str, optional):  path to the log. Defaults to 'log_notebook.log'. 
 - <b>`sep`<.py# plot'. 
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
    bootstrapped: bool = False,
    cmap: str = 'Reds',
    label_colorbar: str = None,
    gridsize: int = 25,
    bbox_to_anchor: list = [1, 1],
    loc: str = 'upper left',
    title: str = None,
    params_plot: dict = {},
    params_plot_trendline: dict = {},
    params_set_label: dict = {},
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
 - <b>`bootstrapped`</b> (bool, optional):  bootstrap data. Defaults to False. 
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
    p_min: float = 0.01,
    ax=None,
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

# <kbd>module</kbd> `roux.viz.sequence`
For plotting sequences. 


---

## <kbd>function</kbd> `plot_domain`

```python
plot_domain(
    d1: dict,
    x: float = 1,
    xoff: float = 0,
    y: float = 1,
    height: float = 0.8,
    ax: Axes = None,
    **kws
) → Axes
```

Plot protein domain. 



**Args:**
 
 - <b>`d1`</b> (dict):  plotting data including intervals. 
 - <b>`x`</b> (float, optional):  x position. Defaults to 1. 
 - <b>`xoff`</b> (float, optional):  x-offset. Defaults to 0. 
 - <b>`y`</b> (float, optional):  y position. Defaults to 1. 
 - <b>`height`</b> (float, optional):  height. Defaults to 0.8. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `plot_protein`

```python
plot_protein(
    df: DataFrame,
    ax: Axes = None,
    label: str = None,
    alignby: str = None,
    test: bool = False,
    **kws
) → Axes
```

Plot protein. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input data. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`label`</b> (str, optional):  proein name. Defaults to None. 
 - <b>`alignby`</b> (str, optional):  align proteins by this domain. Defaults to None. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `plot_gene`

```python
plot_gene(
    df1: DataFrame,
    label: str = None,
    kws_plot: dict = {},
    test: bool = False,
    outd: str = None,
    ax: Axes = None,
    off_figw: float = 1,
    off_figh: float = 1
) → Axes
```

Plot genes. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`label`</b> (str, optional):  label to show. Defaults to None. 
 - <b>`kws_plot`</b> (dict, optional):  parameters provided to the `plot` function. Defaults to {}. 
 - <b>`test`</b> (bool, optional):  test mode. Defaults to False. 
 - <b>`outd`</b> (str, optional):  output directory. Defaults to None. 
 - <b>`ax`</b> (plt.Axes, optional):  `plt.Axes` object. Defaults to None. 
 - <b>`off_figw`</b> (float, optional):  width offset. Defaults to 1. 
 - <b>`off_figh`</b> (float, optional):  height offset. Defaults to 1. 



**Returns:**
 
 - <b>`plt.Axes`</b>:  `plt.Axes` object. 


---

## <kbd>function</kbd> `plot_genes_legend`

```python
plot_genes_legend(df: DataFrame, d1: dict)
```

Make the legends for the genes. 



**Args:**
 
 - <b>`df`</b> (pd.DataFrame):  input data. 
 - <b>`d1`</b> (dict):  plotting data. 


---

## <kbd>function</kbd> `plot_genes_data`

```python
plot_genes_data(
    df1: DataFrame,
    release: int,
    species: str,
    custom: bool = False,
    colsort: str = None,
    cmap: str = 'Spectral',
    fast: bool = False
) → tuple
```

Plot gene-wise data. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`release`</b> (int):  Ensembl release. 
 - <b>`species`</b> (str):  species name. 
 - <b>`custom`</b> (bool, optional):  customised. Defaults to False. 
 - <b>`colsort`</b> (str, optional):  column to sort by. Defaults to None. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Spectral'. 
 - <b>`fast`</b> (bool, optional):  parallel processing. Defaults to False. 



**Returns:**
 
 - <b>`tuple`</b>:  (dataframe, dictionary) 


---

## <kbd>function</kbd> `plot_genes`

```python
plot_genes(
    df1,
    custom=False,
    colsort=None,
    release=100,
    cmap='Spectral',
    **kws_plot_gene
)
```

Plot many genes. 



**Args:**
 
 - <b>`df1`</b> (pd.DataFrame):  input data. 
 - <b>`release`</b> (int):  Ensembl release. 
 - <b>`custom`</b> (bool, optional):  customised. Defaults to False. 
 - <b>`colsort`</b> (str, optional):  column to sort by. Defaults to None. 
 - <b>`cmap`</b> (str, optional):  colormap. Defaults to 'Spectral'. 

Keyword Args: 
 - <b>`kws_plot_gene`</b>:  parameters provided to the `plot_genes_data` function.  



**Returns:**
 
 - <b>`tuple`</b>:  (dataframe, dictionary) 


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

## <kbd>function</kbd> `run_notebook`

```python
run_notebook(
    parameters: dict,
    input_notebook_path: str,
    kernel: str,
    output_notebook_path: str = None,
    test=False,
    verbose=False,
    **kws_papermill
)
```

[UNDER DEVELOPMENT] Execute a single notebooks.     


---

## <kbd>function</kbd> `run_notebooks`

```python
run_notebooks(
    input_notebook_path: str,
    inputs: list,
    output_path: str,
    kernel: str,
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
