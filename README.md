<!-- PROJECT SHIELDS -->
<div align="center">
  
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url] -->
<a href="">[![PyPI](https://img.shields.io/pypi/v/roux?style=for-the-badge)![Python](https://img.shields.io/pypi/pyversions/roux?style=for-the-badge)](https://pypi.org/project/roux)</a>
<a href="">[![build](https://img.shields.io/github/actions/workflow/status/rraadd88/roux/build.yml?style=for-the-badge)](https://github.com/rraadd88/roux/actions/workflows/build.yml)</a>
<a href="">[![Issues](https://img.shields.io/github/issues/rraadd88/roux.svg?style=for-the-badge)](https://github.com/rraadd88/roux/issues)</a>
<br />
<a href="">[![Downloads](https://img.shields.io/pypi/dm/roux?style=for-the-badge)](https://pepy.tech/project/roux)</a>
<a href="">[![GNU License](https://img.shields.io/github/license/rraadd88/roux.svg?style=for-the-badge)](https://github.com/rraadd88/roux/blob/master/LICENSE)</a>
</div>
  
<!-- PROJECT LOGO -->
<div align="center">
  <img src="https://github.com/rraadd88/roux/assets/9945034/c2a84fca-0cc5-4ecc-8c9a-d83833fd920d" alt="logo" />
  <h1 align="center">roux</h1>
  <p align="center">
    Streamlined scientific compute and workflow management toolkit.
    <br />
    <a href="https://github.com/rraadd88/roux#examples">Examples</a>
    ·
    <a href="https://github.com/rraadd88/roux#api">Explore the API</a>
  </p>
</div>  

![image](./examples/image.png)   

# Usage  

## Examples

[⌗ Dataframes.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_df.ipynb)  
[⌗⌗ Paired Dataframes.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_dfs.ipynb)  
[💾 General Input/Output.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_io.ipynb)  
[⬤⬤ Sets.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_set.ipynb)  
[🔤 Strings encoding/decoding.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_str.ipynb)  
[🗃 File paths Input/Output.](https://github.com/rraadd88/roux/blob/master/examples/roux_lib_sys.ipynb)  
[🏷 Classification.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_classify.ipynb)  
[✨ Clustering.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_cluster.ipynb)  
[✨ Correlations.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_corr.ipynb)  
[✨ Differences.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_diff.ipynb)  
[📈 Data fitting.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_fit.ipynb)  
[📊 Data normalization.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_norm.ipynb)  
[⬤⬤ Comparison between sets.](https://github.com/rraadd88/roux/blob/master/examples/roux_stat_sets.ipynb)  
[📈🔖Annotating visualisations.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_annot.ipynb)  
[🔧 Subplot-level adjustments.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_ax.ipynb)  
[📈 Diagrams.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_diagram.ipynb)  
[📈 Distribution plots.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_dist.ipynb)  
[📈 Wrapper around Series plotting functions.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_ds.ipynb)  
[📈📈Annotating figure.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_figure.ipynb)  
[📈💾 Visualizations Input/Output.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_io.ipynb)  
[📈 Line plots.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_line.ipynb)  
[📈 Scatter plots.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_scatter.ipynb)  
[📈⬤⬤ Plots of sets.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_sets.ipynb)  
[📈🎨✨ Visualizations theming.](https://github.com/rraadd88/roux/blob/master/examples/roux_viz_theme.ipynb)  
[⚙️🗺️ Reading multiple configs.](https://github.com/rraadd88/roux/blob/master/examples/roux_workflow_io.ipynb)  
[⚙️⏩ Running multiple tasks.](https://github.com/rraadd88/roux/blob/master/examples/dev_roux_workflow_task.ipynb)  
[⚙️⏩ Workflow using notebooks](https://github.com/rraadd88/roux/blob/master/examples/dev_workflow.ipynb)  
  
## CLI 

ℹ️ Available command line tools and their usage.  
`roux --help`

🗺️ Read configuration.  
`roux read-config path/to/file`  

🗺️ Read metadata.  
`roux read-metadata path/to/file`  

⭐ Remove *'s from a jupyter notebook'.  
`roux removestar path/to/notebook`  

💾 Backup a directory with a timestamp (ISO).  
`roux backup path/to/directory`  

# Installation  
```
pip install roux              # with basic dependencies  
pip install roux[all]         # with all the additional dependencies (recommended). 
```
With additional dependencies as required:
```
pip install roux[viz]         # for visualizations e.g. seaborn etc.
pip install roux[data]        # for data operations e.g. reading excel files etc.
pip install roux[stat]        # for statistics e.g. statsmodels etc.
pip install roux[fast]        # for faster processing e.g. parallelization etc.
pip install roux[workflow]    # for workflow operations e.g. omegaconf etc.
pip install roux[interactive] # for interactive operations in jupyter notebook e.g. watermark, icecream etc.
```
  
# How to cite?  
```
@software{Dandage_roux,
  title   = {roux: Streamlined and Versatile Data Processing Toolkit},
  author  = {Dandage, Rohan},
  year    = {2024},
  url     = {https://zenodo.org/doi/10.5281/zenodo.2682670},
  version = {0.1.2},
  note    = {The URL is a DOI link to the permanent archive of the software.},
}
```
<!--   

# Future directions, for which contributions are welcome  
- [ ] Addition of visualization function as attributes to `rd` dataframes.  
- [ ] Refactoring of the workflow functions.   -->
  
# Similar packages
- https://github.com/v-popov/helper_funcs  
- https://github.com/nficano/yakutils  

<!-- # [API](https://github.com/rraadd88/roux/blob/master/README_API.md) -->