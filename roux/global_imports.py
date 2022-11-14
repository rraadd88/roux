"""For the use in jupyter notebook for example."""
## logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# helper functions
from roux.lib.str import get_bracket, replace_many,get_suffix,get_prefix
from roux.lib.dict import *
from roux.lib.set import *
from roux.lib.io import * #df -> dfs -> io
from roux.lib.dict import * # to replace df to_dict
from roux.workflow.io import read_metadata, read_config, to_diff_notebooks
from roux.workflow.df import *

# diplay tables
# from functools import partialmethod
# pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=1)
# pd.set_option('display.max_rows', 2)

# commonly used stats functions
import scipy as sc
from roux.stat.binary import perc
from roux.stat.io import perc_label

# plots
import matplotlib.pyplot as plt
import seaborn as sns
## settings
plt.set_loglevel('error')
plt.style.use('ggplot')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['figure.figsize'] = (3, 3)
plt.rcParams['axes.facecolor']='none'
plt.rcParams['axes.edgecolor']='#6B6B6B'
# plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = -3, 3
plt.rcParams['axes.formatter.min_exponent'] = 3
# plt.rcParams['legend.frameon']=True
from cycler import cycler
plt.rcParams['axes.prop_cycle']= cycler('color',[
    '#50AADC',#blue
    "#D3DDDC",#gray
    '#F1D929',#yellow
    "#f55f5f",#red
    "#046C9A",#blue
    "#00A08A", "#F2AD00", "#F98400", "#5BBCD6", "#ECCBAE", "#D69C4E", "#ABDDDE", "#000000"])
from roux.viz.colors import get_colors_default
# plt.rcParams['xtick.color']=[0.95,0.95,0.95]
# plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
plt.rcParams['axes.grid']=False
plt.rc('axes', axisbelow=True)
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.labelcolor'] = 'k'
sns.set_context('notebook') # paper < notebook < talk < poster
## helper functions
from roux.viz.figure import *
from roux.viz.io import log_code,get_plot_inputs,to_plot,read_plot#*
from roux.viz.ax_ import *
from roux.viz.annot import *

from tqdm import tqdm#,notebook
# from roux.lib.sys import is_interactive_notebook
if not is_interactive_notebook():
    ## progress bar
    tqdm.pandas()
else:
    from tqdm import notebook
    notebook.tqdm().pandas()
    ## markdown in jupyter containing variables
    from IPython.display import Markdown as info_nb
    ## display vector graphics in jupyter
    # if not get_ipython() is None:
    #     get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")

## session info
import watermark.watermark as watermark
logging.info(watermark(python=True)+watermark(iversions=True,globals_=globals()))

## parallel processing
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=6,progress_bar=True,use_memory_fs=False)
# logging.info("pandarallel.initialize(nb_workers=4,progress_bar=True)")

# metadata
if exists('metadata.yaml'):
    metadata=read_metadata('metadata.yaml')