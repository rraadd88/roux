"""
For importing commonly used functions at the development phase.

Usage: in interactive sessions (e.g. in jupyter notebooks) to facilitate faster code development.

Note: Post-development, to remove *s from the code, use removestar (pip install removestar).
    
    removestar file
"""
## logging functions
import logging
logging.basicConfig(
    level=logging.INFO,
    # format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..): %(message)s'
    )

try:
    from icecream import ic as info
    info.configureOutput(prefix='INFO:icrm:')
except ImportError:
    logging.warning('Optional dependency icecream missing, install by running: pip install roux[interactive]')


## data functions
import itertools
import numpy as np
import pandas as pd

## system functions
import sys
from pathlib import Path
from os.path import exists,dirname,basename,abspath,isdir,splitext
from glob import glob
## system functions from roux
from roux.lib.sys import read_ps, basenamenoext, to_path, makedirs, get_datetime
from roux.lib.io import read_dict, to_dict, read_table, to_table, backup
## data functions from roux
from roux.lib.str import get_bracket, replace_many, get_suffix, get_prefix
from roux.lib.set import dropna, flatten, unique, assert_overlaps_with, validate_overlaps_with, check_non_overlaps_with
from roux.lib.dict import merge_dicts
import roux.lib.dfs as rd # attributes
## workflow functions from roux
from roux.workflow.io import read_metadata,infer_parameters#, read_config, to_diff_notebooks
from roux.workflow.log import print_parameters

# diplay tables
# from functools import partialmethod
# pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=1)
# pd.set_option('display.max_rows', 2)

## stats functions
import scipy as sc

## stats functions from roux
from roux.stat.binary import perc
from roux.stat.io import perc_label

## visualization functions
import matplotlib.pyplot as plt
import seaborn as sns
# settings
FONTSIZE=12
PAD=2
plt.set_loglevel('error')
plt.style.use('ggplot')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['legend.frameon']=False
from cycler import cycler
plt.rcParams['axes.prop_cycle']= cycler('color',[
    '#50AADC',#blue
    "#D3DDDC",#gray
    '#F1D929',#yellow
    "#f55f5f",#red
    "#046C9A",#blue
    "#00A08A", "#F2AD00", "#F98400", "#5BBCD6", "#ECCBAE", "#D69C4E", "#ABDDDE", "#000000"])
# plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
# ticks
# plt.rcParams['xtick.color']=[0.95,0.95,0.95]
plt.rc('axes', grid=False,axisbelow=True,unicode_minus=False,
       labelsize=FONTSIZE,
       labelcolor='k',labelpad=PAD,
       titlesize=FONTSIZE,
       facecolor='none',
       edgecolor='k',linewidth=0.5,)
# plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = -3, 3
plt.rcParams['axes.formatter.min_exponent'] = 3

plt.rcParams["xtick.major.size"] = PAD
plt.rcParams["ytick.major.size"] = PAD
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["xtick.major.pad"] = PAD
plt.rcParams["ytick.major.pad"] = PAD
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# scale
plt.rc('figure',figsize = (3, 3))
plt.rc('figure.subplot',wspace= 0.3,hspace= 0.3)
# sns.set_context('notebook') # paper < notebook < talk < poster

## visualization functions from roux
from roux.viz.io import begin_plot,get_plot_inputs,to_plot,read_plot
from roux.viz.colors import get_colors_default
from roux.viz.diagram import diagram_nb

## logging functions
from tqdm import tqdm
## system functions from roux
from roux.lib.sys import is_interactive_notebook
if not is_interactive_notebook():
    # progress bar
    tqdm.pandas()
else:
    ## logging functions
    from tqdm import notebook
    try:
        notebook.tqdm().pandas()
    except:
        logging.warning("ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html")
    from IPython.display import Markdown as info_nb
    # display vector graphics in jupyter
    # if not get_ipython() is None:
    #     get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")
try:
    import watermark.watermark as watermark # session info
    logging.info(watermark(python=True)+watermark(iversions=True,globals_=globals()))
except ImportError:
    logging.warning('Optional dependency watermark missing, install by running: pip install roux[interactive]')
    
try:
    ## parallel-pocessing functions
    from pandarallel import pandarallel;pandarallel.initialize(nb_workers=6,progress_bar=True,use_memory_fs=False) # attributes
    # logging.info("pandarallel.initialize(nb_workers=4,progress_bar=True)")
except ImportError:
    logging.warning('Optional dependency pandarallel missing, install by running: pip install roux[fast]')