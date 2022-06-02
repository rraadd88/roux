"""
End-point.
-> global_imports
"""
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# temporary change for a context
# ref: https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
# with LoggingContext(logger, level=logging.ERROR):
#     logger.debug('3. This should appear once on stderr.')

# recepies
from pathlib import Path
from roux.lib.str import get_bracket, replace_many,get_suffix,get_prefix
from roux.lib.dict import *
from roux.lib.set import *
from roux.lib.text import read,read_lines
from roux.lib.io import * #io_df -> io_dfs -> io_files
from roux.lib.dict import to_dict # to replace io_df to_dict
from roux.workflow.io import read_metadata
from roux.workflow.df import *

# defaults
from functools import partialmethod
# pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=1)
#pd.set_option('display.max_rows', 2)

# stats    
import scipy as sc
from roux.stat.binary import perc

# paths
pwd=abspath('.')
prjs=['00_metaanalysis']
# plots
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'Myriad Pro'
plt.rcParams['figure.figsize'] = (3, 3)
plt.rcParams['axes.facecolor']='none'
plt.rcParams['axes.edgecolor']='k'
# plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.formatter.limits'] = -3, 3
plt.rcParams['axes.formatter.min_exponent'] = 3
plt.rcParams['legend.frameon']=True
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
plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
plt.rc('axes', axisbelow=True)
plt.rc('axes', unicode_minus=False)
plt.rcParams['axes.labelcolor'] = 'k'
sns.set_context('notebook') # paper < notebook < talk < poster
from roux.viz.figure import *
from roux.viz.io import log_code,get_plot_inputs,to_plot,read_plot#*
from roux.viz.ax_ import *
from roux.viz.annot import *

from tqdm import tqdm#,notebook
# from roux.lib.sys import is_interactive_notebook
if not is_interactive_notebook:
    from IPython import get_ipython
    logging.info("log_notebookp=f'log_notebook.log';open(log_notebookp, 'w').close();get_ipython().run_line_magic('logstart','{log_notebookp} over')")
    tqdm.pandas()
else:
    from tqdm import notebook
    notebook.tqdm().pandas()
    ## display vector graphics in jupyter
    # if not get_ipython() is None:
    #     get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=6,progress_bar=True)
logging.info("pandarallel.initialize(nb_workers=4,progress_bar=True)")

# metadata
metadata=read_metadata()
