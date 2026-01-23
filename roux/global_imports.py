"""
(*Mise en place*) Importing commonly used functions.

Requirements: 

    pip install roux[all]
    
Usage: in interactive sessions (e.g. in jupyter notebooks) to facilitate faster code development.

Note: Post-development, to remove *s from the code, use removestar (pip install removestar).

    removestar file


Examples:

## imports

from roux.global_imports import * #noqa
## outputs
# from roux.workflow.io import set_outputs
output_dir_path,output_paths=set_outputs(output_path=locals().get('output_path'),output_paths=locals().get('output_paths'),)

## modules

#if export_mod:
#    sys.exit(0)
from roux.workflow.io import get_source_path,to_mod
to_mod(get_source_path())

## scripts

## END
#if export_src:
#    sys.exit(0)
from roux.workflow.io import get_source_path,to_src
to_src(get_source_path())
"""
import warnings
warnings.warn(
    "'from roux.global_imports import * #noqa' is intended for interactive use in environments like Jupyter. Elsewhere, explicit imports are recommended.",
    UserWarning,
    stacklevel=2,
)

## begin replacestar    
## data functions
# import itertools
import numpy as np #noqa
import pandas as pd #noqa

## system functions
import sys #noqa
from pathlib import Path #noqa
from os.path import exists,dirname,basename,abspath,isdir,splitext #noqa # pathlib to be preferred in the future
## system functions from roux
from roux.lib.sys import (
    read_ps, with_stem_suffix, #noqa

    ## TODO: to be removed bcz rarely used
    basenamenoext, to_path, makedirs, get_datetime #noqa
    ) 
from roux.lib.io import read_dict, to_dict, read_table, to_table, read_data, to_version, backup #noqa
## data functions from roux
from roux.lib.str import get_bracket, replace_many, get_suffix, get_prefix #noqa
from roux.lib.set import dropna, flatten, unique, assert_overlaps_with, validate_overlaps_with, check_non_overlaps_with #noqa
## dataframe attribute from roux
# attributes
import roux.lib.dfs as rd #noqa
import roux.viz.ds as rs #noqa

## stats functions from roux
from roux.stat.binary import perc #noqa
from roux.stat.io import perc_label #noqa

## visualization functions
import matplotlib.pyplot as plt #noqa
import seaborn as sns #noqa

## visualization functions from roux
from roux.viz.theme import set_theme #noqa
from roux.viz.ax_ import format_ax #noqa
from roux.viz.colors import get_colors_default #noqa
from roux.viz.diagram import diagram_nb #noqa
from roux.viz.io import begin_plot,to_plot,read_plot #noqa

## workflow functions from roux
from roux.workflow.io import infer_parameters #noqa #, read_config, to_diff_notebooks
from roux.workflow.cfgs import read_config, read_metadata #noqa
from roux.workflow.log import test_params #noqa
from roux.workflow.task import run_tasks #noqa

## logging functions
## setting states

import logging
# root logger level for the imported functions
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# g: 1. Create the formatter object
formatter = logging.Formatter(
    '%(levelname)s:(%(funcName)s): %(message)s'
)

# g: 2. Check if the root logger already has handlers
if not root_logger.handlers:
    # g: If no handlers exist, create a new one (e.g., to stderr)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    # g: If handlers already exist (e.g., in a notebook),
    # g: just set the formatter on the existing handler(s).
    # g: We'll modify the first one.
    root_logger.handlers[0].setFormatter(formatter)
    # g: Also ensure the handler's level is permissive (if needed)
    # root_logger.handlers[0].setLevel(logging.INFO)

## for the notebook
try:    
    from roux.lib.log import Logger
    logging = Logger() # level='INFO'
except:
    import logging #noqa
    logging.basicConfig(level='INFO', force=True)

from tqdm import tqdm #noqa
## end replacestar

## system functions from roux
from roux.lib.sys import is_interactive_notebook #noqa
if not is_interactive_notebook():
    # progress bar
    tqdm.pandas()
else:
    ## logging functions
    from tqdm import notebook #noqa
    try:
        notebook.tqdm().pandas()
    except ImportError as e:
        logging.warning(f"{e}: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html")
        
## extra

    # from IPython.display import Markdown as info_nb #noqa    
    # try:
    #     import watermark.watermark as watermark # session info
    #     logging.info(watermark(python=True)+watermark(iversions=True,globals_=globals()))
    # except ImportError:
    #     logging.warning('Optional interactive-use dependencies missing, install by running: pip install roux[interactive]')
    
# try:
#     ## parallel-pocessing functions
#     from pandarallel import pandarallel;pandarallel.initialize(nb_workers=6,progress_bar=True,use_memory_fs=False) # attributes
#     # logging.info("pandarallel.initialize(nb_workers=4,progress_bar=True)")
# except ImportError:
#     logging.warning('Optional dependency pandarallel missing, install by running: pip install roux[fast]')

    # display vector graphics in jupyter
    # if not get_ipython() is None:
    #     get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")

# diplay tables
# from functools import partialmethod
# pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=1)
# pd.set_option('display.max_rows', 2)

# try:
#     ## stats functions
#     import scipy as sc
# except ImportError:
#     logging.warning('Optional dependency scipy missing, install by running: pip install roux[stat]')

# ## if locals exist
from roux.workflow.io import set_outputs #noqa