"""For access to a few functions from the terminal."""

import logging

logging.getLogger().setLevel(logging.INFO)

import argh
from roux.lib.io import backup, to_version, to_zip
from roux.lib.io import pqt2tsv
from roux.lib.sys import read_ps
from roux.workflow.io import read_config, read_metadata, replacestar, to_clean_nb
from roux.workflow.task import run_tasks
from roux.workflow.nb import to_clear_unused_cells, to_clear_outputs

## begin
parser = argh.ArghParser()
parser.add_commands(
    [
        ## io
        read_ps,
        ## backup
        backup,
        to_version,
        to_zip,
        pqt2tsv,
        ## workflow io
        read_config,
        read_metadata,
        ## workflow execution
        run_tasks,
        ## notebook post-processing
        replacestar,
        to_clear_unused_cells,
        to_clear_outputs,
        to_clean_nb,  ## wrapper for above
    ]
)

if __name__ == "__main__":
    parser.dispatch()
