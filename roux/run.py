"""For access to a few functions from the terminal."""
import logging
logging.getLogger().setLevel(logging.INFO)

import argh
from roux.lib.io import backup,to_version,to_zip
from roux.lib.sys import read_ps
from roux.workflow.io import read_config, read_metadata, replacestar
from roux.workflow.task import run_tasks

## begin
parser = argh.ArghParser()
parser.add_commands([read_ps,read_config, read_metadata,backup,to_version,to_zip,replacestar,run_tasks])

if __name__ == '__main__':
    parser.dispatch()