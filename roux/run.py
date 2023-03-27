"""For access to a few functions from the terminal."""
import argh
from roux.lib.io import backup,version,to_zip
from roux.workflow.io import removestar

## begin
import sys
parser = argh.ArghParser()
parser.add_commands([removestar,backup,version,to_zip])

if __name__ == '__main__':
    parser.dispatch()