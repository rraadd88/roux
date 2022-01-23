import argh
from roux.lib.io import backup,version,to_zip

## begin
import sys
parser = argh.ArghParser()
parser.add_commands([backup,version,to_zip])

if __name__ == '__main__':
    parser.dispatch()