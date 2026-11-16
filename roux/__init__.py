# g: version of the installed package (pyproject.toml); None when roux is not installed, e.g. run from a source tree
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("roux")
except PackageNotFoundError:
    __version__ = None
