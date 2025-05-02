"""
## Development

### release new version

    git commit -am "version bump";git push origin master
    python setup.py --version
    git tag -a v$(python setup.py --version) -m "upgrade";git push --tags

"""

import sys
if (sys.version_info[0]) != (3):
    raise RuntimeError('Python 3 required ')

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='roux',
    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages('.',exclude=['test','tests', 'unit','deps','data','examples']),
)
