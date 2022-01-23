"""
## Development

### release new version

    git commit -am "version bump";git push origin master
    python setup.py --version
    git tag -a v$(python setup.py --version) -m "upgrage";git push --tags

"""

import sys
if (sys.version_info[0]) != (3):
     raise RuntimeError('Python 3 required ')

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

## dependencies/requirements
requirements = {
'base':  """biopython>=1.71
            regex>=2018.7.11
            pandas>=0.25.3
            pyyaml>=5.1
            numpy>=1.17.3
            matplotlib>=2.2
            requests>=2.19.1
            scipy>=1.1.0
            tqdm>=4.38.0
            fastparquet
            pandarallel
            xlrd
            openpyxl
            argh
            icecream""".split('\n'),
'stat':['statsmodels',
        # 'pingouin',
        ],
'viz':['seaborn>=0.8',
      ],
'query':['pyensembl',
         'pybiomart'],
'workflow':['snakemake',
            'gitpython'],
'dev':[
    'black',
    'coveralls == 3.*',
    'flake8',
    'isort',
    'pytest == 6.*',
    'pytest-cov == 2.*',
],
}
extras_require={k:l for k,l in requirements.items() if not k=='base'}
## all: extra except dev
extras_require['all']=[l for k,l in extras_require.items() if not k=='dev']
### flatten
extras_require['all']=[s for l in extras_require['all'] for s in l]
### unique
extras_require['all']=list(set(extras_require['all']))

setuptools.setup(
    name='roux',
    version='0.0.1',
    description='Your project description here',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/rraadd88/roux',
    author='rraadd88',
    license='General Public License v. 3',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements['base'],
    extras_require=extras_require,
    entry_points={
    'console_scripts': ['rohan = roux.run:parser.dispatch',],
    },    
    python_requires='>=3.7, <4',
)
