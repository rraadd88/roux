import sys
if (sys.version_info[0]) != (3):
     raise RuntimeError('Python 3 required ')

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

REQUIREMENTS =   """biopython>=1.71
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
                    icecream""".split('\n')
    
DEV_REQUIREMENTS = [
    'black',
    'coveralls == 3.*',
    'flake8',
    'isort',
    'pytest == 6.*',
    'pytest-cov == 2.*',
]

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
    install_requires=REQUIREMENTS,
    extras_require={
        'stat':['statsmodels',
                # 'pingouin',
                ],
        'viz':['seaborn>=0.8',
              ],
        'query':['pyensembl',
                 'pybiomart'],
        'workflow':['snakemake',
                    'gitpython'],
        'dev': DEV_REQUIREMENTS,
    },
    entry_points={
    'console_scripts': ['rohan = roux.run:parser.dispatch',],
    },    
    python_requires='>=3.7, <4',
)
