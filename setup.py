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

## dependencies/requirements
requirements = {
'base':[
    'numpy>=1.17.3',
    'pandas>=0.25.3,<=2.1.4',
    'pyyaml>=5.1',
    'matplotlib>=2.2',
    'seaborn>=0.8',
    'tqdm>=4.38.0', # logging
    # 'regex>=2018.7.11',
    ],
'interactive':[
    'ipywidgets',
    'ipykernel',
    # 'watermark', # session info
],
'data':[
    'openpyxl', # special io
],
'fast':[
    'pyarrow', # fast
],
'stat':[
    'scipy>=1.1.0',
    'statsmodels',
    'networkx',
    'scikit-learn',
        ],
'viz':[
    'altair',
      ],
'workflow':[
    'omegaconf',
    'papermill',
    'argh',
    'ruff',
    'nbconvert',
    'removestar',
],
## development and maintenance
'dev':[
    'pytest',
    'jupyter','ipywidgets','ipykernel',
    # 'sphinx','recommonmark',
    'black',
    'coveralls == 3.*',
    'flake8',
    'isort',
    'pytest-cov >= 2,< 6',
    'testbook',
    'lazydocs', 'regex', ## docs
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
    version='0.1.3',
    description='Convenience functions.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/rraadd88/roux',
    author='rraadd88',
    author_email='rohanadandage@gmail.com',
    license='General Public License v. 3',
    packages=setuptools.find_packages('.',exclude=['test','tests', 'unit','deps','data','examples']),
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    install_requires=requirements['base'],
    extras_require=extras_require,
    entry_points={
    'console_scripts': ['roux = roux.run:parser.dispatch',],
    },
    python_requires='>=3.7, <4',
)
