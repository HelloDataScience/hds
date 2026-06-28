import re
from pathlib import Path

from setuptools import find_packages, setup


def read_version():
    init_file = Path(__file__).parent / 'hds' / '__init__.py'
    text = init_file.read_text(encoding='UTF-8')
    match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", text, re.M)
    if match is None:
        raise RuntimeError('Unable to find __version__ in hds/__init__.py')
    return match.group(1)


with open('README.md', encoding='UTF-8') as file:
    long_description = file.read()

setup(
    name='hds',
    version=read_version(),
    author='HelloDataScience',
    author_email='hellodatasciencekorea@gmail.com',
    description='Functions for EDA, Statistics and Machine Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HelloDataScience/hds',
    project_urls={
        'Bug Tracker': 'https://github.com/HelloDataScience/hds/issues',
    },
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'matplotlib',
        'statsmodels',
        'scikit-learn',
        'graphviz',
        'requests',
        'bs4',
        'varname',
        'ipywidgets',
        'ipython',
    ],
)
