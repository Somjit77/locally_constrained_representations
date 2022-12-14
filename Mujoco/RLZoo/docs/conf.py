# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))  # Important
sys.path.insert(0, os.path.abspath(os.path.join("..", "rlzoo")))  # Important

# from rlzoo.algorithms import *
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'RLzoo'
copyright = '2020, Zihan Ding, Tianyang Yu, Yanhua Huang, Hongming Zhang, Hao Dong'
author = 'Zihan Ding, Tianyang Yu, Yanhua Huang, Hongming Zhang, Hao Dong'

# The full version, including alpha/beta/rc tags
release = '1.0.3'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    # 'sphinxcontrib.bibtex',
    'recommonmark'
]

autodoc_mock_imports = [
    'cv2',
    'hyperdash',
    'gridfs',
    'horovod',
    'hyperdash',
    'imageio',
    'lxml',
    'matplotlib',
    'nltk',
    # 'numpy',
    'PIL',
    'progressbar',
    'pymongo',
    'scipy',
    'skimage',
    'sklearn',
    # 'tensorflow',
    'tqdm',
    'h5py',
    # 'tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops',  # TL C++ Packages
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_logo = './img/rlzoo-logo.png'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
