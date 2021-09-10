# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# for extensions
sys.path.insert(0, os.path.abspath('.'))

# This is needed for readthedocs for some reason
sys.path.insert(1, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------
import clipppy

project = 'CLIPPPY'
copyright = '2021, Kosio Karchev'
author = 'Kosio Karchev'
release = str(clipppy.__version__)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    # 'sphinx.ext.viewcode',
    'autoapi.extension',
    'recommonmark',
    '_ext.regex_lexer',
    '_ext.any_tilde'
]

todo_include_todos = True

autodoc_typehints = 'signature'

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'pyro': ('http://docs.pyro.ai/en/stable/', None),
}

napoleon_google_docstring = False

autoapi_dirs = ['../../src/clipppy']
autoapi_file_patterns = ['*.pyi', '*.py']
autoapi_ignore = [
    '**/contrib.py',
    '**/autocli.py',
    '**/cli.py',
]
autoapi_options = [
    'members', 'undoc-members', 'private-members', 'special-members',
    'show-inheritance', 'show-inheritance-diagram',
]
autoapi_member_order = 'groupwise'
autoapi_keep_files = True

default_role = 'any'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
root_doc = 'index'
exclude_patterns = ['userguide/*/*']

trim_footnote_reference_space = True

# language=rst
rst_prolog = '''
.. |Clipppy| replace:: Clipppy
.. |citation needed| replace:: `[citation needed]`:superscript:
.. _ruamel.yaml: https://yaml.readthedocs.io/en/latest/
.. _pyrofit: https://github.com/cweniger/pyrofit-core

.. |builtinsvar| replace:: `ClipppyYAML.constructor`\ ``.``\ `~ScopeMixin.builtins`
.. |scopevar| replace:: `ClipppyYAML.constructor`\ ``.``\ `~ScopeMixin.scope`

.. role:: strike
    :class: strike
.. role:: underline
    :class: underline

.. role:: arg(literal)

.. role:: raw-html(raw)
    :format: html

.. role:: python(code)
    :class: highlight
    :language: python3
.. role:: regex(code)
    :class: highlight
    :language: regex
.. role:: json(code)
    :class: highlight
    :language: json
.. role:: yaml(code)
    :class: highlight
    :language: yaml
'''

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['style.css']
