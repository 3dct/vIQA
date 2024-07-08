# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'vIQA'
copyright = '2024, Lukas Behammer'
author = 'Lukas Behammer'

with open("../../src/viqa/__init__.py") as f:
    setup_lines = f.readlines()
version = "vUndefined"
for line in setup_lines:
    if line.startswith("__version__"):
        version = line.split('"')[1]
        break

release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx_copybutton',
    'pytest_doctestplus.sphinx.doctestplus',
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


# -- Options for Autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'exclude-members': '__init__',
}

autodoc_mock_imports = [
    "viqa.fr_metrics.stat_utils",
    "viqa.nr_metrics.qmeasure_utils",
]
