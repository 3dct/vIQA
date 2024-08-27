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
    'sphinx.ext.intersphinx',
    'sphinxcontrib.email',
    'sphinx_copybutton',
    'sphinx_github_changelog',
    'sphinx_last_updated_by_git',
    'pytest_doctestplus.sphinx.doctestplus',
    # 'hoverxref.extension', # Works only on readthedocs
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_favicon = '../../branding/logo/Logo_vIQA_wo-text.svg'
html_title = f"{project} documentation v{release}"

# -- Options for Autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': None,
    'member-order': 'bysource',
    'exclude-members': '__init__',
    'inherited-members': None,
}

autodoc_mock_imports = [
    "viqa.fr_metrics.stat_utils",
    "viqa.nr_metrics.qmeasure_utils",
]

# -- Options for Hoverxref ---------------------------------------------------
hoverxref_domains = ['py']
hoverxref_intersphinx = [
    'python', 'numpy', 'scipy', 'matplotlib', 'scikit-image', 'nibabel', 'piq'
]

# -- Options for Github Changelog --------------------------------------------
sphinx_github_changelog_token = os.getenv("GH_TOKEN")

# -- Options for Intersphinx -------------------------------------------------
intersphinx_mapping = {
    "scikit-image": ("https://scikit-image.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "piq": ("https://piq.readthedocs.io/en/latest/", None),
}
