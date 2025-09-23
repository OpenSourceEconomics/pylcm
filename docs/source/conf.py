# Configuration file for the Sphinx documentation builder.  # noqa: INP001
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

# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "lcm"
copyright = "2024, Tim Mensinger, Janos Gabler"  # noqa: A001
author = "Tim Mensinger, Janos Gabler"


# Set variable so that todos are shown in local build
on_rtd = os.environ.get("READTHEDOCS") == "True"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_nb",
    "sphinx_panels",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False

extlinks = {
    "ghuser": ("https://github.com/%s", "@"),
    "gh": ("https://github.com/optimagic-dev/optimagic/pulls/%s", "#"),
}

intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.12", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
}

linkcheck_ignore = [
    r"https://tinyurl\.com/*.",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "lcm"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
