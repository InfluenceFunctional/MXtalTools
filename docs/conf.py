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

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../mxtaltools/"))


# -- Project information -----------------------------------------------------

project = "MXtalTools"
copyright = "2024, Michael Kilgour"
author = "Michael Kilgour"

# The full version, including alpha/beta/rc tags
version = "0.1.0"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
#               'sphinx.ext.coverage', 'sphinx.ext.pngmath', 'sphinx.ext.ifconfig',
#               'epub2', 'mobi', 'autoimage', 'code_example']
#
# todo_include_todos = True
# templates_path = ['_templates']
# source_suffix = '.rst'
# master_doc = 'index'
# exclude_patterns = []
# add_function_parentheses = True
#
# # -- Options for HTML output ---------------------------------------------------
#
# html_theme = 'book'
# html_theme_path = ['themes']
# html_title = "Music for Geeks and Nerds"
# #html_short_title = None
# #html_logo = None
# #html_favicon = None
# html_static_path = ['_static']
# html_domain_indices = False
# html_use_index = False
# html_show_sphinx = False
# htmlhelp_basename = 'MusicforGeeksandNerdsdoc'
# html_show_sourcelink = False
#
# # List of patterns, relative to source directory, that match files and
# # directories to ignore when looking for source files.
# # This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
#
#
#
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "pytorch": ("https://pytorch.org/docs/stable/", None),
#     "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
#     "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
# }
#
# autodoc_default_options = {
#     "inherited-members": False,
#     "show-inheritance": True,
#     "autosummary": False,
# }
#
# # The reST default role to use for all documents.
# default_role = "any"
#
# # -- Options for HTML output -------------------------------------------------
#
# # The theme to use for HTML and HTML Help pages.  See the documentation for
# # a list of builtin themes.
# #
# html_theme = "sphinx_rtd_theme"
#
# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# # html_static_path = ['_static']
#
# myst_update_mathjax = False
