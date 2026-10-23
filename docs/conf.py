# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import importlib.util
import os
import re
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Dependencies -------------------------------------------------------------
# The build imports mxtaltools for real, so torch and torch_geometric must be
# installed (Read the Docs installs CPU wheels; see .readthedocs.yaml). The
# packages below are compiled PyG extensions, licensed software, or optional
# MLIP backends. Each is replaced by Sphinx's mock only when it is not installed.

_OPTIONAL_PACKAGES = [
    'torch_scatter', 'torch_cluster', 'torch_sparse',  # PyG C extensions
    'mace', 'e3nn', 'cuequivariance', 'cuequivariance_torch',  # MACE backend
    'fairchem',  # UMA backend
    'ccdc',  # CSD Python API
    'ovito',
]
autodoc_mock_imports = [p for p in _OPTIONAL_PACKAGES if importlib.util.find_spec(p) is None]


# -- Project information -----------------------------------------------------

project = "MXtalTools"
copyright = "2024-2026, Michael Kilgour"
author = "Michael Kilgour"
version = "0.1.0"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.duration',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'design', 'Thumbs.db', '.DS_Store']

# -- Autodoc / Autosummary --------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
    "members": True,
    "undoc-members": True,
}

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output -------------------------------------------------

epub_show_urls = 'footnote'


# -- API page import check ---------------------------------------------------

_API_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'modules.rst')


def _api_page_modules():
    """Every module listed in an autosummary block on the API page."""
    with open(_API_PAGE, encoding='utf-8') as f:
        return re.findall(r'^[ \t]+(mxtaltools(?:\.\w+)+)[ \t]*$', f.read(), flags=re.M)


def _check_api_modules_import(app):
    """Fail the build when a module on the API page cannot be imported.

    autosummary only logs a warning for such a module and leaves its page out,
    so without this check the build succeeds with pages missing.
    """
    from sphinx.errors import ExtensionError
    from sphinx.ext.autodoc.mock import mock

    failures = []
    with mock(app.config.autodoc_mock_imports):
        for name in _api_page_modules():
            try:
                importlib.import_module(name)
            except Exception as e:
                failures.append(f'{name}: {type(e).__name__}: {e}')
    if failures:
        raise ExtensionError(
            f'{len(failures)} module(s) listed in {_API_PAGE} failed to import:\n  '
            + '\n  '.join(failures))


def setup(app):
    app.connect('builder-inited', _check_api_modules_import, priority=100)
