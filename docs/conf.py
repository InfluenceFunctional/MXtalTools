# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import types

sys.path.insert(0, os.path.abspath("../"))

# -- Mock heavy dependencies before any imports ------------------------------
# Sphinx's autodoc_mock_imports doesn't handle issubclass() properly for
# class inheritance (e.g., class MyModel(torch.nn.Module)).
# We create mock modules with real classes that support subclassing.


class _MockBase:
    """Base class that supports being subclassed by documented code."""
    def __init__(self, *args, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return type(name, (_MockBase,), {'__module__': 'mock'})


class _MockModule(types.ModuleType):
    """A mock module that returns subclassable objects for any attribute."""

    def __init__(self, name='mock'):
        super().__init__(name)
        self.__all__ = []
        self.__path__ = [name]
        self.__file__ = ''
        self.__package__ = name

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        # Check if a child mock module exists in sys.modules
        child = self.__name__ + '.' + name
        if child in sys.modules:
            return sys.modules[child]
        # Otherwise return a subclassable class
        return type(name, (_MockBase,), {'__module__': self.__name__})


_MOCK_MODULES = [
    # PyTorch ecosystem
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.nn.modules',
    'torch.nn.modules.module', 'torch.optim', 'torch.optim.lr_scheduler',
    'torch.utils', 'torch.utils.data', 'torch.cuda', 'torch.autograd',
    'torch.distributions', 'torch.linalg', 'torch.fft',
    'torch.random', 'torch.serialization',
    # PyTorch Geometric
    'torch_geometric', 'torch_geometric.data', 'torch_geometric.data.data',
    'torch_geometric.data.storage',
    'torch_geometric.nn', 'torch_geometric.nn.conv',
    'torch_geometric.nn.pool', 'torch_geometric.nn.aggr',
    'torch_geometric.nn.dense', 'torch_geometric.nn.dense.linear',
    'torch_geometric.nn.inits',
    'torch_geometric.loader', 'torch_geometric.loader.dataloader',
    'torch_geometric.utils', 'torch_geometric.transforms',
    'torch_geometric.typing',
    # PyTorch extensions
    'torch_scatter', 'torch_cluster', 'torch_sparse',
    # RDKit
    'rdkit', 'rdkit.Chem', 'rdkit.Chem.AllChem', 'rdkit.Chem.Descriptors',
    'rdkit.Chem.rdMolTransforms', 'rdkit.Chem.rdmolops',
    'rdkit.Chem.Draw', 'rdkit.Chem.rdchem', 'rdkit.Geometry',
    # CSD
    'ccdc', 'ccdc.io', 'ccdc.crystal', 'ccdc.molecule',
    'ccdc.descriptors', 'ccdc.search',
    # ASE
    'ase', 'ase.io', 'ase.build', 'ase.optimize', 'ase.calculators',
    'ase.spacegroup', 'ase.geometry', 'ase.cell', 'ase.visualize',
    # MACE / E3NN
    'mace', 'mace.calculators',
    'e3nn', 'e3nn.o3',
    # Ovito
    'ovito', 'ovito.io', 'ovito.data', 'ovito.modifiers', 'ovito.pipeline',
    # Wandb
    'wandb',
]

for _mod_name in _MOCK_MODULES:
    sys.modules[_mod_name] = _MockModule(_mod_name)


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
exclude_patterns = ['_build', 'docs', 'Thumbs.db', '.DS_Store']

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
