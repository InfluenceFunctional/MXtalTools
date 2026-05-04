{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

.. rubric:: Submodules

.. autosummary::
   :toctree:

{% for item in modules %}
   {{ item }}
{%- endfor %}
