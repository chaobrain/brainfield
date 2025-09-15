``brainmass`` documentation
============================

`brainmass <https://github.com/chaobrain/brainmass>`_ implements neural filed models with `brainstate <https://github.com/chaobrain/brainstate>`_,
a JAX-based framework for large-scale brain dynamics programming.




----

Features
^^^^^^^^^

.. grid::


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Intuitive Programming
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``brainmass`` provides simple interface to build complex neural mass models.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Differentiable Optimization
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``brainmass`` supports differentiable optimizations to fit model parameters to empirical data.



----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainmass[cpu]

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U brainmass[cuda12]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainmass[tpu]

----


See also the brain modeling ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/wilsonwowan-osillator.ipynb
   examples/parameter-exploration.ipynb
   examples/momi2023.ipynb
   examples/jansenrit_single_node_simulation.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   ../changelog.md
   api.rst

