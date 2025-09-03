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
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts-en.ipynb
   quickstart/concepts-zh.ipynb




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials


   tutorials/state-en.ipynb
   tutorials/state-zh.ipynb
   tutorials/pygraph-en.ipynb
   tutorials/pygraph-zh.ipynb
   tutorials/program_compilation-en.ipynb
   tutorials/program_compilation-zh.ipynb
   tutorials/program_augmentation-en.ipynb
   tutorials/program_augmentation-zh.ipynb
   tutorials/optimizers-en.ipynb
   tutorials/optimizers-zh.ipynb
   tutorials/gspmd-en.ipynb
   tutorials/gspmd-zh.ipynb
   tutorials/random_numbers-en.ipynb
   tutorials/random_numbers-zh.ipynb
   tutorials/checkpointing-en.ipynb
   tutorials/checkpointing-zh.ipynb
   tutorials/artificial_neural_networks-en.ipynb
   tutorials/artificial_neural_networks-zh.ipynb
   tutorials/spiking_neural_networks-en.ipynb
   tutorials/spiking_neural_networks-zh.ipynb




.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/core_examples.rst
   examples/ann_training-en.ipynb
   examples/ann_training-zh.ipynb
   examples/snn_simulation-en.ipynb
   examples/snn_simulation-zh.ipynb
   examples/snn_training-en.ipynb
   examples/snn_training-zh.ipynb



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   api.rst

