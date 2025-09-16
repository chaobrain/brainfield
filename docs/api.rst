API Documentation
=================


.. currentmodule:: brainmass
.. automodule:: brainmass


A large-scale brain network modeling framework typically consists of four hierarchical layers:

    Structural Connectivity (DTI / Structural MRI)
       ↓
    Neural Mass Models (NMMs)
       ↓
    Biophysical Forward Model
       ↓
    Observed Signals (EEG, MEG, fMRI BOLD)



Node Dynamics
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    JansenRitModel
    WilsonCowanModel
    WongWangModel
    HopfModel



Noise Processes
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    OUProcess
    GaussianNoise
    WhiteNoise
    ColoredNoise
    BrownianNoise
    PinkNoise
    BlueNoise
    VioletNoise



Node Coupling
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    DiffusiveCoupling
    AdditiveCoupling
    diffusive_coupling
    additive_coupling



Forward Models
--------------


.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    BOLDSignal
    LeadFieldModel


