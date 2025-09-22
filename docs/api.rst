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



Quick Start
-----------

The snippets below show the typical workflow: pick a node model, optionally add noise
and coupling, then transform model observables to measurements via a forward model.

.. code-block:: python

   import brainunit as u
   import jax.numpy as jnp
   import brainmass

   # 1) Choose a neural mass model (single region here)
   nmm = brainmass.WilsonCowanModel(in_size=1)
   nmm.init_state()

   # 2) Optional: add stateful noise sources
   noise_E = brainmass.OUProcess(in_size=1, sigma=0.5 * u.Hz, tau=20. * u.ms)
   noise_I = brainmass.OUProcess(in_size=1, sigma=0.5 * u.Hz, tau=20. * u.ms)
   nmm.noise_E = noise_E
   nmm.noise_I = noise_I

   # 3) Simulate a few steps
   ts = []
   for _ in range(1000):
       rE = nmm.update()  # returns updated excitatory activity
       ts.append(rE)

   # 4) Map node observable to a BOLD signal (per region)
   bold = brainmass.BOLDSignal(in_size=1)
   bold.init_state()
   for r in ts:
       bold.update(z=r)  # pass a neural activity proxy (e.g., rate)
   y_bold = bold.bold()  # Quantity with BOLD units

   # 5) EEG/MEG sensor mapping via a lead-field model
   R, M = 1, 64  # regions, sensors
   L = jnp.ones((R, M)) * (u.volt / (u.nA * u.meter))  # EEG lead field
   lead = brainmass.EEGLeadFieldModel(in_size=R, out_size=M, L=L)
   # Suppose s is your region dipole moment (nA*m). If the NMM output is in mV,
   # provide a biophysical scale (nA*m/mV) when constructing the lead-field model.
   # y = lead(s)  # shape (T, M) with EEG units


Node dynamics
-------------

These are the core neural population models. Each class derives from
``brainstate.nn.Dynamics`` and follows a consistent API:

- Call ``.init_state(batch_size=None)`` before simulation.
- Advance dynamics via ``.update(...)``; most models return a convenient observable
  (e.g., excitatory rate or an EEG-like proxy).
- Internal variables are accessible as hidden states, e.g., ``model.rE.value``.
- Units are enforced with ``brainunit``; time constants are typically in ``u.ms`` or ``u.s``,
  firing rates in ``u.Hz``, and membrane potentials in ``u.mV``.

Example (Wilson–Cowan):

.. code-block:: python

   wc = brainmass.WilsonCowanModel(in_size=(10,))  # 10 regions
   wc.init_state()
   for _ in range(100):
       rE = wc.update(rE_ext=0.2, rI_ext=0.1)  # add external drive

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    JansenRitModel
    WilsonCowanModel
    WongWangModel
    FitzHughNagumoModel
    ThresholdLinearModel
    KuramotoModel
    HopfOscillator
    VanDerPolOscillator
    StuartLandauOscillator



Noise processes
---------------

Noise processes generate unit-safe stochastic drive you can add to node models.

- ``GaussianNoise``/``WhiteNoise``: i.i.d. noise at each call (no internal state).
- ``OUProcess``: stateful Ornstein–Uhlenbeck process with decay time ``tau``.
- ``BrownianNoise``: integrated white noise (random walk).
- ``ColoredNoise``: 1/f^beta spectral shaping; preset subclasses: ``PinkNoise``,
  ``BlueNoise``, ``VioletNoise``.

Attach noise to a model by passing instances into the constructor or assigning the
attributes, e.g., ``WilsonCowanModel(noise_E=..., noise_I=...)``.

.. code-block:: python

   ou = brainmass.OUProcess(in_size=(10,), sigma=0.2 * u.Hz, tau=50. * u.ms)
   wc = brainmass.WilsonCowanModel(in_size=(10,), noise_E=ou)
   wc.init_state()
   wc.update()  # OU noise is added internally each step

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



Node coupling
-------------

Coupling maps activity from source nodes to target nodes via a connectivity matrix.
Two forms are provided in both functional and modular variants:

- Diffusive coupling: ``k * sum_j conn[i, j] * (x_{i,j} - y_i)``
- Additive coupling: ``k * sum_j conn[i, j] * x_{i,j}``

Shapes and conventions:

- ``conn``: ``(N_out, N_in)`` or flattened ``(N_out*N_in,)``.
- Source values ``x`` must broadcast to ``(..., N_out, N_in)`` or flattened.
- Target values ``y`` broadcast to ``(..., N_out)``.
- Returns an array with shape ``(..., N_out)`` (unit-safe if inputs carry units).

For imperative use inside a composite model, prefer the class-based variants and
``brainstate.nn.Prefetch`` to wire states across modules.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    DiffusiveCoupling
    AdditiveCoupling
    diffusive_coupling
    additive_coupling



Forward models
--------------

Forward models map region-level neural observables or dipole moments to sensor space.
They are unit-aware and accept either plain arrays with explicit unit arguments or
``brainunit.Quantity`` inputs with attached units.

Common workflows:

- BOLD: integrate hemodynamics per region via ``BOLDSignal.update(z=...)`` then
  read out with ``.bold()``.
- EEG/MEG: construct a lead-field model with ``L`` of shape ``(R, M)`` (regions×sensors).
  Provide a scale to convert your NMM observable (e.g., mV) to dipole moment (nA·m),
  or pass dipoles directly. The call returns ``(T, M)`` sensor signals.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

    BOLDSignal
    LeadFieldModel
    EEGLeadFieldModel
    MEGLeadFieldModel



Parameters
----------

Unit-safe, trainable parameters and helpers.

- ``ArrayParam`` wraps an array/quantity with an optional bijective transform
  (e.g., softplus) so you can optimize in unconstrained space while enforcing
  positivity or range constraints on the exposed ``.data``.

Example:

.. code-block:: python

   import braintools
   p = brainmass.ArrayParam(1.0 * u.ms, transform=braintools.SoftplusTransform())
   # Optimize p.value in ℝ; read/write p.data to get the constrained parameter

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ArrayParam


