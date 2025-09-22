# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable

import brainstate
import braintools
import brainunit as u

from ._typing import Initializer
from .noise import Noise

__all__ = [
    'VanDerPolOscillator',
]


class VanDerPolOscillator(brainstate.nn.Dynamics):
    r"""Van der Pol oscillator (two-dimensional form).

     In the study of dynamical systems, the van der Pol oscillator
    (named for Dutch physicist Balthasar van der Pol) is a non-conservative,
    oscillating system with non-linear damping. It evolves in time according
    to the second-order differential equation

    $$
    {d^{2}x \over dt^{2}}-\mu (1-x^{2}){dx \over dt}+x=0
    $$

    where $x$ is the position coordinate—which is a function of the time $t$—
    and $\mu$ is a scalar parameter indicating the nonlinearity and the
    strength of the damping.

    Implements the Van der Pol oscillator using the Liénard transformation
    that yields the planar system

    .. math::

        \dot x = \mu\,\left(x - \tfrac{1}{3}x^3 - y\right) + I_x(t),

    .. math::

        \dot y = \frac{1}{\mu}\,x + I_y(t),

    where :math:`x` is the state (often interpreted as position or activation),
    :math:`y` is the auxiliary variable, and :math:`\mu > 0` controls the
    nonlinearity and damping. The model exhibits a stable limit cycle for any
    :math:`\mu > 0`.

    Another commonly used form based on the transformation $y={\dot {x}}$ leads to:

    $$
    {\dot {x}}=y
    $$

    $$
    {\dot {y}}=\mu (1-x^{2})y-x
    $$

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node/population. Can be an ``int`` or a tuple of
        ``int``. All parameters are broadcastable to this shape.
    mu : Initializer, optional
        Nonlinearity/damping parameter (dimensionless). Default is ``1.0``.
    noise_x : Noise or None, optional
        Additive noise process for the :math:`x`-equation. If provided, called
        each update and added to ``x_inp``. Default is ``None``.
    noise_y : Noise or None, optional
        Additive noise process for the :math:`y`-equation. If provided, called
        each update and added to ``y_inp``. Default is ``None``.
    init_x : Callable, optional
        Initializer for the state ``x``. Default is
        ``brainstate.init.Uniform(0, 0.05)``.
    init_y : Callable, optional
        Initializer for the state ``y``. Default is
        ``brainstate.init.Uniform(0, 0.05)``.
    method : str, optional
        Time stepping method. One of ``'exp_euler'`` (exponential Euler; default)
        or any method supported under ``braintools.quad`` (e.g., ``'rk4'``,
        ``'midpoint'``, ``'heun'``, ``'euler'``).

    Attributes
    ----------
    x : brainstate.HiddenState
        State container for :math:`x` (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.
    y : brainstate.HiddenState
        State container for :math:`y` (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dx` and :meth:`dy` carry unit
      ``1/ms`` so that a step size ``dt`` with unit ``ms`` is consistent.
    - For ``method='exp_euler'`` the update uses ``brainstate.nn.exp_euler_step``.
      Otherwise, it dispatches to the corresponding routine in
      ``braintools.quad``.

    References
    ----------
    - van der Pol, B. (1926). On “relaxation-oscillations”. The London,
      Edinburgh, and Dublin Philosophical Magazine and Journal of Science,
      2(11), 978–992.
    - Kaplan, D., & Glass, L. (1995). Understanding Nonlinear Dynamics.
      Springer (pp. 240–244).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # parameters
        mu: Initializer = 1.0,

        # noise parameters
        noise_x: Noise = None,
        noise_y: Noise = None,

        # other parameters
        init_x: Callable = brainstate.init.Uniform(0, 0.05),
        init_y: Callable = brainstate.init.Uniform(0, 0.05),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        # model parameters
        self.mu = brainstate.init.param(mu, self.varshape)

        # initializers
        assert isinstance(noise_x, Noise) or noise_x is None, "noise_x must be a Noise instance or None."
        assert isinstance(noise_y, Noise) or noise_y is None, "noise_y must be a Noise instance or None."
        assert callable(noise_x), "noise_x must be a callable."
        assert callable(noise_y), "noise_y must be a callable."
        self.init_x = init_x
        self.init_y = init_y
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Initialize model states ``x`` and ``y``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.x = brainstate.HiddenState(brainstate.init.param(self.init_x, self.varshape, batch_size))
        self.y = brainstate.HiddenState(brainstate.init.param(self.init_y, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset model states ``x`` and ``y`` using the initializers.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension for reinitialization. If ``None``, keeps
            current batch shape but resets values. Default is ``None``.
        """
        self.x.value = brainstate.init.param(self.init_x, self.varshape, batch_size)
        self.y.value = brainstate.init.param(self.init_y, self.varshape, batch_size)

    def dx(self, x, y, inp):
        """Right-hand side for the state ``x``.

        Parameters
        ----------
        x : array-like
            Current value of ``x`` (dimensionless).
        y : array-like
            Current value of ``y`` (dimensionless), broadcastable to ``x``.
        inp : array-like or scalar
            External input to ``x`` (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        return self.mu * (x - x ** 3 / 3 - y) / u.ms + inp / u.ms

    def dy(self, y, x, inp=0.):
        """Right-hand side for the state ``y``.

        Parameters
        ----------
        y : array-like
            Current value of ``y`` (dimensionless).
        x : array-like
            Current value of ``x`` (dimensionless), broadcastable to ``y``.
        inp : array-like or scalar, optional
            External input to ``y`` (includes noise if enabled). Default is
            ``0.``.

        Returns
        -------
        array-like
            Time derivative ``dy/dt`` with unit ``1/ms``.
        """
        return (x / self.mu + inp) / u.ms

    def derivative(self, state, t, x_inp, y_inp):
        """Vector field for ODE integrators.

        This packs :meth:`dx` and :meth:`dy` into a single callable of the form
        ``f(state, t, x_inp, y_inp)`` to be used by ``braintools.quad``
        integrators when ``method != 'exp_euler'``.

        Parameters
        ----------
        state : tuple of array-like
            Current state as ``(x, y)``.
        t : array-like or scalar
            Current time (ignored in the autonomous dynamics).
        x_inp : array-like or scalar
            External input to ``x`` passed through to :meth:`dx`.
        y_inp : array-like or scalar
            External input to ``y`` passed through to :meth:`dy`.

        Returns
        -------
        tuple of array-like
            Derivatives as ``(dx/dt, dy/dt)`` each with unit ``1/ms``.
        """
        V, w = state
        dVdt = self.dx(V, w, x_inp)
        dwdt = self.dy(w, V, y_inp)
        return (dVdt, dwdt)

    def update(self, x_inp=None, y_inp=None):
        """Advance the oscillator by one time step.

        Parameters
        ----------
        x_inp : array-like or scalar or None, optional
            External input to ``x``. If ``None``, treated as zero. If
            ``noise_x`` is set, its output is added. Default is ``None``.
        y_inp : array-like or scalar or None, optional
            External input to ``y``. If ``None``, treated as zero. If
            ``noise_y`` is set, its output is added. Default is ``None``.

        Returns
        -------
        array-like
            The updated state ``x`` with the same shape as the internal state.

        Notes
        -----
        - For ``method='exp_euler'`` uses ``brainstate.nn.exp_euler_step`` to
          update each component.
        - Otherwise, dispatches to the corresponding integrator under
          ``braintools.quad`` (e.g., RK4), using :meth:`derivative`.
        """
        x_inp = 0. if x_inp is None else x_inp
        y_inp = 0. if y_inp is None else y_inp
        if self.noise_x is not None:
            x_inp = x_inp + self.noise_x()
        if self.noise_y is not None:
            y_inp = y_inp + self.noise_y()
        if self.method == 'exp_euler':
            x = brainstate.nn.exp_euler_step(self.dx, self.x.value, self.y.value, x_inp)
            y = brainstate.nn.exp_euler_step(self.dy, self.y.value, self.x.value, y_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            x, y = method(self.derivative, (self.x.value, self.y.value), 0 * u.ms, x_inp, y_inp)
        self.x.value = x
        self.y.value = y
        return x

