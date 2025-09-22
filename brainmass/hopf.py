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
import brainunit as u

from .noise import Noise

__all__ = [
    'HopfModel',
]


class HopfModel(brainstate.nn.Dynamics):
    r"""Normal-form Hopf oscillator (two-dimensional rate model).

    This model implements the supercritical Hopf normal form for a single node
    in terms of its real and imaginary components, often used as a simple
    mesoscopic model of oscillatory neural population activity.

    The complex form is

    .. math::
        \frac{dz}{dt} = (a + i\,\omega)\,z - \beta\,|z|^{2} z + I_{\text{ext}}(t),

    where :math:`z = x + i y` and :math:`|z|^2 = x^2 + y^2`. In real variables:

    .. math::
        \begin{aligned}
        \dot x &= (a - \beta r)\,x - \omega\,y + \text{coupled}_x + I_x(t),\\
        \dot y &= (a - \beta r)\,y + \omega\,x + \text{coupled}_y + I_y(t), \\
        r &= x^2 + y^2.
        \end{aligned}

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. Can be an int or tuple of ints.
    a : brainstate.typing.ArrayLike, optional
        Bifurcation parameter (dimensionless). For ``a > 0`` the system exhibits
        a stable limit cycle; for ``a < 0`` the origin is a stable focus.
        Broadcastable to ``in_size``. Default is ``0.25``.
    w : brainstate.typing.ArrayLike, optional
        Angular frequency :math:`\omega` (dimensionless in this implementation).
        Broadcastable to ``in_size``. Default is ``0.2``.
    K_gl : brainstate.typing.ArrayLike, optional
        Global coupling gain (dimensionless), included for convenience when used
        in networked settings. Not applied directly in the local node dynamics.
        Broadcastable to ``in_size``. Default is ``1.0``.
    beta : brainstate.typing.ArrayLike, optional
        Nonlinear saturation coefficient (dimensionless) setting the limit-cycle
        amplitude (approximately :math:`\sqrt{a/\beta}` when ``a>0``).
        Broadcastable to ``in_size``. Default is ``1.0``.
    noise_x : Noise or None, optional
        Additive noise process to ``x``. If provided, called each step and added
        to ``ext_x``. Default is ``None``.
    noise_y : Noise or None, optional
        Additive noise process to ``y``. If provided, called each step and added
        to ``ext_y``. Default is ``None``.

    Attributes
    ----------
    x : brainscale.ETraceState
        State container for the real component ``x``.
    y : brainscale.ETraceState
        State container for the imaginary component ``y``.

    Notes
    -----
    Time derivatives returned by ``dx`` and ``dy`` carry unit ``1/ms`` so that
    an explicit (exponential) Euler integrator with time step ``dt`` having
    unit ``ms`` evolves the state consistently with units.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        a: brainstate.typing.ArrayLike = 0.25,  # Hopf bifurcation parameter
        w: brainstate.typing.ArrayLike = 0.2,  # Oscillator frequency
        K_gl: brainstate.typing.ArrayLike = 1.0,  # global coupling strength
        beta: brainstate.typing.ArrayLike = 1.0,  # nonlinear saturation coefficient

        # noise
        noise_x: Noise = None,
        noise_y: Noise = None,

        # initialization
        init_x: Callable = brainstate.init.ZeroInit(),
        init_y: Callable = brainstate.init.ZeroInit(),
    ):
        super().__init__(in_size=in_size)

        self.a = brainstate.init.param(a, self.varshape)
        self.w = brainstate.init.param(w, self.varshape)
        self.K_gl = brainstate.init.param(K_gl, self.varshape)
        self.beta = brainstate.init.param(beta, self.varshape)
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.init_x = init_x
        self.init_y = init_y

    def init_state(self, batch_size=None, **kwargs):
        """Initialize oscillator states to zero.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.x = brainstate.HiddenState(brainstate.init.param(self.init_x, self.varshape, batch_size))
        self.y = brainstate.HiddenState(brainstate.init.param(self.init_y, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset oscillator states to zero.

        Parameters
        ----------
        batch_size : int or None, optional
            Batch dimension to reset with. If ``None``, keeps current batch
            shape but resets values. Default is ``None``.
        """
        self.x.value = brainstate.init.param(self.init_x, self.varshape, batch_size)
        self.y.value = brainstate.init.param(self.init_y, self.varshape, batch_size)

    def dx(self, x, y, inp):
        """Right-hand side for ``x``.

        Parameters
        ----------
        x : array-like
            Current real component.
        y : array-like
            Current imaginary component (broadcastable to ``x``).
        inp : array-like or scalar
            External input to ``x`` (includes coupling and noise).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        r = x ** 2 + y ** 2
        dx_dt = (self.a - self.beta * r) * x - self.w * y + inp
        return dx_dt / u.ms

    def dy(self, y, x, inp):
        """Right-hand side for ``y``.

        Parameters
        ----------
        y : array-like
            Current imaginary component.
        x : array-like
            Current real component (broadcastable to ``y``).
        inp : array-like or scalar
            External input to ``y`` (includes coupling and noise).

        Returns
        -------
        array-like
            Time derivative ``dy/dt`` with unit ``1/ms``.
        """
        r = x ** 2 + y ** 2
        dy_dt = (self.a - self.beta * r) * y + self.w * x + inp
        return dy_dt / u.ms

    def update(self, ext_x=None, ext_y=None):
        """Advance the oscillator by one time step.

        Parameters
        ----------
        ext_x : array-like or scalar or None, optional
            External drive to ``x`` (in addition to ``coupled_x``). If ``None``,
            treated as zero. If ``noise_x`` is set, its output is added.
        ext_y : array-like or scalar or None, optional
            External drive to ``y`` (in addition to ``coupled_y``). If ``None``,
            treated as zero. If ``noise_y`` is set, its output is added.

        Notes
        -----
        Uses an exponential-Euler step via ``brainstate.nn.exp_euler_step`` for
        each component.
        """
        ext_x = 0. if ext_x is None else ext_x
        ext_y = 0. if ext_y is None else ext_y

        # add noise
        if self.noise_x is not None:
            assert isinstance(self.noise_y, Noise), "noise_y must be an Noise if noise_x is not None"
            ext_x += self.noise_x()

        if self.noise_y is not None:
            assert isinstance(self.noise_x, Noise), "noise_x must be an v if noise_y is not None"
            ext_y += self.noise_y()

        x_next = brainstate.nn.exp_euler_step(self.dx, self.x.value, self.y.value, ext_x)
        y_next = brainstate.nn.exp_euler_step(self.dy, self.y.value, self.x.value, ext_y)
        self.x.value = x_next
        self.y.value = y_next
        return x_next
