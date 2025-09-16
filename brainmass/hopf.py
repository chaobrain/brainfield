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


import brainscale
import brainstate
import brainunit as u
import jax.numpy as jnp

from .noise import Noise

__all__ = [
    'HopfModel',
]


class HopfModel(brainstate.nn.Dynamics):
    r"""
    The adaptive linear-nonlinear (aln) cascade model is a low-dimensional 
    population model of spiking neural networks. Mathematically, 
    it is a dynamical system of non-linear ODEs. 
    The dynamical variables of the system simulated in the aln model describe the average 
    firing rate and other macroscopic variables of a randomly connected, 
    delay-coupled network of excitatory and inhibitory adaptive exponential 
    integrate-and-fire neurons (AdEx) with non-linear synaptic currents.
    
    .. math::
    \frac{\mathrm{d}z}{\mathrm{d}t} = (a + \mathrm{i}\omega)\,z - \beta\,|z|^{2}z + I_{\text{ext}}(t)
    with :math:`|z|^{2} = x^{2} + y^{2}`.  
    Split into real/imaginary parts the system reads

    .. math::
    \begin{aligned}
    \dot x &= (a - \beta\,r)\,x - \omega\,y + coupled_x + I_{x}(t) \\
    \dot y &= (a - \beta\,r)\,y + \omega\,x + coupled_y + I_{y}(t)
    \end{aligned}
    \quad\text{with}\quad r = x^{2} + y^{2}.

    Parameters
    ----------
    x, y : dynamical variables
    Real and imaginary components of the oscillator (firing-rate analogue).
    a : bifurcation parameter
    > 0  →  limit-cycle (oscillatory);  ≤ 0  →  stable focus (silent).
    ω : angular frequency
    Intrinsic oscillation frequency (rad s⁻¹).
    β : nonlinear saturation coefficient
    Sets the limit-cycle amplitude (√ a/β ).
    K_gl : global coupling gain
    Scales diffusive input from other nodes.
    I_x, I_y : external inputs
    Additive currents (noise, coupling, stimulus) acting on x and y.
    coupled_x, coupled_y : coupling mechanism for neural network modules
 
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

    ):
        super().__init__(in_size=in_size)

        self.a = brainstate.init.param(a, self.varshape)
        self.w = brainstate.init.param(w, self.varshape)
        self.K_gl = brainstate.init.param(K_gl, self.varshape)
        self.beta = brainstate.init.param(beta, self.varshape)
        self.noise_x = noise_x
        self.noise_y = noise_y

    def init_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.x = brainscale.ETraceState(jnp.zeros(size))
        self.y = brainscale.ETraceState(jnp.zeros(size))

    def reset_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        # initial values of the state variables
        self.x.value = brainstate.init.param(jnp.zeros, size)
        self.y.value = brainstate.init.param(jnp.zeros, size)

    def dx(self, x, y, inp):
        r = x ** 2 + y ** 2
        dx_dt = (self.a - self.beta * r) * x - self.w * y + inp
        return dx_dt / u.ms

    def dy(self, y, x, inp):
        r = x ** 2 + y ** 2
        dy_dt = (self.a - self.beta * r) * y + self.w * x + inp
        return dy_dt / u.ms

    def update(self, coupled_x, coupled_y, ext_x=None, ext_y=None):
        ext_x = 0. if ext_x is None else ext_x
        ext_y = 0. if ext_y is None else ext_y

        # add noise
        if self.noise_x is not None:
            assert isinstance(self.noise_y, Noise), "noise_y must be an Noise if noise_x is not None"
            ext_x += self.noise_x()

        if self.noise_y is not None:
            assert isinstance(self.noise_x, Noise), "noise_x must be an v if noise_y is not None"
            ext_y += self.noise_y()

        x_next = brainstate.nn.exp_euler_step(self.dx, self.x.value, self.y.value, coupled_x + ext_x)
        y_next = brainstate.nn.exp_euler_step(self.dy, self.y.value, self.x.value, coupled_y + ext_y)
        self.x.value = x_next
        self.y.value = y_next
        return x_next, y_next
