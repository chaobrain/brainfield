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


import brainstate
import jax.numpy as jnp

from .noise import OUProcess

__all__ = [
    'WilsonCowanModel',
]


class WilsonCowanModel(brainstate.nn.Dynamics):
    r"""
    The Wilson-Cowan model is a mathematical model of neural dynamics
    that describes the interaction between excitatory and inhibitory
    populations of neurons. It is commonly used to study the dynamics
    of neural networks and their response to external inputs.

    $$
    \begin{aligned}
    \tau_e \frac{\mathrm{~d} a_e}{\mathrm{~d} t} & =-a_e(t)+\left[1-r_e a_e(t)\right] F_e\left(w_{e e} a_e(t)-w_{e i} a_i(t)+I_e(t)\right) \\
    \tau_i \frac{\mathrm{~d} a_i}{\mathrm{~d} t} & =-a_i(t)+\left[1-r_i a_i(t)\right] F_i\left(w_{i e} a_e(t)-w_{i i} a_i(t)+I_i(t)\right)
    \end{aligned}
    $$

    $$
    F_j(x)=\frac{1}{1+e^{-\gamma_j\left(x-\theta_j\right)}}, \quad j=e, i
    $$

    - $r_E(t)$ represents the average activation (or firing rate) of the excitatory population at time $t$,
    - $r_I(t)$ the activation (or firing rate) of the inhibitory population,
    - The parameters $\\tau_E$ and $\\tau_I$ control the timescales of the dynamics of each population.
    - Connection strengths are given by: $w_{EE}$ (E $\\rightarrow$ E), $w_{EI}$ (I $\\rightarrow$ E),
      $w_{IE}$ (E $\\rightarrow$ I), and $w_{II}$ (I $\\rightarrow$ I).
    - The terms $w_{EI}$ and $w_{IE}$ represent connections from inhibitory to excitatory
      population and vice versa, respectively.
    - The transfer functions (or F-I curves) $F_E(x;a_E,\\theta_E)$ and $F_I(x;a_I,\\theta_I)$
      can be different for the excitatory and the inhibitory populations.

    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: brainstate.typing.ArrayLike = 1.,  # excitatory time constant
        a_E: brainstate.typing.ArrayLike = 1.2,  # excitatory gain
        theta_E: brainstate.typing.ArrayLike = 2.8,  # excitatory firing threshold

        # Inhibitory parameters
        tau_I: brainstate.typing.ArrayLike = 1.,  # inhibitory time constant
        a_I: brainstate.typing.ArrayLike = 1.,  # inhibitory gain
        theta_I: brainstate.typing.ArrayLike = 4.0,  # inhibitory firing threshold

        # Connection parameters
        wEE: brainstate.typing.ArrayLike = 12.,  # local E-E coupling
        wIE: brainstate.typing.ArrayLike = 4.,  # local E-I coupling
        wEI: brainstate.typing.ArrayLike = 13.,  # local I-E coupling
        wII: brainstate.typing.ArrayLike = 11.,  # local I-I coupling

        # Refractory parameter
        r: brainstate.typing.ArrayLike = 1.,

        # noise
        noise_E: OUProcess = None,
        noise_I: OUProcess = None,
    ):
        super().__init__(in_size=in_size)

        self.a_E = brainstate.init.param(a_E, self.varshape)
        self.a_I = brainstate.init.param(a_I, self.varshape)
        self.tau_E = brainstate.init.param(tau_E, self.varshape)
        self.tau_I = brainstate.init.param(tau_I, self.varshape)
        self.theta_E = brainstate.init.param(theta_E, self.varshape)
        self.theta_I = brainstate.init.param(theta_I, self.varshape)
        self.wEE = brainstate.init.param(wEE, self.varshape)
        self.wIE = brainstate.init.param(wIE, self.varshape)
        self.wEI = brainstate.init.param(wEI, self.varshape)
        self.wII = brainstate.init.param(wII, self.varshape)
        self.r = brainstate.init.param(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I

    def init_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.rE = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.rI = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))

    def reset_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.rE.value = brainstate.init.param(jnp.zeros, size)
        self.rI.value = brainstate.init.param(jnp.zeros, size)

    def F(self, x, a, theta):
        return 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))

    def drE(self, rE, rI, ext):
        xx = self.wEE * rE - self.wIE * rI + ext
        return (-rE + (1 - self.r * rE) * self.F(xx, self.a_E, self.theta_E)) / self.tau_E

    def drI(self, rI, rE, ext):
        xx = self.wEI * rE - self.wII * rI + ext
        return (-rI + (1 - self.r * rI) * self.F(xx, self.a_I, self.theta_I)) / self.tau_I

    def update(self, rE_ext=None, rI_ext=None):
        # excitatory input
        rE_ext = 0. if rE_ext is None else rE_ext
        rI_ext = 0. if rI_ext is None else rI_ext
        if self.noise_E is not None:
            assert isinstance(self.noise_I, OUProcess), "noise_I must be an OUProcess if noise_E is None"
            rE_ext += self.noise_E()
        rE_ext = self.sum_delta_inputs(rE_ext, label='E')

        # inhibitory input
        if self.noise_E is not None:
            assert isinstance(self.noise_I, OUProcess), "noise_I must be an OUProcess if noise_E is None"
            rI_ext += self.noise_I()
        rI_ext = self.sum_delta_inputs(rI_ext, label='I')

        # update the state variables
        rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, rE_ext)
        rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, rI_ext)
        self.rE.value = rE
        self.rI.value = rI
        return rE
