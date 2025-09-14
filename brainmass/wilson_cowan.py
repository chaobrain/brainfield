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
import jax.numpy as jnp

from .noise import Noise

__all__ = [
    'WilsonCowanModel',
]


class WilsonCowanModel(brainstate.nn.Dynamics):
    r"""
    Wilson-Cowan neural mass model for excitatory-inhibitory population dynamics.
    
    This model describes the dynamics of two interacting neural populations 
    (excitatory and inhibitory) and is fundamental for understanding neural 
    oscillations, bistability, and other emergent network behaviors in cortical circuits.

    Mathematical Description:
    ========================
    
    The model is governed by two coupled differential equations:
    
    .. math::
        \tau_E \frac{da_E}{dt} = -a_E(t) + [1 - r \cdot a_E(t)] F_E(w_{EE} a_E(t) - w_{EI} a_I(t) + I_E(t))
        
    .. math::
        \tau_I \frac{da_I}{dt} = -a_I(t) + [1 - r \cdot a_I(t)] F_I(w_{IE} a_E(t) - w_{II} a_I(t) + I_I(t))
    
    where the sigmoidal transfer function is:
    
    .. math::
        F_j(x) = \frac{1}{1 + e^{-a_j(x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}}, \quad j \in \{E, I\}

    Parameters
    ==========
    - **tau_E, tau_I** (ms): Time constants controlling the response speed of excitatory 
      and inhibitory populations
    - **a_E, a_I** (dimensionless): Gain parameters controlling the steepness of the 
      activation functions
    - **theta_E, theta_I** (dimensionless): Threshold parameters for population activation
    - **wEE** (dimensionless): Excitatory-to-excitatory recurrent connection strength
    - **wEI** (dimensionless): Inhibitory-to-excitatory connection strength
    - **wIE** (dimensionless): Excitatory-to-inhibitory connection strength  
    - **wII** (dimensionless): Inhibitory-to-inhibitory connection strength
    - **r** (dimensionless): Refractory parameter affecting maximum activation levels

    State Variables
    ==============
    - **rE**: Excitatory population activation (dimensionless, normalized firing rate)
    - **rI**: Inhibitory population activation (dimensionless, normalized firing rate)

    References
    ==========
    Wilson, H.R. & Cowan, J.D. "Excitatory and inhibitory interactions in localized 
    populations of model neurons." Biophysical Journal 12, 1–24 (1972).

    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: brainstate.typing.ArrayLike = 1. * u.ms,  # excitatory time constant (ms)
        a_E: brainstate.typing.ArrayLike = 1.2,  # excitatory gain (dimensionless)
        theta_E: brainstate.typing.ArrayLike = 2.8,  # excitatory firing threshold (dimensionless)

        # Inhibitory parameters
        tau_I: brainstate.typing.ArrayLike = 1. * u.ms,  # inhibitory time constant (ms)
        a_I: brainstate.typing.ArrayLike = 1.,  # inhibitory gain (dimensionless)
        theta_I: brainstate.typing.ArrayLike = 4.0,  # inhibitory firing threshold (dimensionless)

        # Connection parameters
        wEE: brainstate.typing.ArrayLike = 12.,  # local E-E coupling (dimensionless)
        wIE: brainstate.typing.ArrayLike = 4.,  # local E-I coupling (dimensionless)
        wEI: brainstate.typing.ArrayLike = 13.,  # local I-E coupling (dimensionless)
        wII: brainstate.typing.ArrayLike = 11.,  # local I-I coupling (dimensionless)

        # Refractory parameter
        r: brainstate.typing.ArrayLike = 1.,  # refractory parameter (dimensionless)

        # noise
        noise_E: Noise = None,  # excitatory noise process
        noise_I: Noise = None,  # inhibitory noise process

        # initialization
        rE_init: Callable = brainstate.init.ZeroInit(),
        rI_init: Callable = brainstate.init.ZeroInit(),
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
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState(brainstate.init.param(self.rE_init, self.varshape, batch_size))
        self.rI = brainstate.HiddenState(brainstate.init.param(self.rI_init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.rE.value = brainstate.init.param(self.rE_init, self.varshape, batch_size)
        self.rI.value = brainstate.init.param(self.rI_init, self.varshape, batch_size)

    def F(self, x, a, theta):
        return 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))

    def drE(self, rE, rI, ext):
        """Differential equation for excitatory population."""
        xx = self.wEE * rE - self.wIE * rI + ext
        return (-rE + (1 - self.r * rE) * self.F(xx, self.a_E, self.theta_E)) / self.tau_E

    def drI(self, rI, rE, ext):
        """Differential equation for inhibitory population."""
        xx = self.wEI * rE - self.wII * rI + ext
        return (-rI + (1 - self.r * rI) * self.F(xx, self.a_I, self.theta_I)) / self.tau_I

    def update(self, rE_ext=None, rI_ext=None):
        """Update the model state for one time step.
        
        Args:
            rE_ext: External input to excitatory population
            rI_ext: External input to inhibitory population
            
        Returns:
            Current excitatory population activation
        """
        # excitatory input
        rE_ext = 0. if rE_ext is None else rE_ext
        rI_ext = 0. if rI_ext is None else rI_ext
        if self.noise_E is not None:
            rE_ext = rE_ext + self.noise_E()
        rE_ext = self.sum_delta_inputs(rE_ext, label='E')

        # inhibitory input
        if self.noise_I is not None:
            rI_ext = rI_ext + self.noise_I()
        rI_ext = self.sum_delta_inputs(rI_ext, label='I')

        # update the state variables
        rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, rE_ext)
        rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, rI_ext)
        self.rE.value = rE
        self.rI.value = rI
        return rE
