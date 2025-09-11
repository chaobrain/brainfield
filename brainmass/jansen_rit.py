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

from typing import Union, Callable

import brainunit as u
import jax.numpy as jnp

import brainstate
from brainstate.nn import exp_euler_step

__all__ = [
    'JansenRitModel',
]


class JansenRitModel(brainstate.nn.Dynamics):
    r"""
    Jansen-Rit neural mass model.

    Jansen-Rit neural mass model is governed by the system of coupled differential equations.
    In this system, indices 0, 2, and 4 represent the pyramidal, excitatory, and inhibitory
    neuron populations, respectively. The output signal $y(t)=a_2y_2-a_4y_4$ represents the
    difference between the pyramidal’s excitatory and inhibitory postsynaptic potentials.
    This value is a proxy for EEG sources because EEG is thought to reflect mainly the
    postsynaptic potentials in the apical dendrites of pyramidal cells [1]. Table 1 shows
    the default parameter values and ranges used in our simulations based on physiologically
    plausible values from previous studies [2, 3].

    $$
    \begin{aligned}
    & \dot{y}_{0}=y_1,\\
    & \dot{y_2}=y_3,\\
    & \dot{y_4}=y_5,\\
    &\dot{y}_{1}=A_eb_eS(I_p+a_2y_2-a_4y_4)-2b_ey_1-b_e^2y_0,\\
    &\dot{y}_{3}=A_eb_eS(a_1y_0)-2b_ey_3-b_e^2y_2,\\
    &\ddot{y}_{5}=A_ib_iS(I_i+a_3y_0)-2b_iy_5-b_i^2y_4.
    \end{aligned}
    $$

    The sigmoid function $S(v)$ translates the mean membrane potential $v$ of a specific population
    into its mean firing rate. It is expressed as:

    $$
    S(v)=S_{\max } \cdot \frac{1}{1+e^{-r\left(v-v_0\right)}}
    $$

    The parameters $v_0$ and $r$ regulate the midpoint and steepness of the sigmoidal curve,
    whereas $S_{\max }$ captures the maximal firing rate for the population. As $v$ increases,
    $S(v)$ gradually rises from 0 to $S_{\max }$, capturing the activation level of the neuron
    population. Default, we used $v_0=6, r=0.56$ and $S_{\max }=5$ spikes per second.

    $y_0$, $y_2$, and $y_4$ represent the average membrane potentials of the pyramidal,
    excitatory, and inhibitory neuron populations, respectively. They have the unit of mV.
    $y_1$, $y_3$, and $y_5$ are the first derivatives of $y_0$, $y_2$, and $y_4$,
    respectively, with the unit of mV/s. $I_p$ and $I_i$ are the external inputs to the
    excitatory and inhibitory populations, respectively, with the unit of mV. In this study,
    we set both $I_p$ and $I_i$ to zero.

    Standard parameter settings for Jansen-Rit model. Only the
    value from parameters with a range is estimated in this study::

    | Parameter | Description | Default | Range |
    | :--- | :--- | :--- | :--- |
    | $A_e$ | Excitatory gain | 3.25 mV | $2.6-9.75 \mathrm{mV}$ |
    | $A_i$ | Inhibitory gain | 22 mV | $17.6-110.0 \mathrm{mV}$ |
    | $b_e$ | Excit. time const. | $100 \mathrm{~s}^{-1}$ | $5-150 \mathrm{~s}^{-1}$ |
    | $b_i$ | Inhib. time const. | $50 \mathrm{~s}^{-1}$ | $25-75 \mathrm{~s}^{-1}$ |
    | C | Connect. const. | 135 | 65-1350 |
    | $a_1$ | Connect. param. | 1.0 | 0.5-1.5 |
    | $a_2$ | Connect. param. | 0.8 | 0.4-1.2 |
    | $a_3$ | Connect. param. | 0.25 | 0.125-0.375 |
    | $a_4$ | Connect. param. | 0.25 | 0.125-0.375 |
    | $v_{\text {max }}$ | Max firing rate | $5 \mathrm{~s}^{-1}$ | - |
    | $v_0$ | Firing threshold | 6 mV | - |
    | $r$ | Sigmoid steepness | 0.56 | - |


    References
    ----------
    [1] Nunez P L, Srinivasan R. Electric fields of the brain: the neurophysics of EEG[M]. Oxford university press, 2006.
    [2] Jansen B H, Rit V G. Electroencephalogram and visual evoked potential generation in a mathematical model of
        coupled cortical columns[J]. Biological cybernetics, 1995, 73(4): 357-366.
    [3] David O, Friston K J. A neural mass model for MEG/EEG:: coupling and neuronal dynamics[J]. NeuroImage, 2003, 20(3): 1743-1755.
    """

    def __init__(
        self,
        size: int,
        Ae: Union[brainstate.typing.ArrayLike, Callable] = 3.25 * u.mV,  # Excitatory gain
        Ai: Union[brainstate.typing.ArrayLike, Callable] = 22. * u.mV,  # Inhibitory gain
        be: Union[brainstate.typing.ArrayLike, Callable] = 100. / u.second,  # Excit. time const
        bi: Union[brainstate.typing.ArrayLike, Callable] = 50. / u.second,  # Inhib. time const.
        C: Union[brainstate.typing.ArrayLike, Callable] = 135.,  # Connect. const.
        a1: Union[brainstate.typing.ArrayLike, Callable] = 1.,  # Connect. param.
        a2: Union[brainstate.typing.ArrayLike, Callable] = 0.8,  # Connect. param.
        a3: Union[brainstate.typing.ArrayLike, Callable] = 0.25,  # Connect. param
        a4: Union[brainstate.typing.ArrayLike, Callable] = 0.25,  # Connect. param.
        s_max: Union[brainstate.typing.ArrayLike, Callable] = 5. / u.second,  # Max firing rate
        v0: Union[brainstate.typing.ArrayLike, Callable] = 6. * u.mV,  # Firing threshold
        r: Union[brainstate.typing.ArrayLike, Callable] = 0.56,  # Sigmoid steepness
    ):
        super().__init__(size)

        self.Ae = brainstate.init.param(Ae, self.varshape)
        self.Ai = brainstate.init.param(Ai, self.varshape)
        self.be = brainstate.init.param(be, self.varshape)
        self.bi = brainstate.init.param(bi, self.varshape)
        self.C = brainstate.init.param(C, self.varshape)
        self.a1 = brainstate.init.param(a1, self.varshape)
        self.a2 = brainstate.init.param(a2, self.varshape)
        self.a3 = brainstate.init.param(a3, self.varshape)
        self.a4 = brainstate.init.param(a4, self.varshape)
        self.s_max = brainstate.init.param(s_max, self.varshape)
        self.v0 = brainstate.init.param(v0, self.varshape)
        self.r = brainstate.init.param(r, self.varshape)

    def init_state(self, batch_size=None, **kwargs):
        dtype = brainstate.environ.dftype()
        size = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self.M = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV)
        self.Mv = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV / u.second)
        self.E = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV)
        self.Ev = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV / u.second)
        self.I = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV)
        self.Iv = brainstate.HiddenState(jnp.zeros(size, dtype=dtype) * u.mV / u.second)

    def reset_state(self, batch_size=None, **kwargs):
        dtype = brainstate.environ.dftype()
        size = self.varshape if batch_size is None else (batch_size, *self.varshape)
        self.M.value = jnp.zeros(size, dtype=dtype) * u.mV
        self.Mv.value = jnp.zeros(size, dtype=dtype) * u.mV / u.second
        self.E.value = jnp.zeros(size, dtype=dtype) * u.mV
        self.Ev.value = jnp.zeros(size, dtype=dtype) * u.mV / u.second
        self.I.value = jnp.zeros(size, dtype=dtype) * u.mV
        self.Iv.value = jnp.zeros(size, dtype=dtype) * u.mV / u.second

    def S(self, v):
        return self.s_max / (1 + jnp.exp(self.r * (self.v0 - v) / u.mV))

    def dmv(self, y1, y0, y2, y4, Ip):
        return self.Ae * self.be * self.S(Ip + self.a2 * y2 - self.a4 * y4) - 2 * self.be * y1 - self.be ** 2 * y0

    def dev(self, y3, y0, y2):
        return self.Ae * self.be * self.S(self.a1 * y0) - 2 * self.be * y3 - self.be ** 2 * y2

    def div(self, y5, y0, y4, Ii):
        return self.Ai * self.bi * self.S(self.a3 * y0 + Ii) - 2 * self.bi * y5 - self.bi ** 2 * y4

    def update(self, Ip=0. * u.mV, Ii=0. * u.mV):
        M = exp_euler_step(lambda y0, y1: y1, self.M.value, self.Mv.value)
        E = exp_euler_step(lambda y2, y3: y3, self.E.value, self.Ev.value)
        I = exp_euler_step(lambda y4, y5: y5, self.I.value, self.Iv.value)
        Mv = exp_euler_step(self.dmv, self.Mv.value, self.M.value, self.E.value, self.I.value, Ip)
        Ev = exp_euler_step(self.dev, self.Ev.value, self.M.value, self.E.value)
        Iv = exp_euler_step(self.div, self.Iv.value, self.M.value, self.I.value, Ii)
        self.M.value = M
        self.Mv.value = Mv
        self.E.value = E
        self.Ev.value = Ev
        self.I.value = I
        self.Iv.value = Iv
        return self.eeg()

    def eeg(self):
        return self.a2 * self.E.value - self.a4 * self.I.value
