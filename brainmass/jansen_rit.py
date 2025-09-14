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

import brainstate
import braintools
import brainunit as u
from brainstate.nn import exp_euler_step

from .noise import Noise

__all__ = [
    'JansenRitModel',
]


class Scale:
    def __init__(self, slope: float, fn: Callable = u.math.tanh):
        self.slope = slope
        self.fn = fn

    def __call__(self, x):
        x, unit = u.split_mantissa_unit(x)
        return u.maybe_decimal(self.slope * self.fn(x / self.slope) * unit)


class Identity:
    def __call__(self, x):
        return x


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
    S(v)=2*S_{\max } \cdot \frac{1}{1+e^{-r\left(v-v_0\right)}}
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

    Standard parameter settings for the Jansen-Rit model. Only parameters with a
    specified "Range" are estimated in this study.

    .. list-table::
       :widths: 12 30 14 18
       :header-rows: 1

       * - Parameter
         - Description
         - Default
         - Range
       * - Ae
         - Excitatory gain
         - 3.25 mV
         - 2.6-9.75 mV
       * - Ai
         - Inhibitory gain
         - 22 mV
         - 17.6-110.0 mV
       * - be
         - Excitatory time const.
         - 100 s^-1
         - 5-150 s^-1
       * - bi
         - Inhibitory time const.
         - 50 s^-1
         - 25-75 s^-1
       * - C
         - Connectivity constant
         - 135
         - 65-1350
       * - a1
         - Connectivity parameter
         - 1.0
         - 0.5-1.5
       * - a2
         - Connectivity parameter
         - 0.8
         - 0.4-1.2
       * - a3
         - Connectivity parameter
         - 0.25
         - 0.125-0.375
       * - a4
         - Connectivity parameter
         - 0.25
         - 0.125-0.375
       * - smax
         - Max firing rate
         - 2.5 s^-1
         - -
       * - v0
         - Firing threshold
         - 6 mV
         - -
       * - r
         - Sigmoid steepness
         - 0.56
         - -

    References
    ----------

    - [1] Nunez P L, Srinivasan R. Electric fields of the brain: the neurophysics of EEG[M]. Oxford university press, 2006.
    - [2] Jansen B H, Rit V G. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns[J]. Biological cybernetics, 1995, 73(4): 357-366.
    - [3] David O, Friston K J. A neural mass model for MEG/EEG:: coupling and neuronal dynamics[J]. NeuroImage, 2003, 20(3): 1743-1755.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        Ae: Union[brainstate.typing.ArrayLike, Callable] = 3.25 * u.mV,  # Excitatory gain
        Ai: Union[brainstate.typing.ArrayLike, Callable] = 22. * u.mV,  # Inhibitory gain
        be: Union[brainstate.typing.ArrayLike, Callable] = 100. * u.Hz,  # Excit. time const
        bi: Union[brainstate.typing.ArrayLike, Callable] = 50. * u.Hz,  # Inhib. time const.
        C: Union[brainstate.typing.ArrayLike, Callable] = 135.,  # Connect. const.
        a1: Union[brainstate.typing.ArrayLike, Callable] = 1.,  # Connect. param.
        a2: Union[brainstate.typing.ArrayLike, Callable] = 0.8,  # Connect. param.
        a3: Union[brainstate.typing.ArrayLike, Callable] = 0.25,  # Connect. param
        a4: Union[brainstate.typing.ArrayLike, Callable] = 0.25,  # Connect. param.
        s_max: Union[brainstate.typing.ArrayLike, Callable] = 2.5 * u.Hz,  # Max firing rate
        v0: Union[brainstate.typing.ArrayLike, Callable] = 6. * u.mV,  # Firing threshold
        r: Union[brainstate.typing.ArrayLike, Callable] = 0.56,  # Sigmoid steepness
        M_init: Callable = brainstate.init.ZeroInit(unit=u.mV),
        E_init: Callable = brainstate.init.ZeroInit(unit=u.mV),
        I_init: Callable = brainstate.init.ZeroInit(unit=u.mV),
        Mv_init: Callable = brainstate.init.ZeroInit(unit=u.mV / u.second),
        Ev_init: Callable = brainstate.init.ZeroInit(unit=u.mV / u.second),
        Iv_init: Callable = brainstate.init.ZeroInit(unit=u.mV / u.second),
        fr_scale: Callable = Identity(),
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,
        method: str = 'exp_euler'
    ):
        super().__init__(in_size)

        self.Ae = brainstate.init.param(Ae, self.varshape)
        self.Ai = brainstate.init.param(Ai, self.varshape)
        self.be = brainstate.init.param(be, self.varshape)
        self.bi = brainstate.init.param(bi, self.varshape)
        self.a1 = brainstate.init.param(a1, self.varshape)
        self.a2 = brainstate.init.param(a2, self.varshape)
        self.a3 = brainstate.init.param(a3, self.varshape)
        self.a4 = brainstate.init.param(a4, self.varshape)
        self.v0 = brainstate.init.param(v0, self.varshape)
        self.C = brainstate.init.param(C, self.varshape)
        self.r = brainstate.init.param(r, self.varshape)
        self.s_max = brainstate.init.param(s_max, self.varshape)

        assert callable(fr_scale), 'fr_scale must be a callable function'
        assert callable(M_init), 'M_init must be a callable function'
        assert callable(E_init), 'E_init must be a callable function'
        assert callable(I_init), 'I_init must be a callable function'
        assert callable(Mv_init), 'Mv_init must be a callable function'
        assert callable(Ev_init), 'Ev_init must be a callable function'
        assert callable(Iv_init), 'Iv_init must be a callable function'
        self.M_init = M_init
        self.E_init = E_init
        self.I_init = I_init
        self.Mv_init = Mv_init
        self.Ev_init = Ev_init
        self.Iv_init = Iv_init
        self.fr_scale = fr_scale
        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_M = noise_M
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.M = brainstate.HiddenState(brainstate.init.param(self.M_init, self.varshape, batch_size))
        self.E = brainstate.HiddenState(brainstate.init.param(self.E_init, self.varshape, batch_size))
        self.I = brainstate.HiddenState(brainstate.init.param(self.I_init, self.varshape, batch_size))
        self.Mv = brainstate.HiddenState(brainstate.init.param(self.Mv_init, self.varshape, batch_size))
        self.Ev = brainstate.HiddenState(brainstate.init.param(self.Ev_init, self.varshape, batch_size))
        self.Iv = brainstate.HiddenState(brainstate.init.param(self.Iv_init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.M.value = brainstate.init.param(self.M_init, self.varshape, batch_size)
        self.E.value = brainstate.init.param(self.E_init, self.varshape, batch_size)
        self.I.value = brainstate.init.param(self.I_init, self.varshape, batch_size)
        self.Mv.value = brainstate.init.param(self.Mv_init, self.varshape, batch_size)
        self.Ev.value = brainstate.init.param(self.Ev_init, self.varshape, batch_size)
        self.Iv.value = brainstate.init.param(self.Iv_init, self.varshape, batch_size)

    def S(self, v):
        # Sigmoid ranges from 0 to s_max, centered at v0
        return self.s_max / (1 + u.math.exp(self.r * (self.v0 - v) / u.mV))

    def dMv(self, Mv, M, E, I, inp):
        # Pyramidal population driven by the difference of PSPs (no extra C here)
        fr = self.S(E - I + inp)
        return self.Ae * self.be * self.fr_scale(fr) - 2 * self.be * Mv - self.be ** 2 * M

    def dEv(self, Ev, M, E, inp=0. * u.Hz):
        # Excitatory interneuron population: A*a*(p + C2*S(C1*M)) - 2*a*y' - a^2*y
        s_M = self.C * self.a2 * self.S(self.C * self.a1 * M)
        fr_total = self.fr_scale(inp + s_M)
        return self.Ae * self.be * fr_total - 2 * self.be * Ev - self.be ** 2 * E

    def dIv(self, Iv, M, I, inp):
        # Inhibitory interneuron population: B*b*(C4*S(C3*M)) - 2*b*y' - b^2*y
        s_M = self.C * self.a4 * self.S(self.C * self.a3 * M + inp)
        fr_total = self.fr_scale(s_M)
        return self.Ai * self.bi * fr_total - 2 * self.bi * Iv - self.bi ** 2 * I

    def derivative(self, state, t, M_inp, E_inp, I_inp):
        M, E, I, Mv, Ev, Iv = state
        dM = Mv
        dE = Ev
        dI = Iv
        dMv = self.dMv(Mv, M, E, I, M_inp)
        dEv = self.dEv(Ev, M, E, E_inp)
        dIv = self.dIv(Iv, M, I, I_inp)
        return (dM, dE, dI, dMv, dEv, dIv)

    def update(
        self,
        M_inp=0. * u.mV,
        E_inp=0. * u.Hz,
        I_inp=0. * u.mV,
    ):
        M_inp = M_inp if self.noise_M is None else M_inp + self.noise_M()
        E_inp = E_inp if self.noise_E is None else E_inp + self.noise_E()
        I_inp = I_inp if self.noise_I is None else I_inp + self.noise_I()
        if self.method == 'exp_euler':
            dt = brainstate.environ.get_dt()
            M = self.M.value + self.Mv.value * dt
            E = self.E.value + self.Ev.value * dt
            I = self.I.value + self.Iv.value * dt
            Mv = exp_euler_step(self.dMv, self.Mv.value, self.M.value, self.E.value, self.I.value, M_inp)
            Ev = exp_euler_step(self.dEv, self.Ev.value, self.M.value, self.E.value, E_inp)
            Iv = exp_euler_step(self.dIv, self.Iv.value, self.M.value, self.I.value, I_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            state = (self.M.value, self.E.value, self.I.value, self.Mv.value, self.Ev.value, self.Iv.value)
            M, E, I, Mv, Ev, Iv = method(self.derivative, state, 0. * u.ms, M_inp, E_inp, I_inp)
        self.M.value = M
        self.E.value = E
        self.I.value = I
        self.Mv.value = Mv
        self.Ev.value = Ev
        self.Iv.value = Iv
        return self.eeg()

    def eeg(self):
        # EEG-like proxy: difference between excitatory and inhibitory PSPs at pyramidal
        return self.E.value - self.I.value
