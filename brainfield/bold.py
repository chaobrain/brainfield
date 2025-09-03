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

from .integration import ode_rk2_step

__all__ = [
    'BOLDSignal',
]


class BOLDSignal(brainstate.nn.Dynamics):
    r"""
    Balloon-Windkessel hemodynamic model of Friston et al. (2003).

    The Balloon-Windkessel model describes the coupling of perfusion to BOLD signal, with a
    dynamical model of the transduction of neuronal activity into perfusion changes. The
    model assumes that the BOLD signal is a static nonlinear function of the normalized total
    deoxyhemoglobin voxel content, normalized venous volume, resting net oxygen extraction
    fraction by the capillary bed, and resting blood volume fraction. The BOLD-signal estimation
    for each brain area is computed by the level of synaptic activity in that particular cortical
    area, noted $z_i$ for a given cortical are $i$.

    For the i-th region, synaptic activity $z_i$ causes an increase in a vasodilatory signal $x_i$
    that is subject to autoregulatory feedback. Inflow $f_i$ responds in proportion to this signal
    with concomitant changes in blood volume $v_i$ and deoxyhemoglobin content $q_i$. The equations
    relating these biophysical processes are as follows:

    $$
    \begin{gathered}
    \dot{x}_i=z_i-k_i x_i-\gamma_i\left(f_i-1\right) \\
    \dot{f}_i=x_i \\
    \tau_i \dot{v}_i=f_i-v_i^{1 / \alpha} \\
    \tau_i \dot{q}_i=\frac{f_i}{\rho}\left[1-(1-\rho)^{1 / f_i}\right]-q_i v_i^{1 / \alpha-1},
    \end{gathered}
    $$

    where $\rho$ is the resting oxygen extraction fraction. The BOLD signal is given by the following:

    $$
    \mathrm{BOLD}_i=V_0\left[k_1\left(1-q_i\right)+k_2\left(1-q_i / v_i\right)+k_3\left(1-v_i\right)\right],
    $$

    where $V_0 = 0.02, k1 = 7\rho, k2 = 2$, and $k3 = 2\rho − 0.2$. All biophysical parameters were taken
    as in Friston et al. (2003). The BOLD model converts the local synaptic activity of a given cortical
    area into an observable BOLD signal and does not actively couple the signals from other cortical areas.

    Parameters
    ----------
    in_size : int
        Size of the input vector (number of brain regions).
    gamma : float or callable, optional
        Rate of signal decay (default is 0.41).
    k : float or callable, optional
        Rate of flow-dependent elimination (default is 0.65).
    alpha : float or callable, optional
        Grubb's exponent (default is 0.32).
    tau : float or callable, optional
        Hemodynamic transit time (default is 0.98).
    rho : float or callable, optional
        Resting oxygen extraction fraction (default is 0.34).
    V0 : float, optional
        Resting blood volume fraction (default is 0.02).

    References
    ----------
    .. [1] Friston KJ, Harrison L, Penny W (2003) Dynamic causal modelling. Neuroimage 19:1273–1302,
           doi:10.1016/S1053-8119(03)00202-7
    """

    def __init__(
        self,
        in_size,
        gamma: Union[brainstate.typing.ArrayLike, Callable] = 0.41,
        k: Union[brainstate.typing.ArrayLike, Callable] = 0.65,
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.32,
        tau: Union[brainstate.typing.ArrayLike, Callable] = 0.98,
        rho: Union[brainstate.typing.ArrayLike, Callable] = 0.34,
        V0: float = 0.02,
    ):
        super().__init__(in_size)

        self.gamma = brainstate.init.param(gamma, self.varshape)
        self.k = brainstate.init.param(k, self.varshape)
        self.alpha = brainstate.init.param(alpha, self.varshape)
        self.tau = brainstate.init.param(tau, self.varshape)
        self.rho = brainstate.init.param(rho, self.varshape)

        self.V0 = V0
        self.k1 = 7 * self.rho
        self.k2 = 2.
        self.k3 = 2 * self.rho - 0.2

        self.init = brainstate.init.Constant(1.)

    def init_state(self, batch_size=None, **kwargs):
        self.x = brainstate.HiddenState(brainstate.init.param(self.init, self.varshape, batch_size))
        self.f = brainstate.HiddenState(brainstate.init.param(self.init, self.varshape, batch_size))
        self.v = brainstate.HiddenState(brainstate.init.param(self.init, self.varshape, batch_size))
        self.q = brainstate.HiddenState(brainstate.init.param(self.init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.x.value = brainstate.init.param(self.init, self.varshape, batch_size)
        self.f.value = brainstate.init.param(self.init, self.varshape, batch_size)
        self.v.value = brainstate.init.param(self.init, self.varshape, batch_size)
        self.q.value = brainstate.init.param(self.init, self.varshape, batch_size)

    def derivative(self, y, t, z):
        x, f, v, q = y
        dx = z - self.k * x - self.gamma * (f - 1)
        df = x
        dv = (f - jnp.power(v, 1 / self.alpha)) / self.tau
        E = 1 - jnp.power(1 - self.rho, 1 / f)
        dq = (f * E / self.rho - jnp.power(v, 1 / self.alpha) * q / v) / self.tau
        return dx, df, dv, dq

    def update(self, z):
        x, f, v, q = ode_rk2_step(self.derivative, (self.s.value, self.f.value, self.v.value, self.q.value), 0., z)
        self.x.value = x
        self.f.value = f
        self.v.value = v
        self.q.value = q

    def bold(self):
        return self.V0 * (self.k1 * (1 - self.q.value) +
                          self.k2 * (1 - self.q.value / self.rho) +
                          self.k3 * (1 - self.v.value))
