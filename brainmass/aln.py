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
from jax import jit

from .noise import Noise

__all__ = [
    'AlnModel',
]


class AlnModel(brainstate.nn.Dynamics):
    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Transfer function lookup tables (pre-computed)
        precalc_r: brainstate.typing.ArrayLike,
        precalc_V: brainstate.typing.ArrayLike,
        precalc_tau_mu: brainstate.typing.ArrayLike,
        Irange: brainstate.typing.ArrayLike,
        sigmarange: brainstate.typing.ArrayLike,
        dI: brainstate.typing.ArrayLike,
        ds: brainstate.typing.ArrayLike,

        # Synaptic time constants
        tau_se: brainstate.typing.ArrayLike = 2.0,
        tau_si: brainstate.typing.ArrayLike = 5.0,

        # Synaptic coupling strengths
        Jee_max: brainstate.typing.ArrayLike = 2.43,
        Jei_max: brainstate.typing.ArrayLike = -3.3,
        Jie_max: brainstate.typing.ArrayLike = 2.7,
        Jii_max: brainstate.typing.ArrayLike = -1.62,

        # Synaptic efficacies
        cee: brainstate.typing.ArrayLike = 0.3,
        cie: brainstate.typing.ArrayLike = 0.3,
        cei: brainstate.typing.ArrayLike = 0.5,
        cii: brainstate.typing.ArrayLike = 0.5,

        # Neuron model parameters
        a: brainstate.typing.ArrayLike = 18.0,
        b: brainstate.typing.ArrayLike = 100.0,
        EA: brainstate.typing.ArrayLike = -80.0,
        tauA: brainstate.typing.ArrayLike = 200.0,
        C: brainstate.typing.ArrayLike = 200.0,
        gL: brainstate.typing.ArrayLike = 10.0,
        taum: brainstate.typing.ArrayLike = 20.0,

        # External input noise parameters
        sigmae_ext: brainstate.typing.ArrayLike = 1.5,
        sigmai_ext: brainstate.typing.ArrayLike = 1.5,

        # Connections parameters
        Ke: brainstate.typing.ArrayLike = 800,
        Ki: brainstate.typing.ArrayLike = 200,
        c_gl: brainstate.typing.ArrayLike = 0.3,  # connections between areas
        ke_gl: brainstate.typing.ArrayLike = 250.,  # number of incoming E connections
        ext_exc_rate: brainstate.typing.ArrayLike = 0.0,  # external neuronal firing rate input
        ext_inh_rate: brainstate.typing.ArrayLike = 0.0,

        # Noise processes
        noise_E: Noise = None,
        noise_I: Noise = None,
    ):
        super().__init__(in_size=in_size)
        self.dt = 0.1

        # Store all parameters
        self.precalc_r = precalc_r
        self.precalc_V = precalc_V
        self.precalc_tau_mu = precalc_tau_mu
        self.Irange = Irange
        self.sigmarange = sigmarange
        self.dI = dI
        self.ds = ds
        self.tau_se = brainstate.init.param(tau_se, self.varshape)
        self.tau_si = brainstate.init.param(tau_si, self.varshape)
        self.cee = brainstate.init.param(cee, self.varshape)
        self.cei = brainstate.init.param(cei, self.varshape)
        self.cie = brainstate.init.param(cie, self.varshape)
        self.cii = brainstate.init.param(cii, self.varshape)
        self.c_gl = brainstate.init.param(c_gl, self.varshape)
        self.ke_gl = brainstate.init.param(ke_gl, self.varshape)
        self.ext_exc_rate = brainstate.init.param(ext_exc_rate, self.varshape)
        self.ext_inh_rate = brainstate.init.param(ext_inh_rate, self.varshape)
        self.Jee_max = brainstate.init.param(Jee_max, self.varshape)
        self.Jei_max = brainstate.init.param(Jei_max, self.varshape)
        self.Jie_max = brainstate.init.param(Jie_max, self.varshape)
        self.Jii_max = brainstate.init.param(Jii_max, self.varshape)
        self.a = brainstate.init.param(a, self.varshape)
        self.b = brainstate.init.param(b, self.varshape)
        self.EA = brainstate.init.param(EA, self.varshape)
        self.tauA = brainstate.init.param(tauA, self.varshape)
        self.C = brainstate.init.param(C, self.varshape)
        self.gL = brainstate.init.param(gL, self.varshape)
        self.taum = brainstate.init.param(taum, self.varshape)
        self.sigmae_ext = brainstate.init.param(sigmae_ext, self.varshape)
        self.sigmai_ext = brainstate.init.param(sigmai_ext, self.varshape)
        self.Ke = brainstate.init.param(Ke, self.varshape)
        self.Ki = brainstate.init.param(Ki, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I

        # Pre-calculate rescaled synaptic efficacies
        self.cee_scaled = self.cee * self.tau_se / self.Jee_max
        self.cie_scaled = self.cie * self.tau_se / self.Jie_max
        self.cei_scaled = self.cei * self.tau_si / abs(self.Jei_max)
        self.cii_scaled = self.cii * self.tau_si / abs(self.Jii_max)

        # Pre-calculate squared J values
        self.sq_Jee_max = self.Jee_max ** 2
        self.sq_Jei_max = self.Jei_max ** 2
        self.sq_Jie_max = self.Jie_max ** 2
        self.sq_Jii_max = self.Jii_max ** 2

        self.noise_E = noise_E
        self.noise_I = noise_I

        self._sigmarange = float(sigmarange[0] if jnp.ndim(sigmarange) else sigmarange)
        self._ds = float(ds[0] if jnp.ndim(ds) else ds)
        self._Irange = float(Irange[0] if jnp.ndim(Irange) else Irange)
        self._dI = float(dI[0] if jnp.ndim(dI) else dI)

    def init_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape

        # Firing rates
        self.rates_exc = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.rates_inh = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        # Filtered mean inputs
        self.mufe = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.mufi = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))

        # Adaptation current
        self.IA = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))

        # Synaptic means
        self.seem = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.seim = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.siem = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.siim = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))

        # Synaptic variances
        self.seev = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.seiv = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.siev = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))
        self.siiv = brainstate.HiddenState(brainstate.init.param(jnp.zeros, size))

    def calculate_total_input_firing_rate(self, rowsum, rowsumq):
        r_exc_kHz = self.rates_exc.value * 1e-3
        r_inh_kHz = self.rates_inh.value * 1e-3
        z1ee = self.cee_scaled * self.Ke * r_exc_kHz + self.c_gl * self.ke_gl * rowsum + self.c_gl * self.ke_gl * self.ext_exc_rate
        z1ei = self.cei_scaled * self.Ki * r_inh_kHz
        z1ie = self.cie_scaled * self.Ke * r_exc_kHz + self.c_gl * self.ke_gl * self.ext_inh_rate
        z1ii = self.cii_scaled * self.Ki * r_inh_kHz

        z2ee = self.cee_scaled ** 2 * self.Ke * r_exc_kHz + self.c_gl ** 2 * self.ke_gl * rowsumq + self.c_gl ** 2 * self.ke_gl * self.ext_exc_rate
        z2ei = self.cei_scaled ** 2 * self.Ki * r_inh_kHz
        z2ie = self.cie_scaled ** 2 * self.Ke * r_exc_kHz + self.c_gl ** 2 * self.ke_gl * self.ext_inh_rate
        z2ii = self.cii_scaled ** 2 * self.Ki * r_inh_kHz
        return z1ee, z1ei, z1ie, z1ii, z2ee, z2ei, z2ie, z2ii, r_exc_kHz, r_inh_kHz

    def calculate_mean_synaptic_gating(self, z1ee, z1ei, z1ie, z1ii):
        d_seem = ((1 - self.seem.value) * z1ee - self.seem.value) / self.tau_se
        d_seim = ((1 - self.seim.value) * z1ei - self.seim.value) / self.tau_si
        d_siem = ((1 - self.siem.value) * z1ie - self.siem.value) / self.tau_se
        d_siim = ((1 - self.siim.value) * z1ii - self.siim.value) / self.tau_si
        return d_seem, d_seim, d_siem, d_siim

    def calulate_mean_synaptic_gating_variance(self, z1ee, z1ei, z1ie, z1ii, z2ee, z2ei, z2ie, z2ii):
        d_seev = ((1 - self.seem.value) ** 2 * z2ee + (
            z2ee - 2 * self.tau_se * (z1ee + 1)) * self.seev.value) / self.tau_se ** 2
        d_seiv = ((1 - self.seim.value) ** 2 * z2ei + (
            z2ei - 2 * self.tau_si * (z1ei + 1)) * self.seiv.value) / self.tau_si ** 2
        d_siev = ((1 - self.siem.value) ** 2 * z2ie + (
            z2ie - 2 * self.tau_se * (z1ie + 1)) * self.siev.value) / self.tau_se ** 2
        d_siiv = ((1 - self.siim.value) ** 2 * z2ii + (
            z2ii - 2 * self.tau_si * (z1ii + 1)) * self.siiv.value) / self.tau_si ** 2
        return d_seev, d_seiv, d_siev, d_siiv

    def calculate_standard_deviation(self, z1ee, z1ei, z1ie, z1ii):
        sigmae_num_1 = 2 * self.sq_Jee_max * self.seev.value * self.tau_se * self.taum
        sigmae_den_1 = (1 + z1ee) * self.taum + self.tau_se
        sigmae_num_2 = 2 * self.sq_Jei_max * self.seiv.value * self.tau_si * self.taum
        sigmae_den_2 = (1 + z1ei) * self.taum + self.tau_si
        sigmae = jnp.sqrt(sigmae_num_1 / sigmae_den_1 + sigmae_num_2 / sigmae_den_2 + self.sigmae_ext ** 2)

        sigmai_num_1 = 2 * self.sq_Jie_max * self.siev.value * self.tau_se * self.taum
        sigmai_den_1 = (1 + z1ie) * self.taum + self.tau_se
        sigmai_num_2 = 2 * self.sq_Jii_max * self.siiv.value * self.tau_si * self.taum
        sigmai_den_2 = (1 + z1ii) * self.taum + self.tau_si
        sigmai = jnp.sqrt(sigmai_num_1 / sigmai_den_1 + sigmai_num_2 / sigmai_den_2 + self.sigmai_ext ** 2)

        return sigmae, sigmai

    def count_mean_current(self, ext_exc=0., ext_inh=0., noise_e=0, noise_i=0):
        ext_exc = 0. if ext_exc is None else ext_exc
        ext_inh = 0. if ext_inh is None else ext_inh
        noise_e = 0. if noise_e is None else noise_e
        noise_i = 0. if noise_i is None else noise_i
        mue = self.Jee_max * self.seem.value + self.Jei_max * self.seim.value + noise_e + ext_exc
        mui = self.Jie_max * self.siem.value + self.Jii_max * self.siim.value + noise_i + ext_inh
        return mue, mui

    @staticmethod
    @jit
    def _interpolate(table, x, dx, xi, y, dy, yi):
        xid = (xi - x) / dx
        yid = (yi - y) / dy

        xid1 = jnp.floor(xid).astype(int)
        yid1 = jnp.floor(yid).astype(int)

        dxid = xid - xid1
        dyid = yid - yid1

        # Clip indices to be within bounds
        xid1 = jnp.clip(xid1, 0, table.shape[1] - 2)
        yid1 = jnp.clip(yid1, 0, table.shape[0] - 2)

        val = (table[yid1, xid1] * (1 - dxid) * (1 - dyid) +
               table[yid1, xid1 + 1] * dxid * (1 - dyid) +
               table[yid1 + 1, xid1] * (1 - dxid) * dyid +
               table[yid1 + 1, xid1 + 1] * dxid * dyid)
        return val

    def update(self, rowsum, rowsumq, ext_exc_current=0., ext_inh_current=0.):
        ext_exc = 0. if ext_exc_current is None else ext_exc_current
        ext_inh = 0. if ext_inh_current is None else ext_inh_current
        noise_e = self.noise_E() if self.noise_E is not None else 0.0
        noise_i = self.noise_I() if self.noise_I is not None else 0.0

        # Interpolate from lookup tables
        z1ee, z1ei, z1ie, z1ii, z2ee, z2ei, z2ie, z2ii, r_exc_kHz, r_inh_kHz = self.calculate_total_input_firing_rate(
            rowsum, rowsumq)
        sigmae, sigmai = self.calculate_standard_deviation(z1ee, z1ei, z1ie, z1ii)
        mue, mui = self.count_mean_current(ext_exc, ext_inh, noise_e, noise_i)
        mu_eff_exc = mue - self.IA.value / self.C

        args = (self._sigmarange, self._ds, self._Irange, self._dI)
        rates_exc_new = self._interpolate(self.precalc_r, *args, sigmae, mu_eff_exc) * 1e3  # to Hz

        print("mu_eff_exc min/max:", mu_eff_exc.min(), mu_eff_exc.max())
        print("sigmae min/max:", sigmae.min(), sigmae.max())
        print("rates_exc_new min/max:", rates_exc_new.min(), rates_exc_new.max())
        Vmean_exc = self._interpolate(self.precalc_V, *args, sigmae, mu_eff_exc)
        tau_exc = self._interpolate(self.precalc_tau_mu, *args, sigmae, mu_eff_exc)

        rates_inh_new = self._interpolate(self.precalc_r, *args, sigmai, self.mufi.value) * 1e3  # to Hz
        tau_inh = self._interpolate(self.precalc_tau_mu, *args, sigmai, self.mufi.value)

        # --- Calculate derivatives for all state variables ---
        d_mufe = (mue - self.mufe.value) / tau_exc
        d_mufi = (mui - self.mufi.value) / tau_inh
        d_IA = (self.a * (Vmean_exc - self.EA) - self.IA.value + self.tauA * self.b * r_exc_kHz) / self.tauA

        # --- Update states using Euler integration ---
        dt = self.dt
        self.rates_exc.value = rates_exc_new  # This is an algebraic update, not differential
        self.rates_inh.value = rates_inh_new

        self.mufe.value += dt * d_mufe
        self.mufi.value += dt * d_mufi
        self.IA.value += dt * d_IA

        d_seem, d_seim, d_siem, d_siim = self.calculate_mean_synaptic_gating(z1ee, z1ei, z1ie, z1ii)
        d_seev, d_seiv, d_siev, d_siiv = self.calulate_mean_synaptic_gating_variance(
            z1ee, z1ei, z1ie, z1ii, z2ee, z2ei, z2ie, z2ii)

        self.seem.value += brainstate.nn.exp_euler_step(lambda seem: ((1 - seem) * z1ee - seem) / self.tau_se,
                                                        self.seem.value)
        self.seim.value += brainstate.nn.exp_euler_step(lambda seim: ((1 - seim) * z1ei - seim) / self.tau_si,
                                                        self.seim.value)
        self.siem.value += brainstate.nn.exp_euler_step(lambda siem: ((1 - siem) * z1ie - siem) / self.tau_se,
                                                        self.siem.value)
        self.siim.value += brainstate.nn.exp_euler_step(lambda siim: ((1 - siim) * z1ii - siim) / self.tau_si,
                                                        self.siim.value)

        # Ensure variance is non-negative
        self.seev.value = jnp.maximum(0., self.seev.value + dt * d_seev)
        self.seiv.value = jnp.maximum(0., self.seiv.value + dt * d_seiv)
        self.siev.value = jnp.maximum(0., self.siev.value + dt * d_siev)
        self.siiv.value = jnp.maximum(0., self.siiv.value + dt * d_siiv)
        return self.rates_exc.value
