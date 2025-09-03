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
import jax
import jax.numpy as jnp

__all__ = [
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'sde_euler',
    'sde_milstein',
]


def ode_euler_step(f, y, t, *args):
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    return jax.tree.map(lambda x, _k1,: x + dt * _k1, y, k1)


def ode_rk2_step(f, y, t, *args):
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(jax.tree.map(lambda x, k: x + dt * k, y, k1), t + dt, *args)
    return jax.tree.map(lambda x, _k1, _k2: x + dt / 2 * (_k1 + _k2), y, k1, k2)


def ode_rk3_step(f, y, t, *args):
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(jax.tree.map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(jax.tree.map(lambda x, k: x + dt / 2 * k, y, k2), t + dt / 2, *args)
    k4 = f(jax.tree.map(lambda x, k: x + dt * k, y, k3), t + dt, *args)
    return jax.tree.map(lambda x, _k1, _k2, _k3, _k4: x + dt / 6 * (_k1 + 4 * _k2 + _k3), y, k1, k2, k3, k4)


def ode_rk4_step(f, y, t, *args):
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(jax.tree.map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(jax.tree.map(lambda x, k: x + dt / 2 * k, y, k2), t + dt / 2, *args)
    k4 = f(jax.tree.map(lambda x, k: x + dt * k, y, k3), t + dt, *args)
    return jax.tree.map(
        lambda x, _k1, _k2, _k3, _k4: x + dt / 6 * (_k1 + 2 * _k2 + 2 * _k3 + _k4),
        y, k1, k2, k3, k4
    )


def sde_euler_step(df, dg, y, t, sde_type='ito', **kwargs):
    assert sde_type in ['ito', ]

    dt = brainstate.environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)
    y_bars = jax.tree.map(
        lambda y0, drift, diffusion: y0 + drift * dt + diffusion * brainstate.random.randn_like(y0) * dt_sqrt,
        y, df(y, t, **kwargs), dg(y, t, **kwargs)
    )
    return y_bars


def sde_milstein_step(df, dg, y, t, sde_type='ito', **kwargs):
    assert sde_type in ['ito', 'stra']

    dt = brainstate.environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)

    # drift values
    drifts = df(y, t, **kwargs)

    # diffusion values
    diffusions = dg(y, t, **kwargs)

    # intermediate results
    y_bars = jax.tree.map(lambda y0, drift, diffusion: y0 + drift * dt + diffusion * dt_sqrt, y, drifts, diffusions)
    diffusion_bars = dg(y_bars, t, **kwargs)

    # integral results
    def f_integral(y0, drift, diffusion, diffusion_bar):
        noise = brainstate.random.randn_like(y0) * dt_sqrt
        noise_p2 = (noise ** 2 - dt) if sde_type == 'ito' else noise ** 2
        minus = (diffusion_bar - diffusion) / 2 / dt_sqrt
        return y0 + drift * dt + diffusion * noise + minus * noise_p2

    integrals = jax.tree.map(f_integral, y, drifts, diffusions, diffusion_bars)
    return integrals
