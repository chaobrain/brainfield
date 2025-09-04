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
import jax
import jax.numpy as jnp
from brainstate.typing import PyTree

__all__ = [
    'ode_euler_step',
    'ode_rk2_step',
    'ode_rk3_step',
    'ode_rk4_step',
    'sde_euler_step',
    'sde_milstein_step',
]

ODE = Callable[[PyTree, float, ...], PyTree]


def ode_euler_step(f: ODE, y: PyTree, t, *args):
    """
    Euler method for solving ordinary differential equations.
    
    The Euler method is the simplest numerical method for solving ODEs of the form:
    dy/dt = f(y, t)
    
    The method approximates the solution using:
    y_{n+1} = y_n + dt * f(y_n, t_n)
    
    Args:
        f (ODE): The differential equation function dy/dt = f(y, t, *args)
        y (PyTree): Current state vector
        t (float): Current time
        *args: Additional arguments passed to function f
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This is a first-order method with O(dt) local truncation error.
        It's the least accurate but most computationally efficient method.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    return jax.tree.map(lambda x, _k1: x + dt * _k1, y, k1)


def ode_rk2_step(f: ODE, y: PyTree, t, *args):
    """
    Second-order Runge-Kutta method (RK2) for solving ODEs.
    
    Also known as the midpoint method or Heun's method, this method provides
    better accuracy than Euler by using two function evaluations:
    
    k1 = f(y_n, t_n)
    k2 = f(y_n + dt*k1, t_n + dt)
    y_{n+1} = y_n + dt/2 * (k1 + k2)
    
    Args:
        f (ODE): The differential equation function dy/dt = f(y, t, *args)
        y (PyTree): Current state vector
        t (float): Current time
        *args: Additional arguments passed to function f
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This is a second-order method with O(dt²) local truncation error.
        More accurate than Euler with only one additional function evaluation.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(jax.tree.map(lambda x, k: x + dt * k, y, k1), t + dt, *args)
    return jax.tree.map(lambda x, _k1, _k2: x + dt / 2 * (_k1 + _k2), y, k1, k2)


def ode_rk3_step(f: ODE, y: PyTree, t, *args):
    """
    Third-order Runge-Kutta method (RK3) for solving ODEs.
    
    This method uses three function evaluations to achieve third-order accuracy:
    
    k1 = f(y_n, t_n)
    k2 = f(y_n + dt/2*k1, t_n + dt/2)
    k3 = f(y_n - dt*k1 + 2*dt*k2, t_n + dt)
    y_{n+1} = y_n + dt/6 * (k1 + 4*k2 + k3)
    
    Args:
        f (ODE): The differential equation function dy/dt = f(y, t, *args)
        y (PyTree): Current state vector
        t (float): Current time
        *args: Additional arguments passed to function f
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This is a third-order method with O(dt³) local truncation error.
        More accurate than RK2 but requires one additional function evaluation.
    """
    dt = brainstate.environ.get_dt()
    k1 = f(y, t, *args)
    k2 = f(jax.tree.map(lambda x, k: x + dt / 2 * k, y, k1), t + dt / 2, *args)
    k3 = f(jax.tree.map(lambda x, k1_val, k2_val: x - dt * k1_val + 2 * dt * k2_val, y, k1, k2), t + dt, *args)
    return jax.tree.map(lambda x, _k1, _k2, _k3: x + dt / 6 * (_k1 + 4 * _k2 + _k3), y, k1, k2, k3)


def ode_rk4_step(f: ODE, y: PyTree, t, *args):
    """
    Fourth-order Runge-Kutta method (RK4) for solving ODEs.
    
    The classic RK4 method uses four function evaluations to achieve fourth-order accuracy:
    
    k1 = f(y_n, t_n)
    k2 = f(y_n + dt/2*k1, t_n + dt/2)
    k3 = f(y_n + dt/2*k2, t_n + dt/2)
    k4 = f(y_n + dt*k3, t_n + dt)
    y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Args:
        f (ODE): The differential equation function dy/dt = f(y, t, *args)
        y (PyTree): Current state vector
        t (float): Current time
        *args: Additional arguments passed to function f
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This is a fourth-order method with O(dt⁴) local truncation error.
        The most commonly used method due to excellent accuracy/cost trade-off.
    """
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
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs).
    
    Solves SDEs of the form:
    dy = f(y, t)dt + g(y, t)dW
    
    where f is the drift term, g is the diffusion term, and dW is Wiener noise.
    
    The Euler-Maruyama scheme approximates:
    y_{n+1} = y_n + f(y_n, t_n)*dt + g(y_n, t_n)*ΔW_n
    
    where ΔW_n ~ N(0, dt) is the Wiener increment.
    
    Args:
        df (Callable): Drift function f(y, t, **kwargs) -> PyTree
        dg (Callable): Diffusion function g(y, t, **kwargs) -> PyTree  
        y (PyTree): Current state vector
        t (float): Current time
        sde_type (str): Type of SDE interpretation ('ito' only supported)
        **kwargs: Additional arguments passed to df and dg
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This method has strong convergence order 0.5 and weak convergence order 1.0.
        Only Itô interpretation is currently supported.
    """
    assert sde_type in ['ito', ]

    dt = brainstate.environ.get_dt()
    dt_sqrt = jnp.sqrt(dt)
    y_bars = jax.tree.map(
        lambda y0, drift, diffusion: y0 + drift * dt + diffusion * brainstate.random.randn_like(y0) * dt_sqrt,
        y, df(y, t, **kwargs), dg(y, t, **kwargs)
    )
    return y_bars


def sde_milstein_step(df, dg, y, t, sde_type='ito', **kwargs):
    """
    Milstein method for solving stochastic differential equations (SDEs).
    
    Solves SDEs of the form:
    dy = f(y, t)dt + g(y, t)dW
    
    The Milstein scheme includes an additional correction term for higher accuracy:
    y_{n+1} = y_n + f(y_n, t_n)*dt + g(y_n, t_n)*ΔW_n + 
              (1/2)*g(y_n, t_n)*∂g/∂y(y_n, t_n)*((ΔW_n)² - dt)
    
    This method approximates the derivative ∂g/∂y using finite differences:
    ∂g/∂y ≈ (g(y + g*√dt) - g(y)) / √dt
    
    Args:
        df (Callable): Drift function f(y, t, **kwargs) -> PyTree
        dg (Callable): Diffusion function g(y, t, **kwargs) -> PyTree
        y (PyTree): Current state vector
        t (float): Current time
        sde_type (str): SDE interpretation ('ito' or 'stra' for Stratonovich)
        **kwargs: Additional arguments passed to df and dg
        
    Returns:
        PyTree: Updated state vector y_{n+1}
        
    Note:
        This method has strong convergence order 1.0, better than Euler-Maruyama.
        Supports both Itô and Stratonovich interpretations.
        The finite difference approximation is used for the derivative term.
    """
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
