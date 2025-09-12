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


from typing import Union, Tuple

import brainunit as u

import brainstate
from brainstate.nn._dynamics import maybe_init_prefetch

# Typing alias for static type hints
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
]
# Runtime check tuple for isinstance
_PREFETCH_TYPES: Tuple[type, ...] = (
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
)

__all__ = [
    'DiffusiveCoupling',
    'AdditiveCoupling',
]


class DiffusiveCoupling(brainstate.nn.Module):
    r"""
    Diffusive coupling.

    This class implements a diffusive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = k * \sum_j g_{ij} * (x_{D_{ij}} - y_i)
    $$

    where:
        - $\mathrm{current}_i$: the output current for neuron $i$
        - $g_{ij}$: the connection strength between neuron $i$ and neuron $j$
        - $x_{D_{ij}}$: the delayed state variable for neuron $j$, as seen by neuron $i$
        - $y_i$: the state variable for neuron i

    Parameters
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    y : Prefetch
        The delayed state variable for the target units.
    conn : brainstate.typing.Array
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: float
        The global coupling strength. Default is 1.0.

    Attributes
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    y : Prefetch
        The delayed state variable for the target units.
    conn : Array
        The connection matrix.
    """

    def __init__(
        self,
        x: Prefetch,
        y: Prefetch,
        conn: brainstate.typing.Array,
        k: float = 1.0
    ):
        super().__init__()
        if not isinstance(x, _PREFETCH_TYPES):
            raise TypeError(f'The first argument x must be a Prefetch, got {type(x)}')
        if not isinstance(y, _PREFETCH_TYPES):
            raise TypeError(f'The second argument y must be a Prefetch, got {type(y)}')
        self.x = x
        self.y = y
        self.k = k

        # Connection matrix (support 1D flattened (N_out*N_in,) or 2D (N_out, N_in))
        self.conn = u.math.asarray(conn)
        if self.conn.ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (flattened) or 2D matrix; got {self.conn.ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.x)
        maybe_init_prefetch(self.y)

    def update(self):
        # y: (..., N_out)
        y_val = self.y()
        if y_val.ndim < 1:
            raise ValueError(f'y must have at least 1 dimension; got shape {y_val.shape}')
        n_out = y_val.shape[-1]
        y_exp = u.math.expand_dims(y_val, axis=-1)  # (..., N_out, 1)

        # x expected shape on trailing dims: (N_out, N_in) or flattened N_out*N_in
        x_val = self.x()
        if x_val.ndim < 1:
            raise ValueError(f'x must have at least 1 dimension; got shape {x_val.shape}')

        # Build (N_out, N_in) connection matrix
        if self.conn.ndim == 1:
            if self.conn.size % n_out != 0:
                raise ValueError(
                    f'Flattened connection length {self.conn.size} is not divisible by N_out={n_out}.'
                )
            n_in = self.conn.size // n_out
            conn2d = u.math.reshape(self.conn, (n_out, n_in))
        else:
            conn2d = self.conn
            if conn2d.shape[0] != n_out:
                raise ValueError(
                    f'Connection rows ({conn2d.shape[0]}) must match y size ({n_out}).'
                )
            n_in = conn2d.shape[1]

        # Reshape x to (..., N_out, N_in)
        if x_val.ndim >= 2 and x_val.shape[-2:] == (n_out, n_in):
            x_mat = x_val
        elif x_val.shape[-1] == n_out * n_in:
            x_mat = u.math.reshape(x_val, (*x_val.shape[:-1], n_out, n_in))
        else:
            raise ValueError(
                f'x has incompatible shape {x_val.shape}; expected (..., {n_out}, {n_in}) '
                f'or flattened (..., {n_out*n_in}).'
            )

        # Broadcast conn across leading dims if needed
        diff = x_mat - y_exp  # (..., N_out, N_in)
        diffusive = diff * conn2d  # broadcasting on leading dims
        return self.k * diffusive.sum(axis=-1)  # (..., N_out)


class AdditiveCoupling(brainstate.nn.Module):
    r"""
    Additive coupling.

    This class implements an additive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = k * \sum_j g_{ij} * x_{D_{ij}}
    $$

    where:
        - $\mathrm{current}_i$: the output current for neuron $i$
        - $g_{ij}$: the connection strength between neuron $i$ and neuron $j$
        - $x_{D_{ij}}$: the delayed state variable for neuron $j$, as seen by neuron $i$

    Parameters
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    conn : brainstate.typing.Array
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: float
        The global coupling strength. Default is 1.0.

    Attributes
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    conn : Array
        The connection matrix.
    """

    def __init__(
        self,
        x: Prefetch,
        conn: brainstate.typing.Array,
        k: float = 1.0
    ):
        super().__init__()
        if not isinstance(x, _PREFETCH_TYPES):
            raise TypeError(f'The first argument x must be a Prefetch, got {type(x)}')
        self.x = x
        self.k = k

        # Connection matrix
        self.conn = u.math.asarray(conn)
        if self.conn.ndim != 2:
            raise ValueError(f'Only support 2D connection matrix; got {self.conn.ndim}D.')

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.x)

    def update(self):
        # x expected trailing dims to match connection (N_out, N_in) or flattened N_out*N_in
        x_val = self.x()
        n_out, n_in = self.conn.shape

        if x_val.ndim >= 2 and x_val.shape[-2:] == (n_out, n_in):
            x_mat = x_val
        elif x_val.shape[-1] == n_out * n_in:
            x_mat = u.math.reshape(x_val, (*x_val.shape[:-1], n_out, n_in))
        else:
            raise ValueError(
                f'x has incompatible shape {x_val.shape}; expected (..., {n_out}, {n_in}) '
                f'or flattened (..., {n_out*n_in}).'
            )

        additive = self.conn * x_mat  # broadcasting on leading dims
        return self.k * additive.sum(axis=-1)  # (..., N_out)
