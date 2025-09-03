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


from typing import Union

import brainstate
import brainunit as u
from brainstate.nn._dynamics import maybe_init_prefetch

Prefetch = Union[brainstate.nn.PrefetchDelayAt, brainstate.nn.PrefetchDelay, brainstate.nn.Prefetch]

__all__ = [
    'DiffusiveCoupling',
]


class DiffusiveCoupling(brainstate.nn.Module):
    r"""
    Diffusive coupling.

    This class implements a diffusive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = \sum_j (g_{ij} * x_{D_{ij}} - y_i)
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
    ):
        super().__init__()
        assert isinstance(x, Prefetch), f'The first element must be a Prefetch. But got {type(x)}.'
        assert isinstance(y, Prefetch), f'The second element must be a Prefetch. But got {type(y)}.'
        self.x = x
        self.y = y

        # Connection matrix
        self.conn = u.math.asarray(conn)
        assert self.conn.ndim in (1, 2), f'Only support 1d, 2d connection matrix. But we got {self.conn.ndim}d.'

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.x)
        maybe_init_prefetch(self.y)

    def update(self):
        delayed_x = self.x()
        y = u.math.expand_dims(self.y(), axis=1)  # (..., 1)
        if self.conn.ndim == 1:
            assert self.conn.size == delayed_x.shape[-1], (f'Connection matrix size {self.conn.size} does not '
                                                           f'match the variable size {delayed_x.shape[-1]}.')
            diffusive = (self.conn * delayed_x).reshape(y.shape[0], -1) - y
        elif self.conn.ndim == 2:
            delayed_x = delayed_x.reshape(y.shape[0], -1)
            assert self.conn.shape == delayed_x.shape, (f'Connection matrix shape {self.conn.shape} does not '
                                                        f'match the variable shape {delayed_x.shape}.')
            diffusive = (self.conn * delayed_x) - y
        else:
            raise NotImplementedError(f'Only support 1d, 2d connection matrix. But we got {self.conn.ndim}d.')
        return diffusive.sum(axis=1)
