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
import brainunit as u

__all__ = [
    'OUProcess',
]


class OUProcess(brainstate.nn.Dynamics):
    r"""
    The Ornstein–Uhlenbeck process.

    The Ornstein–Uhlenbeck process :math:`x_{t}` is defined by the following
    stochastic differential equation:

    .. math::

       \tau dx_{t}=-\theta \,x_{t}\,dt+\sigma \,dW_{t}

    where :math:`\theta >0` and :math:`\sigma >0` are parameters and :math:`W_{t}`
    denotes the Wiener process.

    Parameters
    ==========
    in_size: int, sequence of int
      The model size.
    mean: ArrayLike
      The noise mean value.
    sigma: ArrayLike
      The noise amplitude.
    tau: ArrayLike
      The decay time constant.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        mean: brainstate.typing.ArrayLike = 0.,  # noise mean value
        sigma: brainstate.typing.ArrayLike = 1.,  # noise amplitude
        tau: brainstate.typing.ArrayLike = 10.,  # time constant
    ):
        super().__init__(in_size=in_size)

        # parameters
        self.mean = mean
        self.sigma = sigma
        self.tau = tau

    def init_state(self, batch_size=None, **kwargs):
        size = self.in_size if batch_size is None else (batch_size, *self.in_size)
        self.x = brainstate.HiddenState(u.math.zeros(size, unit=u.get_unit(self.mean)))

    def reset_state(self, batch_size=None, **kwargs):
        size = self.in_size if batch_size is None else (batch_size, *self.in_size)
        self.x.value = u.math.zeros(size, unit=u.get_unit(self.mean))

    def update(self):
        df = lambda x: (self.mean - x) / self.tau
        dg = lambda x: self.sigma / u.math.sqrt(self.tau)
        self.x.value = brainstate.nn.exp_euler_step(df, dg, self.x.value)
        return self.x.value
