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

from ._common import XY_Oscillator
from ._typing import Initializer
from .noise import Noise

__all__ = [
    'StuartLandauOscillator',
]


class StuartLandauOscillator(XY_Oscillator):
    r"""
    Stuart-Landau model with Hopf bifurcation.

    The Stuart–Landau equation describes the behavior of a nonlinear oscillating
    system near the Hopf bifurcation, named after John Trevor Stuart and Lev Landau.

    .. math::

       \frac{dx}{dt} = (a - x^2 - y^2) * x - w*y + I^x_{ext} \\
       \frac{dy}{dt} = (a - x^2 - y^2) * y + w*x + I^y_{ext}

    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # model parameters
        a: Initializer = 0.25,
        w: Initializer = 0.2,

        # noise parameters
        noise_x: Noise = None,
        noise_y: Noise = None,

        # other parameters
        init_x: Callable = brainstate.init.Uniform(0, 0.05),
        init_y: Callable = brainstate.init.Uniform(0, 0.05),
        method: str = 'exp_euler',
    ):
        super().__init__(
            in_size,
            noise_x=noise_x,
            noise_y=noise_y,
            init_x=init_x,
            init_y=init_y,
            method=method,
        )

        # model parameters
        self.a = brainstate.init.param(a, self.varshape, allow_none=False)
        self.w = brainstate.init.param(w, self.varshape, allow_none=False)

    def dx(self, x, y, x_ext):
        return ((self.a - x * x - y * y) * x - self.w * y + x_ext) / u.ms

    def dy(self, y, x, y_ext):
        return ((self.a - x * x - y * y) * y - self.w * y + y_ext) / u.ms
