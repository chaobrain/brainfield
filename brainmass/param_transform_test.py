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


import brainunit as u
import jax.numpy as jnp
import numpy as np

from brainmass.param_transform import SigmoidTransform


class TestSigmoidTransform:
    def test_sigmoid_transform_forward_inverse(self):
        lower = 0.0
        upper = 1.0
        transform = SigmoidTransform(lower, upper)
        x = jnp.array([-10.0, 0.0, 5.0])
        y = transform.forward(x)
        x_recovered = transform.inverse(y)
        assert u.math.allclose(x, np.array(x_recovered), rtol=1e-2)

    def test_sigmoid_transform_forward_inverse_unit(self):
        unit = u.mV
        lower = 0.0 * unit
        upper = 1.0 * unit
        transform = SigmoidTransform(lower, upper)
        x = jnp.array([0.1, 0.2, 0.4]) * unit
        y = transform.inverse(x)
        x_recovered = transform.forward(y)
        assert u.math.allclose(x, x_recovered, rtol=1e-2 * unit)

    def test_sigmoid_transform_bounds(self):
        lower = 2.0
        upper = 5.0
        transform = SigmoidTransform(lower, upper)
        x = jnp.array([-100.0, 0.0, 100.0])
        y = transform.forward(x)
        assert np.all(y >= lower) and np.all(y <= upper)
