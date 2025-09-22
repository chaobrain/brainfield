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

import numpy as np

import brainstate
import brainunit as u
import braintools

import brainmass


class TestArrayParamBasic:
    def test_initialization_identity(self):
        x = np.array([1.0, 2.0, 3.0])
        p = brainmass.ArrayParam(x)  # default IdentityTransform

        assert isinstance(p, brainstate.ParamState)
        assert isinstance(p, u.CustomArray)
        # Internal value equals input under identity transform
        assert u.math.allclose(p.value, x)
        # Exposed data applies transform (identity)
        assert u.math.allclose(p.data, x)

    def test_unit_support_identity(self):
        x = np.array([0.5, 1.5]) * u.nA
        p = brainmass.ArrayParam(x)

        assert u.get_unit(p.value) == u.nA
        assert u.get_unit(p.data) == u.nA
        assert p.value.shape == (2,)
        assert p.data.shape == (2,)

    def test_invalid_value_type_raises(self):
        # Clearly non-arraylike inputs should raise TypeError
        for bad in [None, {"a": 1}, object()]:
            try:
                _ = brainmass.ArrayParam(bad)
                assert False, "Expected TypeError for non-arraylike value"
            except TypeError:
                pass


class TestArrayParamWithCustomTransform:
    class AffineTransform:
        """Simple bijective transform y = a*x + b with an inverse."""

        def __init__(self, a=2.0, b=3.0):
            self.a = a
            self.b = b

        def __call__(self, x):
            return self.a * x + self.b

        def inverse(self, y):
            return (y - self.b) / self.a

    def test_transform_applied_on_access_and_inverse_on_store(self):
        t = self.AffineTransform(a=2.0, b=3.0)
        x = np.array([1.0, 2.0, 3.0])

        p = brainmass.ArrayParam(x, transform=t)

        # Internally stores inverse-transformed values
        expected_internal = (x - 3.0) / 2.0
        assert u.math.allclose(p.value, expected_internal)

        # Exposed data applies forward transform and recovers original x
        assert u.math.allclose(p.data, x)

        # Setting data applies inverse to internal storage
        new_data = np.array([10.0, 20.0, 30.0])
        p.data = new_data
        assert u.math.allclose(p.value, (new_data - 3.0) / 2.0)
        assert u.math.allclose(p.data, new_data)

    def test_transform_with_units(self):
        # Use identity from braintools for a unit-carrying check, then
        # also verify behavior with the custom affine transform.
        x = np.array([2.0, 4.0]) * u.mV

        # Identity preserves units end-to-end
        p1 = brainmass.ArrayParam(x, transform=braintools.IdentityTransform())
        assert u.get_unit(p1.data) == u.mV
        assert u.math.allclose(p1.data, x)

        # Affine transform should preserve unit tagging through __call__ and inverse
        t = self.AffineTransform(a=0.5, b=1.0 * u.mV)
        p2 = brainmass.ArrayParam(x, transform=t)
        # data property should equal the original x
        assert u.math.allclose(p2.data, x)
        # internal representation carries units as well (because inputs carry units)
        assert u.get_unit(p2.value) == u.mV

