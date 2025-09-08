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

from .param_transform import IdentityTransform

__all__ = [
    'Parameter',
]


class Parameter(brainstate.ParamState, brainstate.mixin.ArrayImpl):
    def __init__(self, value, transform=None):
        if transform is None:
            transform = IdentityTransform()
        self._transform = transform
        value = self._transform.inverse(value)
        super().__init__(value=value)

    def _read_value(self):
        self.check_if_deleted()
        return self._transform.forward(self._value)
