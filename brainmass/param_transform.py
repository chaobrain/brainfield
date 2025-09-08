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


from abc import ABC, abstractmethod
from typing import Callable, Sequence

import brainunit as u
import jax.numpy as jnp
from brainstate.typing import ArrayLike
from jax import Array

__all__ = [
    'Transform',
    'IdentityTransform',
    'SigmoidTransform',
    'SoftplusTransform',
    'NegSoftplusTransform',
    'AffineTransform',
    'ChainTransform',
    'MaskedTransform',
    'CustomTransform',
]


def save_exp(x, max_value: float = 20.0):
    """
    Clip the input to a maximum value and return its exponential.
    """
    x = u.math.clip(x, a_max=max_value)
    return u.math.exp(x)


class Transform(ABC):
    """
    Abstract base class for transformations.

    Subclasses must implement the `forward` and `inverse` methods to define
    bijective transformations. The class is callable and applies the `forward`
    transformation by default.
    """

    def __call__(self, x: ArrayLike) -> Array:
        """
        Apply the forward transformation to the input.

        Args:
            x (ArrayLike): Input array to transform.

        Returns:
            Array: Transformed output.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        """
        Forward transformation.

        Args:
            x (ArrayLike): Input array.

        Returns:
            Array: Transformed output.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike) -> Array:
        """
        Inverse transformation.

        Args:
            y (ArrayLike): Transformed array.

        Returns:
            Array: Original input array.
        """
        pass


class IdentityTransform(Transform):
    """
    Identity transformation.
    """

    def forward(self, x: ArrayLike) -> Array:
        return x

    def inverse(self, y: ArrayLike) -> Array:
        return y


class SigmoidTransform(Transform):
    """
    Sigmoid transformation.
    """

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, upper].

        Args:
            lower (ArrayLike): Lower bound of the interval.
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.unit = u.get_unit(lower)

    def forward(self, x: ArrayLike) -> Array:
        y = 1.0 / (1.0 + save_exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        x = (y - self.lower) / self.width
        x = -u.math.log((1.0 / x) - 1.0)
        return x


class SoftplusTransform(Transform):
    """
    Softplus transformation.
    """

    def __init__(self, lower: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval [lower, inf).

        Args:
            lower (ArrayLike): Lower bound of the interval.
        """
        super().__init__()
        self.lower = lower
        self.unit = u.get_unit(lower)

    def forward(self, x: ArrayLike) -> Array:
        return jnp.log1p(save_exp(x)) + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        return u.math.log(save_exp((y - self.lower) / self.unit) - 1.0)


class NegSoftplusTransform(SoftplusTransform):
    """
    Negative softplus transformation.
    """

    def __init__(self, upper: ArrayLike) -> None:
        """This transform maps any value bijectively to the interval (-inf, upper].

        Args:
            upper (ArrayLike): Upper bound of the interval.
        """
        super().__init__(upper)

    def forward(self, x: ArrayLike) -> Array:
        return -super().forward(-x)

    def inverse(self, y: ArrayLike) -> Array:
        return -super().inverse(-y)


class AffineTransform(Transform):
    def __init__(self, scale: ArrayLike, shift: ArrayLike):
        """This transform rescales and shifts the input.

        Args:
            scale (ArrayLike): Scaling factor.
            shift (ArrayLike): Additive shift.

        Raises:
            ValueError: Scale needs to be larger than 0
        """
        if jnp.allclose(scale, 0):
            raise ValueError("a cannot be zero, must be invertible")
        self.a = scale
        self.b = shift

    def forward(self, x: ArrayLike) -> Array:
        return self.a * x + self.b

    def inverse(self, x: ArrayLike) -> Array:
        return (x - self.b) / self.a


class ChainTransform(Transform):
    """
    Chaining together multiple transformations.
    """

    def __init__(self, *transforms: Sequence[Transform]) -> None:
        """A chain of transformations

        Args:
            *transforms (Sequence[Transform]): Transforms to apply
        """
        super().__init__()
        self.transforms: Sequence[Transform] = transforms

    def forward(self, x: ArrayLike) -> Array:
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def inverse(self, y: ArrayLike) -> Array:
        for transform in reversed(self.transforms):
            y = transform.inverse(y)
        return y


class MaskedTransform(Transform):
    def __init__(self, mask: ArrayLike, transform: Transform) -> None:
        """A masked transformation

        Args:
            mask (ArrayLike): Which elements to transform
            transform (Transform): Transformation to apply
        """
        super().__init__()
        self.mask = mask
        self.transform = transform

    def forward(self, x: ArrayLike) -> Array:
        return u.math.where(self.mask, self.transform.forward(x), x)

    def inverse(self, y: ArrayLike) -> Array:
        return u.math.where(self.mask, self.transform.inverse(y), y)


class CustomTransform(Transform):
    """
    Custom transformation.
    """

    def __init__(self, forward_fn: Callable, inverse_fn: Callable) -> None:
        """
        A custom transformation using a user-defined froward and
        inverse function.

        Args:
            forward_fn (Callable): Forward transformation
            inverse_fn (Callable): Inverse transformation
        """
        super().__init__()
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

    def forward(self, x: ArrayLike) -> Array:
        return self.forward_fn(x)

    def inverse(self, y: ArrayLike) -> Array:
        return self.inverse_fn(y)
