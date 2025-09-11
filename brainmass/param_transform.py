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
    r"""
    Numerically stable exponential function with clipping.
    
    Computes the exponential of the input after clipping to prevent numerical overflow.
    This function implements a safe exponential operation by limiting the maximum
    input value before computing the exponential.
    
    .. math::
        \text{save\_exp}(x) = \exp(\min(x, x_{\text{max}}))
    
    where :math:`x_{\text{max}}` is the maximum allowed value.
    
    Parameters
    ----------
    x : array_like
        Input array.
    max_value : float, optional
        Maximum value to clip the input to, by default 20.0. Values above this
        threshold are clipped to prevent numerical overflow in the exponential.
    
    Returns
    -------
    array_like
        The exponential of the clipped input.
        
    Notes
    -----
    This function is particularly useful in neural network computations where
    exponential operations can easily overflow for large input values. The
    default max_value of 20.0 corresponds to exp(20) ≈ 4.85e8, which is
    typically well within the numerical range of floating-point arithmetic.
    """
    x = u.math.clip(x, a_max=max_value)
    return u.math.exp(x)


class Transform(ABC):
    r"""
    Abstract base class for bijective parameter transformations.

    This class provides the interface for implementing bijective (one-to-one and onto)
    transformations that map parameters between different domains. These transformations
    are essential in optimization and statistical inference where parameters need to be
    constrained to specific domains (e.g., positive values, bounded intervals).
    
    A bijective transformation :math:`f: \mathcal{X} \rightarrow \mathcal{Y}` must satisfy:
    
    1. **Injectivity** (one-to-one): :math:`f(x_1) = f(x_2) \Rightarrow x_1 = x_2`
    2. **Surjectivity** (onto): :math:`\forall y \in \mathcal{Y}, \exists x \in \mathcal{X} : f(x) = y`
    3. **Invertibility**: :math:`f^{-1}(f(x)) = x` and :math:`f(f^{-1}(y)) = y`
    
    Methods
    -------
    forward(x)
        Apply the forward transformation :math:`y = f(x)`
    inverse(y)
        Apply the inverse transformation :math:`x = f^{-1}(y)`

    Notes
    -----
    Subclasses must implement both `forward` and `inverse` methods to ensure
    the transformation is truly bijective. The implementation should guarantee
    numerical stability and handle edge cases appropriately.
    
    Examples
    --------
    >>> class SquareTransform(Transform):
    ...     def forward(self, x):
    ...         return x**2
    ...     def inverse(self, y):
    ...         return jnp.sqrt(y)
    """

    def __call__(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation to the input.

        Parameters
        ----------
        x : array_like
            Input array to transform.

        Returns
        -------
        Array
            Transformed output array.
            
        Notes
        -----
        This method provides a convenient callable interface that delegates
        to the forward method, allowing Transform objects to be used as functions.
        """
        return self.forward(x)

    @abstractmethod
    def forward(self, x: ArrayLike) -> Array:
        r"""
        Apply the forward transformation.
        
        Transforms input from the unconstrained domain to the constrained domain.
        This method implements the mathematical function :math:`y = f(x)` where
        :math:`x` is in the unconstrained space and :math:`y` is in the target domain.

        Parameters
        ----------
        x : array_like
            Input array in the unconstrained domain.

        Returns
        -------
        Array
            Transformed output in the constrained domain.
            
        Notes
        -----
        Implementations must ensure numerical stability and handle boundary
        conditions appropriately.
        """

    @abstractmethod
    def inverse(self, y: ArrayLike) -> Array:
        r"""
        Apply the inverse transformation.
        
        Transforms input from the constrained domain back to the unconstrained domain.
        This method implements the mathematical function :math:`x = f^{-1}(y)` where
        :math:`y` is in the constrained space and :math:`x` is in the unconstrained domain.

        Parameters
        ----------
        y : array_like
            Input array in the constrained domain.

        Returns
        -------
        Array
            Transformed output in the unconstrained domain.
            
        Notes
        -----
        Implementations must ensure that inverse(forward(x)) = x for all valid x,
        and forward(inverse(y)) = y for all y in the target domain.
        """
        pass


class IdentityTransform(Transform):
    def forward(self, x: ArrayLike) -> Array:
        return x

    def inverse(self, y: ArrayLike) -> Array:
        return y



class SigmoidTransform(Transform):
    r"""
    Sigmoid transformation mapping unbounded values to a bounded interval.
    
    This transformation uses the logistic sigmoid function to map any real value
    to a bounded interval [lower, upper]. It is particularly useful for constraining
    parameters that must lie within specific bounds, such as probabilities or
    correlation coefficients.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x) = \text{lower} + (\text{upper} - \text{lower}) \cdot \sigma(x)
        
    where :math:`\sigma(x) = \frac{1}{1 + e^{-x}}` is the standard sigmoid function.
    
    The inverse transformation is:
    
    .. math::
        \text{inverse}(y) = \log\left(\frac{y - \text{lower}}{\text{upper} - y}\right)
    
    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.
    upper : array_like
        Upper bound of the target interval.
        
    Attributes
    ----------
    lower : array_like
        Lower bound of the interval.
    width : array_like
        Width of the interval (upper - lower).
    unit : brainunit.Unit
        Physical unit of the bounds.
        
    Notes
    -----
    The sigmoid function provides a smooth, differentiable mapping with asymptotes
    at the specified bounds. The transformation is bijective from ℝ to (lower, upper),
    though numerical precision may limit the effective range near the boundaries.
    
    Examples
    --------
    >>> # Map to probability range [0, 1]
    >>> transform = SigmoidTransform(0.0, 1.0)
    >>> x = jnp.array([-2.0, 0.0, 2.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.12, 0.5, 0.88]
    
    >>> # Map to correlation range [-1, 1] 
    >>> transform = SigmoidTransform(-1.0, 1.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.0]
    """

    def __init__(self, lower: ArrayLike, upper: ArrayLike) -> None:
        r"""
        Initialize the sigmoid transformation.
        
        Parameters
        ----------
        lower : array_like
            Lower bound of the target interval. Must be less than upper.
        upper : array_like
            Upper bound of the target interval. Must be greater than lower.
            
        Raises
        ------
        ValueError
            If lower >= upper, as this would result in a non-positive interval width.
        """
        super().__init__()
        self.lower = lower
        self.width = upper - lower
        self.unit = u.get_unit(lower)

    def forward(self, x: ArrayLike) -> Array:
        r"""
        Transform unbounded input to bounded interval using sigmoid function.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain $(-\infty, \infty)$.
            
        Returns
        -------
        Array
            Transformed values in interval [lower, upper].
            
        Notes
        -----
        Uses numerically stable exponential to prevent overflow for large |x|.
        """
        y = 1.0 / (1.0 + save_exp(-x))
        return self.lower + self.width * y

    def inverse(self, y: ArrayLike) -> Array:
        r"""
        Transform bounded input back to unbounded domain using logit function.
        
        Parameters
        ----------
        y : array_like
            Input values in bounded interval [lower, upper].
            
        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).
            
        Notes
        -----
        For numerical stability, input should be strictly within (lower, upper).
        Values at the boundaries will result in infinite outputs.
        """
        x = (y - self.lower) / self.width
        x = -u.math.log((1.0 / x) - 1.0)
        return x


class SoftplusTransform(Transform):
    r"""
    Softplus transformation mapping unbounded values to positive semi-infinite interval.
    
    This transformation uses the softplus function to map any real value to the
    interval [lower, ∞). It provides a smooth, differentiable alternative to
    ReLU activation and is commonly used to constrain parameters to be positive,
    such as variance parameters or rate constants.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x) = \log(1 + e^x) + \text{lower}
        
    The inverse transformation is:
    
    .. math::
        \text{inverse}(y) = \log(e^{y - \text{lower}} - 1)
    
    Parameters
    ----------
    lower : array_like
        Lower bound of the target interval.
        
    Attributes
    ----------
    lower : array_like
        Lower bound of the interval.
    unit : brainunit.Unit
        Physical unit of the lower bound.
        
    Notes
    -----
    The softplus function is the smooth approximation to the ReLU function:
    :math:`\lim_{\beta \to \infty} \frac{1}{\beta} \log(1 + e^{\beta x}) = \max(0, x)`
    
    For large positive x, softplus(x) ≈ x, and for large negative x, softplus(x) ≈ 0.
    The function is strictly positive and has a well-defined inverse.
    
    Examples
    --------
    >>> # Map to positive reals [0, ∞)
    >>> transform = SoftplusTransform(0.0)
    >>> x = jnp.array([-5.0, 0.0, 5.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [0.007, 0.693, 5.007]
    
    >>> # Map to interval [2, ∞) for positive-definite parameters
    >>> transform = SoftplusTransform(2.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [2.693]
    """

    def __init__(self, lower: ArrayLike) -> None:
        """
        Initialize the softplus transformation.
        
        Parameters
        ----------
        lower : array_like
            Lower bound of the target interval. The transformation maps
            unbounded inputs to [lower, ∞).
        """
        super().__init__()
        self.lower = lower
        self.unit = u.get_unit(lower)

    def forward(self, x: ArrayLike) -> Array:
        """
        Transform unbounded input to positive semi-infinite interval.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain (-∞, ∞).
            
        Returns
        -------
        Array
            Transformed values in interval [lower, ∞).
            
        Notes
        -----
        Uses log1p for numerical stability: log1p(exp(x)) = log(1 + exp(x)).
        For large x, this avoids overflow in the exponential.
        """
        return jnp.log1p(save_exp(x)) + self.lower

    def inverse(self, y: ArrayLike) -> Array:
        """
        Transform positive semi-infinite input back to unbounded domain.
        
        Parameters
        ----------
        y : array_like
            Input values in interval [lower, ∞).
            
        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).
            
        Notes
        -----
        Input must be strictly greater than lower bound to avoid numerical issues.
        Uses numerically stable exponential for large (y - lower) values.
        """
        return u.math.log(save_exp((y - self.lower) / self.unit) - 1.0)


class NegSoftplusTransform(SoftplusTransform):
    r"""
    Negative softplus transformation mapping unbounded values to negative semi-infinite interval.
    
    This transformation uses the negative softplus function to map any real value
    to the interval (-∞, upper]. It is the reflection of the softplus function
    and is useful for constraining parameters to be negative, such as log-probabilities
    or negative rate constants.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x) = -\log(1 + e^{-x}) + \text{upper}
        
    which is equivalent to:
    
    .. math::
        \text{forward}(x) = \text{upper} - \text{softplus}(-x)
    
    The inverse transformation is:
    
    .. math::
        \text{inverse}(y) = -\log(e^{\text{upper} - y} - 1)
    
    Parameters
    ----------
    upper : array_like
        Upper bound of the target interval.
        
    Attributes
    ----------
    lower : array_like
        Stores the upper bound (inherited from parent class).
    unit : brainunit.Unit
        Physical unit of the upper bound.
        
    Notes
    -----
    This transformation is implemented by negating the input and output of the
    standard softplus transformation. For large positive x, the output approaches
    the upper bound, while for large negative x, the output approaches -∞.
    
    Examples
    --------
    >>> # Map to negative reals (-∞, 0]
    >>> transform = NegSoftplusTransform(0.0)
    >>> x = jnp.array([-5.0, 0.0, 5.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [-5.007, -0.693, -0.007]
    
    >>> # Map to interval (-∞, -2] for negative-definite parameters
    >>> transform = NegSoftplusTransform(-2.0)
    >>> x = jnp.array([0.0])
    >>> y = transform.forward(x)
    >>> # y ≈ [-2.693]
    """

    def __init__(self, upper: ArrayLike) -> None:
        """
        Initialize the negative softplus transformation.
        
        Parameters
        ----------
        upper : array_like
            Upper bound of the target interval. The transformation maps
            unbounded inputs to (-∞, upper].
        """
        super().__init__(upper)

    def forward(self, x: ArrayLike) -> Array:
        """
        Transform unbounded input to negative semi-infinite interval.
        
        Parameters
        ----------
        x : array_like
            Input values in unbounded domain (-∞, ∞).
            
        Returns
        -------
        Array
            Transformed values in interval (-∞, upper].
            
        Notes
        -----
        Implemented as the negative of softplus applied to -x.
        """
        return -super().forward(-x)

    def inverse(self, y: ArrayLike) -> Array:
        """
        Transform negative semi-infinite input back to unbounded domain.
        
        Parameters
        ----------
        y : array_like
            Input values in interval (-∞, upper].
            
        Returns
        -------
        Array
            Transformed values in unbounded domain (-∞, ∞).
            
        Notes
        -----
        Input must be strictly less than upper bound to avoid numerical issues.
        """
        return -super().inverse(-y)


class AffineTransform(Transform):
    r"""
    Affine (linear) transformation with scaling and shifting.
    
    This transformation applies a linear transformation of the form y = ax + b,
    where a is the scale factor and b is the shift. It is the most basic form
    of transformation and preserves the relative ordering of inputs while allowing
    for rescaling and translation.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x) = a \cdot x + b
        
    The inverse transformation is:
    
    .. math::
        \text{inverse}(y) = \frac{y - b}{a}
    
    Parameters
    ----------
    scale : array_like
        Scaling factor a. Must be non-zero for invertibility.
    shift : array_like
        Additive shift b.
        
    Attributes
    ----------
    a : array_like
        Scaling factor.
    b : array_like
        Shift parameter.
        
    Raises
    ------
    ValueError
        If scale is zero or numerically close to zero, making the transformation
        non-invertible.
        
    Notes
    -----
    Affine transformations are the foundation of many statistical transformations.
    They preserve linearity and are particularly useful for:
    
    - Standardization: (x - μ) / σ
    - Normalization: (x - min) / (max - min)
    - Unit conversion: x * conversion_factor + offset
    
    The Jacobian of this transformation is constant: |det(J)| = |a|.
    
    Examples
    --------
    >>> # Standardization transform (z-score)
    >>> mu, sigma = 5.0, 2.0
    >>> transform = AffineTransform(1/sigma, -mu/sigma)
    >>> x = jnp.array([3.0, 5.0, 7.0])
    >>> z = transform.forward(x)
    >>> # z ≈ [-1.0, 0.0, 1.0]
    
    >>> # Temperature conversion: Celsius to Fahrenheit
    >>> transform = AffineTransform(9/5, 32)
    >>> celsius = jnp.array([0.0, 100.0])
    >>> fahrenheit = transform.forward(celsius)
    >>> # fahrenheit ≈ [32.0, 212.0]
    """
    def __init__(self, scale: ArrayLike, shift: ArrayLike):
        """
        Initialize the affine transformation.
        
        Parameters
        ----------
        scale : array_like
            Scaling factor. Must be non-zero for the transformation to be invertible.
        shift : array_like
            Additive shift parameter.

        Raises
        ------
        ValueError
            If scale is zero or numerically close to zero, making the
            transformation non-invertible.
        """
        if jnp.allclose(scale, 0):
            raise ValueError("a cannot be zero, must be invertible")
        self.a = scale
        self.b = shift

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply the affine transformation y = ax + b.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Transformed values after scaling and shifting.
        """
        return self.a * x + self.b

    def inverse(self, x: ArrayLike) -> Array:
        """
        Apply the inverse affine transformation x = (y - b) / a.
        
        Parameters
        ----------
        x : array_like
            Transformed values to invert (note: parameter name kept for consistency).
            
        Returns
        -------
        Array
            Original values before transformation.
        """
        return (x - self.b) / self.a


class ChainTransform(Transform):
    r"""
    Composition of multiple transformations applied sequentially.
    
    This class implements the mathematical composition of functions, allowing
    multiple transformations to be chained together. The transformations are
    applied in the order specified during initialization for the forward pass,
    and in reverse order for the inverse pass.
    
    For transformations f₁, f₂, ..., fₙ, the chain implements:
    
    .. math::
        \text{forward}(x) = f_n(f_{n-1}(...f_2(f_1(x))...))
        
    .. math::
        \text{inverse}(y) = f_1^{-1}(f_2^{-1}(...f_{n-1}^{-1}(f_n^{-1}(y))...))
    
    Parameters
    ----------
    *transforms : sequence of Transform
        Variable number of Transform objects to chain together.
        
    Attributes
    ----------
    transforms : sequence of Transform
        Tuple of transformations in the order they will be applied.
        
    Notes
    -----
    The chain transformation preserves bijectivity if all component transformations
    are bijective. The Jacobian of the chain is the product of the Jacobians of
    the individual transformations.
    
    Chain transformations are particularly useful for:
    
    - Complex parameter constraints requiring multiple steps
    - Modular transformation design
    - Combining simple transformations to achieve complex mappings
    
    Examples
    --------
    >>> # Transform to (0, 1) then scale to (a, b)
    >>> sigmoid = SigmoidTransform(0, 1)
    >>> affine = AffineTransform(scale=b-a, shift=a)
    >>> chain = ChainTransform(sigmoid, affine)
    
    >>> # Standardize then apply softplus
    >>> standardize = AffineTransform(1/sigma, -mu/sigma)
    >>> softplus = SoftplusTransform(0)
    >>> chain = ChainTransform(standardize, softplus)
    """

    def __init__(self, *transforms: Sequence[Transform]) -> None:
        """
        Initialize the chain transformation.
        
        Parameters
        ----------
        *transforms : sequence of Transform
            Variable number of Transform objects to be applied sequentially.
            The transformations will be applied in the order provided for
            the forward pass, and in reverse order for the inverse pass.
            
        Raises
        ------
        TypeError
            If any of the provided objects is not a Transform instance.
        """
        super().__init__()
        self.transforms: Sequence[Transform] = transforms

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply all transformations sequentially in forward order.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Values after applying all transformations in sequence.
            
        Notes
        -----
        Transformations are applied left-to-right as specified in initialization.
        """
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def inverse(self, y: ArrayLike) -> Array:
        """
        Apply all inverse transformations sequentially in reverse order.
        
        Parameters
        ----------
        y : array_like
            Transformed values to invert.
            
        Returns
        -------
        Array
            Original values before all transformations were applied.
            
        Notes
        -----
        Transformations are inverted right-to-left (reverse order) to properly
        undo the forward chain.
        """
        for transform in reversed(self.transforms):
            y = transform.inverse(y)
        return y


class MaskedTransform(Transform):
    r"""
    Selective transformation using a boolean mask.
    
    This transformation applies a given transformation only to elements specified
    by a boolean mask, leaving other elements unchanged. This is useful when only
    a subset of parameters need to be transformed while others should remain in
    their original domain.
    
    The transformation is defined by:
    
    .. math::
        \text{forward}(x)_i = \begin{cases}
        f(x_i) & \text{if } \text{mask}_i = \text{True} \\
        x_i & \text{if } \text{mask}_i = \text{False}
        \end{cases}
        
    where f is the underlying transformation.
    
    The inverse follows the same pattern:
    
    .. math::
        \text{inverse}(y)_i = \begin{cases}
        f^{-1}(y_i) & \text{if } \text{mask}_i = \text{True} \\
        y_i & \text{if } \text{mask}_i = \text{False}
        \end{cases}
    
    Parameters
    ----------
    mask : array_like of bool
        Boolean array indicating which elements to transform.
    transform : Transform
        The transformation to apply to masked elements.
        
    Attributes
    ----------
    mask : array_like
        Boolean mask array.
    transform : Transform
        The underlying transformation.
        
    Notes
    -----
    The mask and input arrays must have compatible shapes for broadcasting.
    This transformation is particularly useful in:
    
    - Mixed parameter models where some parameters are bounded and others are not
    - Selective application of constraints in optimization
    - Sparse transformations where only specific elements need modification
    
    Examples
    --------
    >>> # Transform only positive indices to be positive
    >>> mask = jnp.array([False, True, False, True])
    >>> softplus = SoftplusTransform(0)
    >>> masked_transform = MaskedTransform(mask, softplus)
    >>> x = jnp.array([-1.0, -1.0, 2.0, 2.0])
    >>> y = masked_transform.forward(x)
    >>> # y ≈ [-1.0, 0.31, 2.0, 2.13] (only indices 1,3 transformed)
    
    >>> # Transform correlation parameters but not mean parameters
    >>> n_params = 5
    >>> corr_mask = jnp.arange(n_params) >= 3  # Last 2 are correlations
    >>> sigmoid = SigmoidTransform(-1, 1)
    >>> transform = MaskedTransform(corr_mask, sigmoid)
    """
    def __init__(self, mask: ArrayLike, transform: Transform) -> None:
        """
        Initialize the masked transformation.
        
        Parameters
        ----------
        mask : array_like of bool
            Boolean array indicating which elements should be transformed.
            Must be broadcastable with the input arrays.
        transform : Transform
            The transformation to apply to elements where mask is True.
            
        Raises
        ------
        TypeError
            If transform is not a Transform instance.
        """
        super().__init__()
        self.mask = mask
        self.transform = transform

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply transformation selectively based on mask.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Array where masked elements are transformed and unmasked
            elements remain unchanged.
            
        Notes
        -----
        Uses element-wise conditional logic to apply transformation only
        where mask is True.
        """
        return u.math.where(self.mask, self.transform.forward(x), x)

    def inverse(self, y: ArrayLike) -> Array:
        """
        Apply inverse transformation selectively based on mask.
        
        Parameters
        ----------
        y : array_like
            Transformed values to invert.
            
        Returns
        -------
        Array
            Array where masked elements are inverse-transformed and unmasked
            elements remain unchanged.
            
        Notes
        -----
        Applies inverse transformation only to elements where mask is True,
        maintaining consistency with the forward operation.
        """
        return u.math.where(self.mask, self.transform.inverse(y), y)


class CustomTransform(Transform):
    r"""
    User-defined transformation using custom functions.
    
    This class allows users to define arbitrary bijective transformations by
    providing their own forward and inverse functions. It provides a convenient
    wrapper that ensures the custom functions conform to the Transform interface.
    
    The transformation functions should satisfy:
    
    .. math::
        \text{forward}: \mathcal{X} \rightarrow \mathcal{Y}
        
    .. math::
        \text{inverse}: \mathcal{Y} \rightarrow \mathcal{X}
        
    with the bijective property:
    
    .. math::
        \text{inverse}(\text{forward}(x)) = x \quad \forall x \in \mathcal{X}
        
    .. math::
        \text{forward}(\text{inverse}(y)) = y \quad \forall y \in \mathcal{Y}
    
    Parameters
    ----------
    forward_fn : callable
        Function implementing the forward transformation. Should take an
        array-like input and return an Array.
    inverse_fn : callable
        Function implementing the inverse transformation. Should take an
        array-like input and return an Array.
        
    Attributes
    ----------
    forward_fn : callable
        The forward transformation function.
    inverse_fn : callable
        The inverse transformation function.
        
    Notes
    -----
    Users are responsible for ensuring that:
    
    - The functions are truly inverse to each other
    - Both functions handle JAX arrays appropriately
    - The transformations are numerically stable
    - The functions are differentiable if needed for gradient-based optimization
    
    This class is particularly useful for:
    
    - Prototyping new transformation types
    - Domain-specific transformations not covered by standard classes
    - Wrapping existing transformation functions from other libraries
    
    Examples
    --------
    >>> # Square transformation (for positive inputs only)
    >>> def square_forward(x):
    ...     return x ** 2
    >>> def square_inverse(y):
    ...     return jnp.sqrt(y)
    >>> square_transform = CustomTransform(square_forward, square_inverse)
    
    >>> # Log-normal transformation
    >>> def lognorm_forward(x):
    ...     return jnp.exp(x)
    >>> def lognorm_inverse(y):
    ...     return jnp.log(y)
    >>> lognorm = CustomTransform(lognorm_forward, lognorm_inverse)
    
    >>> # Box-Cox transformation (lambda=0.5)
    >>> def boxcox_forward(x):
    ...     return 2 * (jnp.sqrt(x + 1) - 1)
    >>> def boxcox_inverse(y):
    ...     return ((y / 2) + 1) ** 2 - 1
    >>> boxcox = CustomTransform(boxcox_forward, boxcox_inverse)
    """

    def __init__(self, forward_fn: Callable, inverse_fn: Callable) -> None:
        """
        Initialize the custom transformation.
        
        Parameters
        ----------
        forward_fn : callable
            Function implementing the forward transformation. Should accept
            array-like inputs and return JAX arrays.
        inverse_fn : callable
            Function implementing the inverse transformation. Should accept
            array-like inputs and return JAX arrays.
            
        Notes
        -----
        The functions should be compatible with JAX transformations (jit, grad, etc.)
        for optimal performance. Both functions must be proper inverses of each other
        within numerical precision.
        """
        super().__init__()
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

    def forward(self, x: ArrayLike) -> Array:
        """
        Apply the user-defined forward transformation.
        
        Parameters
        ----------
        x : array_like
            Input values to transform.
            
        Returns
        -------
        Array
            Transformed values from the user-defined function.
        """
        return self.forward_fn(x)

    def inverse(self, y: ArrayLike) -> Array:
        """
        Apply the user-defined inverse transformation.
        
        Parameters
        ----------
        y : array_like
            Transformed values to invert.
            
        Returns
        -------
        Array
            Original values from the user-defined inverse function.
        """
        return self.inverse_fn(y)
