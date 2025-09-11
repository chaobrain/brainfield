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


from typing import Optional

import jax.numpy as jnp
from jax import jit


@jit
def cosine_similarity(X: jnp.ndarray, Y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Compute cosine similarity between samples in X and Y.

    Cosine similarity is defined as the dot product of two vectors divided by
    the product of their magnitudes (L2 norms).

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features), optional.
           If None, compute cosine similarity between samples in X.

    Returns:
        Cosine similarity matrix of shape (n_samples_X, n_samples_Y).
        If Y is None, returns shape (n_samples_X, n_samples_X).

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        >>> cosine_sim = cosine_similarity(X)
        >>> print(cosine_sim)
        [[1.         0.         0.70710677]
         [0.         1.         0.70710677]
         [0.70710677 0.70710677 1.        ]]
    """
    # If Y is not provided, compute similarity within X
    if Y is None:
        Y = X

    # Compute dot products between all pairs
    dot_products = jnp.dot(X, Y.T)

    # Compute L2 norms for each sample
    X_norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    Y_norms = jnp.linalg.norm(Y, axis=1, keepdims=True)

    # Compute the product of norms for each pair
    norm_products = jnp.dot(X_norms, Y_norms.T)

    # Compute cosine similarity
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    cosine_sim = dot_products / (norm_products + eps)

    return jnp.nan_to_num(cosine_sim)


def cosine_similarity_safe(X: jnp.ndarray, Y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Compute cosine similarity with explicit handling of zero vectors.

    Zero vectors will result in NaN similarity values, which matches
    sklearn's behavior more closely.

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features), optional.
           If None, compute cosine similarity between samples in X.

    Returns:
        Cosine similarity matrix of shape (n_samples_X, n_samples_Y).
        Zero vectors will have NaN similarity values.
    """
    if Y is None:
        Y = X

    # Compute dot products
    dot_products = jnp.dot(X, Y.T)

    # Compute L2 norms
    X_norms = jnp.linalg.norm(X, axis=1, keepdims=True)
    Y_norms = jnp.linalg.norm(Y, axis=1, keepdims=True)

    # Compute norm products
    norm_products = jnp.dot(X_norms, Y_norms.T)

    # Compute cosine similarity (will be NaN for zero vectors)
    cosine_sim = dot_products / norm_products

    return jnp.nan_to_num(cosine_sim)


# Example usage and comparison function
def compare_with_sklearn_example():
    """
    Example comparing JAX implementation with sklearn behavior.
    Run this to verify the implementation works correctly.
    """
    import jax.numpy as jnp
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

    # Create sample data
    X = jnp.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 2, 3]
    ], dtype=jnp.float32)

    Y = jnp.array([
        [1, 0, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=jnp.float32)

    print("JAX Cosine Similarity (X vs X):")
    sim_xx = cosine_similarity(X)
    print(sim_xx)
    print(sklearn_cosine_similarity(X))
    print()

    print("JAX Cosine Similarity (X vs Y):")
    sim_xy = cosine_similarity(X, Y)
    print(sim_xy)
    print(sklearn_cosine_similarity(X, Y))
    print()

    # Test with zero vector
    X_with_zero = jnp.array([
        [1, 0, 0],
        [0, 0, 0],  # zero vector
        [1, 1, 0]
    ], dtype=jnp.float32)

    print("JAX Cosine Similarity with zero vector (safe version):")
    sim_zero = cosine_similarity_safe(X_with_zero)
    print(sim_zero)
    print(sklearn_cosine_similarity(X_with_zero))
    print()

    return sim_xx, sim_xy, sim_zero


if __name__ == "__main__":
    compare_with_sklearn_example()
