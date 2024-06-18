import flax.linen as nn
import jax.numpy as jnp


class ScaleNorm(nn.Module):
    """
    normalize by phi = phi / |phi|
    """

    def __call__(self, x):
        l2_norm = jnp.linalg.norm(
            x, axis=(tuple(range(1, len(x.shape)))), keepdims=True
        )
        return x / (l2_norm + 1e-6)


if __name__ == "__main__":
    """testing"""
    x = jnp.array([[1, 2], [3, 4]])
    print(x)
    x = ScaleNorm()(x)
    print(x)
