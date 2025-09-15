from typing import Union
import brainstate
import brainunit as u
from brainstate.nn._dynamics import maybe_init_prefetch
import jax.numpy as jnp

Prefetch = Union[brainstate.nn.PrefetchDelayAt, brainstate.nn.PrefetchDelay, brainstate.nn.Prefetch]

__all__ = [
    'HopfCoupling',
]

class HopfCoupling(brainstate.nn.Module):
    r"""
    HopfCoupling implements diffusive (state-difference) coupling for complex Hopf oscillators.
    .. math::
    I_i = K \sum_j g_{ij}{(x_j(t-D_{ij}) - x_i(t)}

    Parameters
    ----------
    I_i : output current injected into oscillator i
    g_{ij} : entry of the connection matrix (1-D COO or 2-D dense)
    x_j(t-D_{ij}) : delayed state of source node j (supplied by coupled_delayed)
    x_i(t) : current state of target node i (supplied by coupled_current)
    K : global gain scaling the whole sum
    The class accepts two Prefetch objects:
    coupled_delayed : delivers source activities after their individual delays
    coupled_current : delivers target activities at the present time step
    Both tensors must have the same trailing dimension (number of nodes).

    """

    def __init__(
        self,
        coupled_delayed: Prefetch,   
        coupled_current: Prefetch,

        conn: brainstate.typing.Array,
        K : float = 1.0
    ):
        super().__init__()
        assert isinstance(coupled_delayed, Prefetch), f'x_imag must be a Prefetch. But got {type(coupled_delayed)}.'
        assert isinstance(coupled_current, Prefetch), f'x_real must be a Prefetch. But got {type(coupled_current)}.'
        self.coupled_delayed = coupled_delayed
        self.coupled_current = coupled_current 
        self.K = K

        # Connection matrix
        self.conn = u.math.asarray(conn)
        assert self.conn.ndim in (1, 2), f'Only support 1d, 2d connection matrix. But we got {self.conn.ndim}d.'

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.coupled_delayed)
        maybe_init_prefetch(self.coupled_current)

    def update(self):
        delayed_val = self.coupled_delayed()
        current_val = self.coupled_current()
        # delayed_val_reshaped = delayed_val.reshape(self.conn.shape)
        if self.conn.ndim == 1:
            current_val_broadcasted = jnp.tile(current_val, self.conn.shape[0])
            coupled = self.conn * (delayed_val - current_val_broadcasted)
        elif self.conn.ndim == 2:
            delayed_val_reshaped = delayed_val.reshape(self.conn.shape)
            current_val_expanded = jnp.expand_dims(current_val, axis=1) # 形状: (80, 1)
            coupled = self.conn * (delayed_val_reshaped - current_val_expanded)
        else:
            raise NotImplementedError(f'Only support 1d, 2d connection matrix. But we got {self.conn.ndim}d.')

        return self.K * coupled.sum(axis=-1)