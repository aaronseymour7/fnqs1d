import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.numpy as jnp
from einops import rearrange

def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)

from functools import partial
@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1]**0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)

class FMHA(nn.Module):
    d_model : int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.v = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64)
        if self.transl_invariant:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff), jnp.float64)
            if self.two_dimensional:
                sq_L_eff = int(self.L_eff**0.5)
                assert sq_L_eff * sq_L_eff == self.L_eff
                self.J = roll2d(self.J, jnp.arange(sq_L_eff), jnp.arange(sq_L_eff))
                self.J = self.J.reshape(self.h, -1, self.L_eff)
            else:
                # 1D translationally-invariant attention kernel: J has shape (h, L_eff, L_eff)
                # built by cyclically rolling each head's L_eff-length kernel row.
                self.J = jax.vmap(roll, (None, 0), out_axes=1)(self.J, jnp.arange(self.L_eff))
        else:
            self.J = self.param("J", nn.initializers.xavier_uniform(), (self.h, self.L_eff, self.L_eff), jnp.float64)
        self.W = nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64, dtype=jnp.float64)

    def __call__(self, x):
        v = self.v(x)
        v = rearrange(v, 'batch L_eff (h d_eff) -> batch L_eff h d_eff', h=self.h)
        v = rearrange(v, 'batch L_eff h d_eff -> batch h L_eff d_eff')
        x = jnp.matmul(self.J, v)
        x = rearrange(x, 'batch h L_eff d_eff  -> batch L_eff h d_eff')
        x = rearrange(x, 'batch L_eff h d_eff ->  batch L_eff (h d_eff)')
        x = self.W(x)
        return x
