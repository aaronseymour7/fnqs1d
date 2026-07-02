from transformers import FlaxPreTrainedModel
import jax.numpy as jnp
from .transformer_fnqs import ViTFNQS
from .config import ViTFNQSConfig


class ViTFNQSModel(FlaxPreTrainedModel):
    config_class = ViTFNQSConfig

    def __init__(
            self,
            config: ViTFNQSConfig,
            input_shape=(jnp.zeros((1, 20)), jnp.zeros((1, 1, 1))),
            seed: int = 0,
            dtype: jnp.dtype = jnp.float64,
            _do_init: bool = True,
            **kwargs,
    ):
        self.model = ViTFNQS(L_eff=config.L_eff,
                              num_layers=config.num_layers,
                              d_model=config.d_model,
                              heads=config.heads,
                              b=config.b,
                              complex=config.complex,
                              disorder=config.disorder,
                              transl_invariant=config.tras_inv,
                              two_dimensional=config.two_dim,
                              )
        if "return_z" not in kwargs:
            self.return_z = False
        else:
            self.return_z = kwargs["return_z"]
        super().__init__(config, ViTFNQS, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def __call__(self, params, spins, coups):
        return self.model.apply(params, spins, coups, return_z=self.return_z)

    def init_weights(self, rng, input_shape):
        return self.model.init(rng, *input_shape)
