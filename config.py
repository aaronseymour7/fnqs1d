from transformers import PretrainedConfig


class ViTFNQSConfig(PretrainedConfig):
    model_type = "vit_fnqs"

    def __init__(
        self,
        L_eff=20,
        num_layers=4,
        d_model=32,
        heads=4,
        b=1,
        complex: bool = True,
        disorder: bool = False,
        tras_inv=True,
        two_dim=False,     # <-- default flipped to False here: this config is for the 1D chain
        **kwargs,
    ):
        self.L_eff = L_eff
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.b = b
        self.complex = complex
        self.disorder = disorder
        self.tras_inv = tras_inv
        self.two_dim = two_dim
        super().__init__(**kwargs)
