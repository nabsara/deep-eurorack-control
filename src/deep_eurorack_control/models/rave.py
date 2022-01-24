


class RAVE:

    def __init__(self):

        self.sampling_rate = 48000
        self.n_band = 16
        self.latent_dim = 128
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.multi_band_decomposition = None  # PQMF

    def train_stage_1_repr_leaning(self):
        pass

    def train_stage_2_adv_fine_tuning(self):
        pass
