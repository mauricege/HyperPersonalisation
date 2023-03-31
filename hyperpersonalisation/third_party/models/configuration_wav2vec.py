"""wav2vec model congfiguration"""

from transformers import Wav2Vec2Config

class Wav2Vec2Config(Wav2Vec2Config):
    def __init__(self, train_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.train_adapters = train_adapters
