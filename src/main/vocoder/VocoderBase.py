import numpy as np
import torch


class VocoderBase:

    def __init__(self):
        pass

    def inference(self, mel_spectogram:torch.Tensor) -> torch.Tensor:
        """Deve ser sobrescrita"""
        pass
