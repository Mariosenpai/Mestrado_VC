import numpy as np
import torch


class VocoderBase:

    def __init__(self):
        pass

    def sr_vocoder(self):
        pass
    def inference(self, mel_spectogram:torch.Tensor) -> torch.Tensor:
        """
        Deve ser sobrescrita
        mel_spectogram : torch.Tensor(Batch,Frequencia,Time)
        """
        pass
