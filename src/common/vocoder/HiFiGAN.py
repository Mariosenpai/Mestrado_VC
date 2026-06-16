import warnings

import torch

from src.common.vocoder.VocoderBase import VocoderBase


class HiFiGAN(VocoderBase):

    def __init__(self):
        VocoderBase.__init__(self)
        self.hifigan, self.vocoder_train_setup, self.denoiser = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub',
            'nvidia_hifigan',
            verbose=False
        )

    def sr_vocoder(self) -> int:
        return 22050

    def inference(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        warnings.filterwarnings('ignore')

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.hifigan.to(device)

        audio = self.hifigan(mel_spectrogram).float()

        return audio
