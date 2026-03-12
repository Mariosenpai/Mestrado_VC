import torch
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings

def sr_hifigan() -> int:
    return 22050
def mel_for_audio(mel_spectrogram):

    warnings.filterwarnings('ignore')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #print(f'Using {device} for inference')

    hifigan, vocoder_train_setup, denoiser = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan',verbose=False)

    hifigan.to(device)

    audio = hifigan(mel_spectrogram).float()

    return audio

if __name__ == "__main__":

    B = 2
    T = 260
    F = 80
    mel = torch.randn(T, F, B).to('cuda')

    audio = mel_for_audio(mel)
    Audio(audio, sr_hifigan())
