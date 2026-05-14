import os

import librosa
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import soundfile as sf


def generate_wav(mel_output,output_path, j,sr=22050):
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = np.load(r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\src\common\vocoder\stats.npy")
    scaler.n_samples_seen_ = 80

    mel_spec = np.power(10.0, scaler.inverse_transform(mel_output)).T

    mel_basis = librosa.filters.mel(sr=22050,n_fft=1024,n_mels=80,fmin=80,fmax=7600)

    mel_to_linear = np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec))

    gl_lb = librosa.griffinlim(mel_to_linear,n_iter=32, hop_length=256, win_length=None or 1024)

    sf.write(os.path.join(output_path, str(j) + ".wav"), gl_lb, sr, "PCM_16")

    return gl_lb


if __name__ == "__main__":
    B = 2
    T = 260
    F = 80
    mel = torch.randn(T, F).detach().numpy()
    generate_wav(mel,"./", 0)