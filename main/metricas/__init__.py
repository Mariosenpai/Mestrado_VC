import numpy as np

from main.metricas.asr import _wer
from main.metricas.lsd_mel import lsd_mel
from main.metricas.mcd import _mcd
from main.metricas.ssim import _ssim
from main.metricas.psnr import _PSNR
from main.metricas.lpips import _lpips
from main.metricas.mcd import compare_mel


def wer(x, y, sr):
    return _wer(x, y, sr)


def lsd(mel_clean, mel_noise):
    return lsd_mel(mel_clean, mel_noise)


def mcd(mel_clean, mel_noise):
    return compare_mel(mel_clean, mel_noise)


# Structured Similarity Index Metric
def ssim(mel_clean:np.array, mel_noise:np.array):
    return _ssim(mel_clean, mel_noise)



# Peak Signal-to-Noise Ratio
def psnr(mel_clean:np.array, mel_noise:np.array):
    return _PSNR(mel_clean, mel_noise)


# Learned Perceptual Image Patch Similarity
def lpips(mel_clean:np.array, mel_noise:np.array):
    return _lpips(mel_clean, mel_noise)
