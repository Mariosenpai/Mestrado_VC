import numpy as np

from src.common.metricas.SNR import calculate_snr_tensor
from src.common.metricas.asr import _wer, _wer_with_one_trans
from src.common.metricas.lsd_mel import lsd_mel
from src.common.metricas.mcd import _mcd
from src.common.metricas.ssim import _ssim
from src.common.metricas.psnr import _PSNR
from src.common.metricas.lpips import _lpips
from src.common.metricas.mcd import compare_mel


class Metricas:

    def __init__(self, mcd, snr, psnr):
        self.mcd = mcd
        self.snr = snr
        self.psnr = psnr


def metricas_avalicao_model(mel_clean, mel_noise) -> Metricas:
    mcd_log = mcd(mel_clean, mel_noise)
    snr_log = snr(mel_clean, mel_noise)
    psnr_log = psnr(mel_clean, mel_noise)

    return Metricas(mcd_log, snr_log, psnr_log)


def wer(x, y, sr_x, sr_y):
    return _wer(x, y, sr_x, sr_y)


def wer_with_trans(trans: str, y, sr_y):
    return _wer_with_one_trans(trans, y, sr_y)


def lsd(mel_clean, mel_noise):
    return lsd_mel(mel_clean, mel_noise)


def mcd(mel_clean, mel_noise):
    return compare_mel(mel_clean, mel_noise)


# Structured Similarity Index Metric
def ssim(mel_clean: np.array, mel_noise: np.array):
    return _ssim(mel_clean, mel_noise)


# Peak Signal-to-Noise Ratio
def psnr(mel_clean: np.array, mel_noise: np.array):
    return _PSNR(mel_clean, mel_noise)


# Learned Perceptual Image Patch Similarity
def lpips(mel_clean: np.array, mel_noise: np.array):
    return _lpips(mel_clean, mel_noise)


def snr(mel_clean: np.array, mel_noise: np.array):
    return calculate_snr_tensor(mel_clean, mel_noise)
