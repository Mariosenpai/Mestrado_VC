import time

import numpy as np

from src.common.metricas.SNR import calculate_snr_tensor
from src.common.metricas.asr import _wer, _wer_with_one_trans
from src.common.metricas.f0_rmse import _f0_rmse, _f0_rmse_log
from src.common.metricas.lsd_mel import lsd_mel
from src.common.metricas.mcd import _mcd
from src.common.metricas.mosnet import Mosnet
from src.common.metricas.msd import _msd
from src.common.metricas.ssim import _ssim
from src.common.metricas.psnr import _PSNR
from src.common.metricas.lpips import _lpips
from src.common.metricas.mcd import compare_mel


class Metricas:

    def __init__(self, mcd = 0.0, snr= 0.0, psnr= 0.0,f0_rmse= 0.0,f0_rmse_log= 0.0,msd= 0.0,mosnet= 0.0):
        self.mcd = mcd
        self.snr = snr
        self.psnr = psnr
        self.f0_rmse = f0_rmse
        self.f0_rmse_log = f0_rmse_log
        self.msd = msd
        self.mosnet = mosnet

        self.model_mosnet = Mosnet()

    def get_metricas(self,step):
        print("mcd:", self.mcd/step,"\n",
              "snr:",self.snr/step,"\n",
              "psnr:",self.psnr/step,"\n",
              "f0_rmse_log:",self.f0_rmse_log/step,"\n",
              "msd:",self.msd/step,"\n",
              "mosnet:",self.mosnet/step,"\n")

    def update(
            self,
            mel_org,
            wav_org,
            mel_clean,
            wav_clean,
            sample_rate: int,
            test_time:bool = False
    ):
        inicio_total = time.perf_counter()

        # ==========================================
        # MÉTRICAS DE ÁUDIO
        # ==========================================
        inicio = time.perf_counter()

        f0_rmse_log_log, msd_log = metricas_para_audio(
            wav_org,
            wav_clean,
            sample_rate
        )

        tempo_audio = time.perf_counter() - inicio

        # ==========================================
        # MÉTRICAS DE MEL
        # ==========================================
        inicio = time.perf_counter()

        mcd_log, snr_log, psnr_log = metricas_para_mel(
            mel_org,
            mel_clean
        )

        tempo_mel = time.perf_counter() - inicio

        # ==========================================
        # MOSNET
        # ==========================================
        inicio = time.perf_counter()

        mos_log = metricas_naturalidade_mos(
            self.model_mosnet,
            wav_clean
        )

        tempo_mosnet = time.perf_counter() - inicio

        # ==========================================
        # ATUALIZAÇÃO DAS MÉTRICAS
        # ==========================================
        inicio = time.perf_counter()

        self.mcd += mcd_log
        self.snr += snr_log
        self.psnr += psnr_log
        self.f0_rmse += 0
        self.f0_rmse_log += f0_rmse_log_log
        self.msd += msd_log
        self.mosnet += mos_log

        tempo_update = time.perf_counter() - inicio

        # ==========================================
        # TEMPO TOTAL
        # ==========================================
        tempo_total = time.perf_counter() - inicio_total
        if test_time:
            print("\n================ TEMPO MÉTRICAS ================")
            print(f"metricas_para_audio       : {tempo_audio:.4f}s")
            print(f"metricas_para_mel         : {tempo_mel:.4f}s")
            print(f"metricas_naturalidade_mos : {tempo_mosnet:.4f}s")
            print(f"atualização dos valores   : {tempo_update:.4f}s")
            print("------------------------------------------------")
            print(f"TOTAL                     : {tempo_total:.4f}s")
            print("------------------------------------------------")
            print(f"MDC         : {mcd_log:.4f}s")
            print(f"SNR         : {snr_log:.4f}s")
            print(f"PSNR        : {psnr_log:.4f}s")
            print(f"F0-RMSE-LOG : {f0_rmse_log_log:.4f}s")
            print(f"MSD         : {msd_log:.4f}s")
            print(f"MOSNET      : {mos_log:.4f}s")
            print("================================================\n")



def metricas_para_mel(mel_clean, mel_noise) -> tuple:
    mcd_log = mcd(mel_clean, mel_noise)
    snr_log = snr(mel_clean, mel_noise)
    psnr_log = psnr(mel_clean, mel_noise)

    return mcd_log, snr_log, psnr_log

def metricas_para_audio(wav_clean:list[np.array], wav_noise:list[np.array], sample_rate:int) -> tuple:
    wav_noise = [wav_noise]
    wav_clean = [wav_clean]
    # f0_rmse_result = f0_rmse(wav_noise, wav_clean, sample_rate)
    f0_rmse_log_result = f0_rmse_log(wav_noise, wav_clean, sample_rate)
    msd_result = msd(wav_noise, wav_clean, sample_rate)

    return  f0_rmse_log_result, msd_result

def metricas_naturalidade_mos(model_mosnet,wav):
    mos_log = model_mosnet.inference(wav)
    return mos_log

def metricas_geral(mel_clean, wav_clean, mel_noise, wav_noise, sample_rate:int) -> Metricas:
    f0_rmse_log, f0_rmse_log_log, msd_log = metricas_para_audio(wav_clean, wav_noise, sample_rate)
    mcd_log, snr_log, psnr_log = metricas_para_mel(mel_clean, mel_noise)
    mos_log = metricas_naturalidade_mos(wav_clean)

    return Metricas(
        mcd=mcd_log,
        snr=snr_log,
        psnr=psnr_log,
        f0_rmse=f0_rmse_log,
        f0_rmse_log=f0_rmse_log_log,
        msd=msd_log,
        mosnet=mos_log
    )


def mosnet(wav) -> float:
    return  Mosnet().inference(wav)
def msd(x:list[np.array], y:list[np.array], sample_rate) -> float:
    return _msd(x, y, sample_rate)
def f0_rmse(x:list[np.array],y:list[np.array],sample_rate:int):
    return _f0_rmse(x,y,sample_rate)
def f0_rmse_log(x:list[np.array],y:list[np.array],sample_rate:int):
    return _f0_rmse_log(x,y,sample_rate)

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
