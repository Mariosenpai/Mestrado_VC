import torch
import torchaudio
from matplotlib import cm
from scipy import signal
import numpy as np
import librosa
import librosa.display
import scipy.ndimage
from main.config import image_size
# from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression for audio signals"""
    return torch.log(torch.clamp(x, min=clip_val) * C)
def mel_spectogram(
    sample_rate,
    hop_length,
    win_length,
    n_fft,
    n_mels,
    f_min,
    f_max,
    power,
    normalized,
    min_max_energy_norm,
    norm,
    mel_scale,
    compression,
    audio,
):
    """calculates MelSpectrogram for a raw audio signal

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_fft : int
        Size of FFT.
    n_mels : int
        Number of mel filterbanks.
    f_min : float
        Minimum frequency.
    f_max : float
        Maximum frequency.
    power : float
        Exponent for the magnitude spectrogram.
    normalized : bool
        Whether to normalize by magnitude after stft.
    min_max_energy_norm : bool
        Whether to normalize by min-max
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    compression : bool
        whether to do dynamic range compression
    audio : torch.Tensor
        input audio signal

    Returns
    -------
    mel : torch.Tensor
    rmse : torch.Tensor
    """
    from torchaudio import transforms

    audio_to_mel = transforms.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        power=power,
        normalized=normalized,
    ).to(audio.device)

    mel_scale = transforms.MelScale(
        sample_rate=sample_rate,
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        norm=norm,
        mel_scale=mel_scale,
    ).to(audio.device)
    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    assert mel.dim() == 2
    assert mel.shape[0] == n_mels
    rmse = torch.norm(mel, dim=0)

    if min_max_energy_norm:
        rmse = (rmse - torch.min(rmse)) / (torch.max(rmse) - torch.min(rmse))

    if compression:
        mel = dynamic_range_compression(mel)

    return mel, rmse

def gerar_mel_spectogram(uri: torch.Tensor | str, rate_original: int = 22050, rate_alvo: int = 22050, n_mels: int = 80,
                         n_fft: int = 1024, hop_length: int = 256, win_length: int = 512) -> torch.Tensor:
    if isinstance(uri, str):
        signal, rate_original = torchaudio.load(uri)
    else:
        signal = uri

    resampler = torchaudio.transforms.Resample(orig_freq=rate_original, new_freq=rate_alvo)

    signal = resampler(signal)

    spectrogram, _ = mel_spectogram(
        audio=signal.squeeze(),
        sample_rate=rate_alvo,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        n_fft=n_fft,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    return spectrogram


def padding(image: torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        _, h, w = image.shape
    else:
        return 0, 0, 0
    max_dim = max(w, h)
    new_image = torch.zeros(2, max_dim, max_dim)
    x_center = (max_dim - w) // 2
    y_center = (max_dim - h) // 2

    new_image[:, y_center:y_center + h, x_center:x_center + w] = image

    return new_image


def np_para_spectograma(audio_np: np.array) -> np.array:
    spectrograma = librosa.stft(np.asarray(audio_np), n_fft=image_size, win_length=image_size,
                                window=signal.windows.hamming(image_size))

    spectrograma = librosa.amplitude_to_db(abs(spectrograma))

    return spectrograma


def normalizar_spectograma(spectrograma: np.array) -> np.array:
    MIN = spectrograma.min()
    MAX = spectrograma.max()
    spectrograma_normalizado = (spectrograma - MIN) / (MAX - MIN)

    return spectrograma_normalizado


def pega_spectograma(audio_np: np.array) -> np.array:
    '''

    :param audio_np:
    :return: mel_espectrograma (np.array,np.array)
    '''
    spectrogram = np_para_spectograma(audio_np)
    return normalizar_spectograma(spectrogram)


def transformar_audio_np_em_spectograma(audio_np: np.array) -> torch.Tensor:
    '''

    :param audio_np: Array numpy com formato específico, como (1, np.array).
    :param image_size: Novo tamanho da imagem (tamanho desejado para a altura e largura do espectrograma).
    :param pad: Se True, aplica preenchimento para ajustar o tamanho.
    :return: Um tensor redimensionado do espectrograma [1, image_size, image_size].
    '''

    train_spectrogram = pega_spectograma(audio_np)

    train_spectrogram = (train_spectrogram - train_spectrogram.min()) / (
            train_spectrogram.max() - train_spectrogram.min())

    return train_spectrogram


def criar_mel_espectrograma_para_HiFiGAN(espectograma: np.array, frequencia: int = 80,
                                         tempo: int = 512) -> torch.Tensor:
    # Simulando um Mel Spectrogram de 513x629
    mel_spec = espectograma  # Substitua pelo seu espectrograma real

    # Calcular os fatores de escala
    scale_freq = frequencia / espectograma.shape[0]
    scale_time = tempo / espectograma.shape[1]

    # Redimensionar com interpolação bicúbica
    mel_spec_resized = scipy.ndimage.zoom(mel_spec, (scale_freq, scale_time), order=3)  # order=3 → Bicúbico

    return mel_spec_resized


def redimencionar_spectorgama(espectograma: np.array, image_size: int) -> np.array:
    '''

    :param audio_np: (x,y)
    :param image_size: 1024 -> viraria (1024x1024)
    :return:
    '''
    # Simulando um Mel Spectrogram de 513x629
    mel_spec = espectograma  # Substitua pelo seu espectrograma real

    # Calcular os fatores de escala
    scale_freq = image_size / espectograma.shape[0]
    scale_time = image_size / espectograma.shape[1]

    # Redimensionar com interpolação bicúbica
    mel_spec_resized = scipy.ndimage.zoom(mel_spec, (scale_freq, scale_time), order=3)  # order=3 → Bicúbico

    return mel_spec_resized


def redimensionar_audio_para_tamanho_original(audio_original, shape_esperado):
    # Calcular os fatores de escala inversos
    scale_freq = shape_esperado[0] / audio_original.shape[0]
    scale_time = shape_esperado[1] / audio_original.shape[1]

    # Redimensionar de volta para (513, 629)
    mel_spec_original = scipy.ndimage.zoom(audio_original, (scale_freq, scale_time),
                                           order=3)  # order=3 → Interpolação bicúbica

    return mel_spec_original


def mel_to_rgb(mel):
    # normaliza para [0,1]
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # aplica colormap
    cmap = cm.get_cmap("magma")
    mel_rgb = cmap(mel_norm)[:, :, :3]  # remove alpha

    # converte para uint8
    mel_rgb = (mel_rgb * 255).astype(np.uint8)

    return mel_rgb