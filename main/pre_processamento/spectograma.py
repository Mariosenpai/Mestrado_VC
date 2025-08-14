import torch
import torchaudio
from scipy import signal
import numpy as np
import librosa
import librosa.display
import scipy.ndimage
from main.config import image_size
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram


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
