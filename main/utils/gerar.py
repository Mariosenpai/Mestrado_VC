import librosa
import numpy as np
import torch
from scipy import signal
from torch import Tensor

from main.arquitetura.auto_encoder.modelo.auto_encoder import GenerativeNetworkAuto
from main.arquitetura.auto_encoder.fit.treinamento.LE import pegar_spectograma_natural_e_eletronico_redimencionado
from main.config import image_size


def spectograma_natural_eletronico_gerado_LE(
        tupla_LE: tuple[Tensor,Tensor],
        generator: GenerativeNetworkAuto
) -> tuple[Tensor, Tensor, Tensor]:
    """

    :param spectrograma: (np.array, np.array)
    :param generator:
    :return: natural, eletronico , espectograma gerado
    """
    ln = []
    le = []

    natural, eletronica = pegar_spectograma_natural_e_eletronico_redimencionado(tupla_LE)

    ln.append(natural)
    le.append(eletronica)

    le = torch.stack(le).to('cpu')

    generated_natural = generator(le).to('cpu')
    gen_natural = generated_natural.detach().numpy().reshape(1024, 1024)

    # generated = np.concatenate((natural.cpu().detach().numpy().reshape(1024,1024), gen_natural), axis=0)

    return ln[0][0], le[0][0], gen_natural


def info_audio(audio: np.array) -> tuple[np.array, int, int, np.array]:
    stft = librosa.stft(np.asarray(audio), n_fft=image_size, win_length=image_size,
                        window=signal.windows.hamming(image_size))
    spectrogram = librosa.amplitude_to_db(abs(stft))

    phase = np.angle(stft)
    MIN = spectrogram.min()
    MAX = spectrogram.max()
    spectrogram = (spectrogram - MIN) / (MAX - MIN)

    return phase, MIN, MAX, spectrogram

