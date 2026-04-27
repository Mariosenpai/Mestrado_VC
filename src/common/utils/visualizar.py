import numpy as np
import torch
from matplotlib import pyplot as plt


def vizualizar_spectrogram(spectrogram: torch.Tensor):
    plt.figure(figsize=(10,12))
    res_mel = spectrogram.detach().cpu().numpy()
    plt.imshow(res_mel, origin='lower')
    plt.xlabel('time')
    plt.ylabel('frequency')
    _=plt.title('Spectrogram')

def audio_duracao(audio: torch.Tensor, sr: int) -> float:
    """
    Retorna a duração do áudio em segundos.

    Parâmetros:
      audio: Caminho para o arquivo de áudio.

    Retorna:
      Duração do áudio (em segundos).
    """

    # Calcula a duração dividindo o número de frames pela taxa de amostragem
    num_samples = audio.shape[-1]
    duration = num_samples / sr
    return duration

def spectograma_3( eletronico: np.array, gerado: np.array, original: np.array , lista_titulos : list[str]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))

    axes[0].imshow(eletronico, cmap='viridis', origin='lower')
    axes[0].set_title(lista_titulos[0])

    axes[1].imshow(gerado, cmap='viridis', origin='lower')
    axes[1].set_title(lista_titulos[1])

    axes[2].imshow(original, cmap='viridis', origin='lower')
    axes[2].set_title(lista_titulos[2])

    plt.tight_layout()
    plt.show()
