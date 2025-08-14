import numpy as np
import torch
from matplotlib import pyplot as plt

from main.arquitetura.auto_encoder.fit.treinamento.CVMPT import pegar_audio_para_ruido

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


def spectograma_CVMPT(audio_np, sr, generator):
    '''
    criar um o audio original redimensionado, o audio original com ruido e o auido processado onde o ruido fui tirado

    :param audio_np: (int,) exemplo de entrada (16000,)
    :param sr: int - sampling rate
    :param generator: o gerador precisa ser iniciado como 'cpu'
    :return: shape da saida (  )
    '''

    batch_com_ruido = []

    audio_sem_ruido, audio_com_ruido = pegar_audio_para_ruido(audio_np, sr)

    batch_com_ruido.append(torch.tensor(audio_com_ruido))

    batch_com_ruido = torch.stack(batch_com_ruido).to('cpu')

    # batch_com_ruido = torch.tensor(np.expand_dims(batch_com_ruido, axis=0))

    audio_processado = generator(batch_com_ruido).to('cpu')
    audio_processado = audio_processado.detach().numpy().reshape(1024, 1024)

    return audio_sem_ruido[0], batch_com_ruido[0][0], torch.tensor(audio_processado)
