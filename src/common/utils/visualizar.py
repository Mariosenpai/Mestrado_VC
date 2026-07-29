import numpy as np
import torch
from matplotlib import pyplot as plt


def vizualizar_spectrogram(spectrogram: torch.Tensor, name_spectrogram: str ="Spectrogram") -> None:
    # Caso não funcione rode esse codigo "%matplotlib inline" no notebook
    plt.figure(figsize=(10, 12))
    res_mel = spectrogram.detach().cpu().numpy()
    plt.imshow(res_mel, origin='lower')
    plt.xlabel('time')
    plt.ylabel('frequency')
    _ = plt.title(name_spectrogram)


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


import matplotlib.pyplot as plt
import numpy as np

def salvar_comparacao_mel(
    mel_entrada,
    mel_ground_truth,
    mel_modelo,
    save_path="comparacao_mel.png"
):
    """
    Cria uma figura com 3 mel espectrogramas lado a lado.

    Args:
        mel_entrada: numpy array [freq, tempo]
        mel_ground_truth: numpy array [freq, tempo]
        mel_modelo: numpy array [freq, tempo]
        save_path: caminho da imagem final
    """

    """
    Salva 3 mel espectrogramas um embaixo do outro.
    """

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    titulos = [
        "Voz Natural",
        "Voz Sintética",
        "Voz Convertida"
    ]

    mels = [
        mel_ground_truth,
        mel_entrada,
        mel_modelo
    ]

    imagens = []

    for ax, mel, titulo in zip(axes, mels, titulos):

        img = ax.imshow(
            mel,
            aspect='auto',
            origin='lower',
            interpolation='none'
        )

        imagens.append(img)

        ax.set_title(titulo, fontsize=18)
        ax.set_xlabel("Time",fontsize=14)
        ax.set_ylabel("Frequencia",fontsize=14)

    # Barra de cor única
    # fig.colorbar(
    #     imagens[-1],
    #     ax=axes,
    #     orientation='vertical',
    #     fraction=0.02,
    #     pad=0.02
    # )

    plt.tight_layout()

    # Salva com alta qualidade
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Imagem salva em: {save_path}")

def spectograma_3(eletronico: np.array, gerado: np.array, original: np.array, lista_titulos: list[str]):
    fig, axes = plt.subplots(1, 3, figsize=(16, 10))

    axes[0].imshow(eletronico, cmap='viridis', origin='lower')
    axes[0].set_title(lista_titulos[0])

    axes[1].imshow(gerado, cmap='viridis', origin='lower')
    axes[1].set_title(lista_titulos[1])

    axes[2].imshow(original, cmap='viridis', origin='lower')
    axes[2].set_title(lista_titulos[2])

    plt.tight_layout()
    plt.show()
