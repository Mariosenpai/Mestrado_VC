import os

import numpy as np
import torch
from tqdm import tqdm

from main.config import image_size, path_save_auto
from main.pre_processamento.audio import add_ruido_audio
from main.pre_processamento.spectograma import redimencionar_spectorgama, transformar_audio_np_em_spectograma
from main.utils.salvar import save_model


def criar_audio_com_ruido(audio_np: np.array, sr: int) -> np.array:
    """

    :param audio_np: [x,x]
    :param sr:
    :return:
    """
    return add_ruido_audio(audio_np, sr, intencidade=0.1, ruido="robotic")


def pegar_audio_para_ruido(audio_np: np.array, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param audio_np:
    :param sr:
    :return:
    """

    audio_com_ruido = criar_audio_com_ruido(audio_np, sr)

    audio_sem_ruido = audio_np
    audio_com_ruido = audio_com_ruido

    # Entrada da função redimencionar_spectorgama() é np.array[X,X]
    audio_com_ruido = redimencionar_spectorgama(transformar_audio_np_em_spectograma(audio_com_ruido), image_size)
    audio_sem_ruido = redimencionar_spectorgama(transformar_audio_np_em_spectograma(audio_sem_ruido), image_size)

    # Adicona uma dimensão para se adquedar a entrada do torch.stack() ficando tensor[1,x,x]
    audio_com_ruido = torch.tensor(audio_com_ruido).unsqueeze(0)
    audio_sem_ruido = torch.tensor(audio_sem_ruido).unsqueeze(0)

    return audio_sem_ruido, audio_com_ruido


def pre_processa_batch(data_batch: tuple[np.array, int], device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param data_batch: Audio e o sr do audio
    :param device:
    :return:
    """
    batch_sem_ruido = []
    batch_com_ruido = []

    for audio_np, sr in data_batch:
        audio_sem_ruido, audio_com_ruido = pegar_audio_para_ruido(audio_np, sr)

        batch_sem_ruido.append(audio_sem_ruido)
        batch_com_ruido.append(audio_com_ruido)

    batch_sem_ruido = torch.stack(batch_sem_ruido).to(device)
    batch_com_ruido = torch.stack(batch_com_ruido).to(device)

    return batch_sem_ruido, batch_com_ruido


def train_autoencoder_mozilla_pt(generator, loss_gen, optimizer_gen, epochs, train_loader, device, nome_arquivo_modelo):
    best_loss = float('inf')  # Inicializa com um valor muito alto
    best_model_path = None  # Caminho do melhor modelo salvo

    for epoch in range(epochs):
        epoch += 1
        history = []
        print(f'Epocas : {epoch} / {epochs} ')
        total_loss = 0
        k = 0
        for i, data_batch in enumerate(tqdm(train_loader)):
            batch_sem_ruido, batch_com_ruido = pre_processa_batch(data_batch, device)

            gen_natural = generator(batch_com_ruido).to(device)

            optimizer_gen.zero_grad()
            loss = loss_gen(gen_natural, batch_sem_ruido)

            total_loss += loss.detach()
            k += 1
            loss.backward()

            if total_loss.item() == 0 or total_loss.item() == "NaN" or total_loss.item() == "inf":
                print(i)
                break

            optimizer_gen.step()

            # print(f'Loss: {loss.item()}')
            history.append(loss.item())
        total_loss = total_loss / k

        # Se for a menor perda já registrada, salva o modelo
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_path = os.path.join(path_save_auto, f"{nome_arquivo_modelo}_best.pt")
            save_model(generator, path_save_auto, f"{nome_arquivo_modelo}_best")
            print(f"Melhor modelo salvo com loss {best_loss}")

        # plt.plot(history, label="loss")

        # plt.show()

        print("Loss : " + str(total_loss.item()))

    print(f"Melhor modelo salvo em: {best_model_path} com loss {best_loss}")
    return generator
