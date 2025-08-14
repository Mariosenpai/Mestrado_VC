import os

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from main.config import path_save_auto, image_size
from main.pre_processamento.spectograma import transformar_audio_np_em_spectograma, redimencionar_spectorgama
from main.utils.salvar import save_model


def pegar_spectograma_natural_e_eletronico_redimencionado(audio_np: np.array) -> tuple[torch.Tensor, torch.Tensor]:

    data_re_n = redimencionar_spectorgama(transformar_audio_np_em_spectograma(audio_np.audio_natural), image_size)
    data_re_e = redimencionar_spectorgama(transformar_audio_np_em_spectograma(audio_np.audio_eletronico), image_size)

    data_re_natural = data_re_n.reshape(1, image_size, image_size)
    data_re_eletronico = data_re_e.reshape(1, image_size, image_size)

    # transformar o np.array em tensor
    data_re_natural = torch.tensor(data_re_natural)
    data_re_eletronico = torch.tensor(data_re_eletronico)

    return data_re_natural, data_re_eletronico


def pre_processa_batch(data_batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_natural = []
    batch_eletronica = []

    for data in data_batch:
        data_natural, data_eletronica = pegar_spectograma_natural_e_eletronico_redimencionado(data)

        batch_natural.append(data_natural)
        batch_eletronica.append(data_eletronica)

    batch_natural = torch.stack(batch_natural).to(device)
    batch_eletronica = torch.stack(batch_eletronica).to(device)

    return batch_natural, batch_eletronica


def train_autoencoder_laringe(generator, loss_gen, optimizer_gen, epochs, train_loader, device, nome_arquivo_modelo,
                              mostrar_histograma=False):

    best_loss = float('inf')  # Inicializa com um valor muito alto
    best_model_path = None  # Caminho do melhor modelo salvo

    for epoch in range(epochs):
        epoch += 1
        history = []
        print(f'Epocas : {epoch} / {epochs} ')
        total_loss = 0
        k = 0
        for i, data_batch in enumerate(tqdm(train_loader)):

            batch_natural, batch_eletronica = pre_processa_batch(data_batch, device)

            gen_natural = generator(batch_eletronica).to(device)

            optimizer_gen.zero_grad()
            loss = loss_gen(gen_natural, batch_natural)

            total_loss += loss.detach()
            k += 1
            loss.backward()

            optimizer_gen.step()

            # print(f'Loss: {loss.item()}')
            history.append(loss.item())
        total_loss = total_loss / k
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_path = os.path.join(path_save_auto, f"{nome_arquivo_modelo}_best.pt")
            save_model(generator, path_save_auto, f"{nome_arquivo_modelo}_best")
            print(f"Melhor modelo salvo com loss {best_loss}")

        if mostrar_histograma:
            plt.plot(history, label="loss")

            plt.show()

        print("Loss : " + str(total_loss.item()))
    print(f"Melhor modelo salvo em: {best_model_path} com loss {best_loss}")
    return generator