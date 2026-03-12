import os

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from main.config import porcentagem_teste


class audio_LE():
    def __init__(self, audio_natural, audio_eletronico):
        self.audio_natural = audio_natural
        self.audio_eletronico = audio_eletronico



class LE_Dataset(torch.utils.data.Dataset):

    def __init__(self, path_natural, path_eletronica, ext, validacao=False):

        self.path_natural = path_natural
        self.path_eletronica = path_eletronica
        self.ext = ext

        all_items_natural = self.pegar_todos_itens_por_extensao(self.path_natural)
        all_items_eletronica = self.pegar_todos_itens_por_extensao(self.path_eletronica)

        # Divide os dados em 90% treino e 10% validação
        train_files_natural, val_files_natural = train_test_split(all_items_natural, test_size=porcentagem_teste,
                                                                  random_state=42)
        train_files_eletronica, val_files_eletronica = train_test_split(all_items_eletronica,
                                                                        test_size=porcentagem_teste, random_state=42)

        # Seleciona o subconjunto com base no parâmetro `validacao`
        self.item_natural = val_files_natural if validacao else train_files_natural
        self.item_eletronica = val_files_eletronica if validacao else train_files_eletronica

        self.len = len(self.item_natural)  # Atualiza o tamanho com todos os arquivos encontrados

    def pegar_todos_itens_por_extensao(self, path):
        # Recorre por todas as subpastas e coleta os arquivos com a extensão fornecida
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(path)
            for file in files if file.endswith(self.ext)
        ]

    def __len__(self):
        return self.len

    def pre_processar_audio(self, audio):
        tamanho_tensor = 210944

        if len(audio) < tamanho_tensor:
            audio_processado = np.pad(audio, (0, tamanho_tensor - len(audio)), mode='constant')
        else:
            audio_processado = audio[:tamanho_tensor]

        return audio_processado

    def __getitem__(self, idx) -> audio_LE:

        try:

            caminho_audio_natural = self.item_natural[idx]
            caminho_audio_eletronica = self.item_eletronica[idx]

            # carregar audio
            audio_natural, _ = librosa.load(caminho_audio_natural, sr=None)
            audio_eletronica, _ = librosa.load(caminho_audio_eletronica, sr=None)

            # pre-processamento do audio
            audio_natural = self.pre_processar_audio(audio_natural)
            audio_eletronica = self.pre_processar_audio(audio_eletronica)

            return audio_LE(audio_natural, audio_eletronica)

        # Arquivo corrompido  ou nao abrindo
        except Exception as e:

            # Log do erro para análise
            print(
                f"---------------\nErro ao carregar o arquivo: {self.item_natural[idx]} ou {self.path_eletronica[idx]}\nErro: {e}\n---------------")
            # Retorna um áudio vazio ou pula para o próximo item
            # print(f"Error = {e}")
            return self.__getitem__((idx + 1) % self.len)


def collate_fn(batch):
    # Cada item em `batch` é uma tupla (audio_natural, audio_eletronico)
    # Retorne como uma lista de tuplas
    return [(audio_natural, audio_eletronico) for audio_natural, audio_eletronico in batch]

def collate_fn_batch(batch):
    return batch