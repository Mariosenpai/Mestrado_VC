import os

import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from main.config import arquivos_ignorados, porcentagem_teste, tamanho_tensor


class LHB_Dataset(torch.utils.data.Dataset):

    def __init__(self, path, ext, validacao=False):
        self.path = path
        self.ext = ext

        # Recorre por todas as subpastas e coleta os arquivos com a extensão fornecida
        all_items = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.path)
            for file in files if file.endswith(self.ext)
        ]

        # Elimina os arquivos indesejaveis
        for arq in arquivos_ignorados:
            all_items.remove(arq)

        # Divide os dados em 90% treino e 10% validação
        train_files, val_files = train_test_split(all_items, test_size=porcentagem_teste, random_state=42)

        # Seleciona o subconjunto com base no parâmetro `validacao`
        self.items_in_dir = val_files if validacao else train_files

        self.len = len(self.items_in_dir)  # Atualiza o tamanho com todos os arquivos encontrados

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        try:

            name = self.items_in_dir[idx]
            audio, _ = librosa.load(name, sr=None)

            if len(audio) < tamanho_tensor:
                audio = np.pad(audio, (0, tamanho_tensor - len(audio)), mode='constant')
            else:
                audio = audio[:tamanho_tensor]

            return audio
        # Arquivo corrompido  ou nao abrindo
        except Exception as e:

            # Log do erro para análise
            print(f"---------------\nErro ao carregar o arquivo: {self.items_in_dir[idx]}\nErro: {e}\n---------------")
            # Retorna um áudio vazio ou pula para o próximo item
            # print(f"Error = {e}")
            return self.__getitem__((idx + 1) % self.len)

        return audio