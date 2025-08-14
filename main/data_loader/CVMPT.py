import os

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from main.config import porcentagem_teste, arquivos_ignorados_CVMPT
from main.pre_processamento.noise import f0_constante
from main.pre_processamento.spectograma import gerar_mel_spectogram

import audio_effects as ae


class CV_mozilla_PT_Dataset(torch.utils.data.Dataset):

    def __init__(self, path: str, ext: str, path_duracoes_tsv_csv: str, validacao=False, sample_rate_final=22050,
                 duracao_min_segundos=3, return_mel_spec=False):
        self.path = path
        self.ext = ext
        self.segundos = duracao_min_segundos  # Duração das fatias em segundos
        self.sample_rate = sample_rate_final
        self.retorne_mel_spec = return_mel_spec

        # Carrega a tabela com durações dos áudios
        self.duracoes = self.carregar_duracoes(path_duracoes_tsv_csv)

        # Recorre por todas as subpastas e coleta os arquivos com a extensão fornecida
        all_items = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.path)
            for file in files if file.endswith(self.ext)
        ]

        print(len(all_items))

        for arq in arquivos_ignorados_CVMPT:
            all_items.remove(arq)

        # Divide os dados em treino e validação
        train_files, val_files = train_test_split(all_items, test_size=porcentagem_teste, random_state=42)

        # Seleciona o subconjunto com base no parâmetro `validacao`
        self.items_in_dir = val_files if validacao else train_files

        # Cria um mapeamento entre segmentos e índices sem carregar áudios
        self.index_map = self.calcular_fatias()

    def carregar_duracoes(self, duracoes_tsv):
        """Lê o arquivo .tsv e cria um dicionário {arquivo: duração}."""
        df = pd.read_csv(duracoes_tsv, sep="\t")
        duracoes = {row["clip"]: row["duration[ms]"] / 1000 for _, row in df.iterrows()}  # Converte ms para segundos
        return duracoes

    def corta_audio(self, waveform, sample_rate, fixed_duration_sec: int = 3):

        fixed_length_samples = sample_rate * fixed_duration_sec
        current_length = waveform.shape[0]

        if current_length > fixed_length_samples:
            # corta o excesso do áudio
            waveform_fixed = waveform[:, :fixed_length_samples]
        elif current_length < fixed_length_samples:
            # adiciona zeros no final do áudio para completar o tamanho desejado
            pad_amount = fixed_length_samples - current_length
            waveform_fixed = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform_fixed = waveform  # já está no tamanho correto

        return waveform_fixed

    def calcular_fatias(self):
        """Mapeia cada arquivo para os índices das suas fatias sem carregar os áudios."""
        index_map = []
        print("******* Calculando índices das fatias *******")

        for file in tqdm(self.items_in_dir):
            nome_arquivo = os.path.basename(file)

            # Pega a duração do arquivo pelo nome no dicionário
            duracao = self.duracoes.get(nome_arquivo, 0)

            if duracao >= self.segundos:
                num_fatias = int(duracao // self.segundos)  # Número de fatias de 10s
                for i in range(num_fatias):
                    # Se o audio for vazio não adiciona o mesmo
                    index_map.append((file, i))  # (caminho do arquivo, índice da fatia)

        return index_map

    def __len__(self):
        return len(self.index_map)

    def carregar_audio(self, path_audio):
        """Carrega o áudio completo apenas quando necessário."""
        audio, sr = librosa.load(path_audio, sr=None)
        return audio, sr

    def auido_vazio(self, file):
        '''
        Varifica se o audio esta vazio (sem conteudo)
        :param file:
        :return:
        '''
        audio, _ = self.carregar_audio(file)
        if set(audio) == {0}:
            return True
        else:
            return False

    def __getitem__(self, idx, mostrar_path=False):
        try:
            file_path, fatia_idx = self.index_map[idx]  # Obtém o arquivo e o índice da fatia
            if mostrar_path:
                print(file_path, fatia_idx)
            audio, sr = self.carregar_audio(file_path)
            segment_size = self.segundos * sr  # Tamanho da fatia em amostras

            start = fatia_idx * segment_size
            end = start + segment_size

            segment = audio[start:end]

            if len(segment) == segment_size:

                if self.retorne_mel_spec:
                    audio_cortado = self.corta_audio(segment, sr)
                    audio = torch.tensor(np.expand_dims(audio_cortado, axis=0), dtype=torch.float32)
                    spectrogram = gerar_mel_spectogram(audio, rate_original=sr, rate_alvo=self.sample_rate)

                    return spectrogram
                else:
                    return segment, sr

            else:
                return self.__getitem__((idx + 1) % self.__len__())  # Pula áudios menores que 10s

        except Exception as e:
            print(f"Erro ao carregar o arquivo: {file_path}\nErro: {e}")
            return self.__getitem__((idx + 1) % self.__len__())  # Pula para o próximo item


class cv_mozilla_pt_dataset_com_ruido(CV_mozilla_PT_Dataset):

    def __init__(self, path: str, ext: str, path_duracoes_tsv_csv: str, validacao=False, sample_rate_final=22050,
                 duracao_min_segundos=3):
        # Chama o inicializador da classe pai
        super().__init__(path, ext, path_duracoes_tsv_csv, validacao, sample_rate_final, duracao_min_segundos)

    def _add_ruido(self, y: np.array, sr: int):
        return f0_constante(y.astype(np.float64), sr)

    def __getitem__(self, idx, mostrar_path=False):
        try:
            file_path, fatia_idx = self.index_map[idx]  # Obtém o arquivo e o índice da fatia
            if mostrar_path:
                print(file_path, fatia_idx)
            audio, sr = self.carregar_audio(file_path)
            segment_size = self.segundos * sr  # Tamanho da fatia em amostras

            start = fatia_idx * segment_size
            end = start + segment_size

            segment = audio[start:end]

            if len(segment) == segment_size:

                audio_cortado = self.corta_audio(segment, sr)
                audio_cortado = np.expand_dims(audio_cortado, axis=0)

                audio = torch.tensor(audio_cortado, dtype=torch.float32)
                audio_ruido = torch.tensor(self._add_ruido(audio_cortado, sr), dtype=torch.float32)

                spectrogram = gerar_mel_spectogram(audio, rate_original=sr, rate_alvo=self.sample_rate)
                spectrogram_ruido = gerar_mel_spectogram(audio_ruido, rate_original=sr, rate_alvo=self.sample_rate)

                return spectrogram, spectrogram_ruido


            else:
                return self.__getitem__((idx + 1) % self.__len__())  # Pula áudios menores que 10s

        except Exception as e:
            print(f"Erro ao carregar o arquivo: {file_path}\nErro: {e}")
            return self.__getitem__((idx + 1) % self.__len__())  # Pula para o próximo item
