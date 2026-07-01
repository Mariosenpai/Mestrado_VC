import os
from typing import List

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.common.pre_processamento.noise import f0_constante

from src.common.pre_processamento.spectograma import gerar_mel_spectogram
from src.main.model.DuractionInputEncoder import DPInputEncoder


class Csv_person:

    def __init__(self, client_id, path, sentence):
        self.client_id = client_id
        self.path = path
        self.sentence = sentence


def get_person(csv):
    lista = []
    csv_read = pd.read_csv(csv, sep="\t")
    for _, i in csv_read.iterrows():
        lista.append(Csv_person(i['client_id'], i['path'], i['sentence']))

    return lista


def get_info_csv(csv_list: List[Csv_person]):
    path_list = []
    client_id_list = []
    sentence_list = []
    for i in csv_list:
        path_list.append(
            rf"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\{i.path}")
        client_id_list.append(i.client_id)
        sentence_list.append(i.sentence)

    return path_list, client_id_list, sentence_list


def carregar_duracoes(duracoes_tsv):
    """Lê o arquivo .tsv e cria um dicionário {arquivo: duração}."""
    df = pd.read_csv(duracoes_tsv, sep="\t")
    duracoes = {row["clip"]: row["duration[ms]"] / 1000 for _, row in df.iterrows()}  # Converte ms para segundos
    return duracoes


def calcular_fatias(items_in_dir, duracoes, segundos):
    """Mapeia cada arquivo para os índices das suas fatias sem carregar os áudios."""
    index_map = []
    print("******* Calculando índices das fatias *******")

    for file in tqdm(items_in_dir):
        nome_arquivo = os.path.basename(file)

        # Pega a duração do arquivo pelo nome no dicionário
        duracao = duracoes.get(nome_arquivo, 0)

        if duracao > 0:
            num_fatias = int(duracao // segundos)  # Número de fatias de 10s
            for i in range(num_fatias):
                # Se o audio for vazio não adiciona o mesmo
                index_map.append((file, i))  # (caminho do arquivo, índice da fatia)

    return index_map


def carregar_audio(path_audio):
    """Carrega o áudio completo apenas quando necessário."""
    audio, sr = librosa.load(path_audio, sr=None)
    return audio, sr


def audio_info(idx, index_map, segundos):
    file_path = index_map[idx]  # Obtém o arquivo e o índice da fatia
    audio, sr = carregar_audio(file_path)
    segment_size = segundos * sr  # Tamanho da fatia em amostras

    start = 0 * segment_size
    end = start + segment_size

    segment = audio[start:end]

    return audio, segment, sr, segment_size, file_path


def corta_audio(waveform, sample_rate, fixed_duration_sec: int = 3):
    fixed_length_samples = int(sample_rate * fixed_duration_sec)

    # garante torch.Tensor
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)

    waveform = waveform.float()

    current_length = waveform.shape[0]

    if current_length > fixed_length_samples:
        waveform_fixed = waveform[:fixed_length_samples]

    elif current_length < fixed_length_samples:
        pad_amount = fixed_length_samples - current_length
        waveform_fixed = torch.nn.functional.pad(waveform, (0, pad_amount))

    else:
        waveform_fixed = waveform

    return waveform_fixed


def gerar_mel_spectograma(
        audio,
        sr: int = 22050,
        sr_alvo: int = 22050,
):
    """
    Gera mel-spectrograma SEM cortar o áudio.
    Retorna mel em formato (T, 80).
    """
    mel = gerar_mel_spectogram(audio, rate_alvo=sr_alvo, rate_original=sr)

    # # garante numpy
    # if isinstance(audio, torch.Tensor):
    #     audio = audio.cpu().numpy()
    #
    # # reamostragem obrigatória (NÃO é corte)
    # if sr != sr_alvo:
    #     audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_alvo)
    #     sr = sr_alvo
    #
    # # gera mel com librosa (padrão VTN / PWG)
    # mel = librosa.feature.melspectrogram(
    #     y=audio,
    #     sr=sr,
    #     n_fft=1024,
    #     hop_length=256,
    #     win_length=None,
    #     n_mels=80,
    #     fmin=80,
    #     fmax=sr // 2, # 7600
    #     power=2.0,
    # )
    #
    # # converte para log-mel
    # mel = librosa.power_to_db(mel, ref=np.max)
    #
    # # mel vem (80, T) → queremos (T, 80)
    # mel = mel.T
    #
    # # sanity check
    # assert mel.ndim == 2
    # assert mel.shape[1] == 80

    return mel


def create_dataset_CVMPT_offline(
        path,
        data_type,
        sr_alvo: int = 22050,
):
    file_mel_save_path = fr"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\data\{data_type}"

    CSV = get_person(path)
    path_list, client_id_list, sentence_list = get_info_csv(CSV)

    dp_model = DPInputEncoder()

    os.makedirs(file_mel_save_path, exist_ok=True)

    for i, wav_path in tqdm(
            enumerate(path_list),
            desc="Criando a base",
            total=len(path_list),
    ):
        # nome do arquivo
        file_name = os.path.basename(wav_path)
        file_name = os.path.splitext(file_name)[0]
        save_path = os.path.join(file_mel_save_path, file_name + ".npy")

        # duraction_input = inference_for_mel(sentence_list[i])[0].transpose(1, 0).detach().cpu().numpy()

        if os.path.isfile(save_path):
            continue

        # === carregar áudio inteiro com SR fixo ===
        audio, sr = librosa.load(wav_path, sr=sr_alvo)

        # === mel real ===
        mel = gerar_mel_spectograma(torch.Tensor(audio), sr, sr_alvo)

        mel_for_dp = mel.detach().transpose(0, 1)
        duraction_input = dp_model(mel_for_dp)

        # === áudio artificial (opcional, NÃO usar como referência visual) ===
        wave_noise = f0_constante(audio.astype(np.float64), sr_alvo)
        mel_noise = gerar_mel_spectograma(torch.Tensor(wave_noise), sr_alvo, sr_alvo)

        # Deixa em np.array
        mel = mel.detach().numpy()
        mel_noise = mel_noise.detach().numpy()
        duraction_input = duraction_input.detach().numpy()

        data = {
            "audio": audio.astype(np.float32),
            "audio_noise": wave_noise.astype(np.float32),
            "mel": mel.astype(np.float32),
            "mel_noise": mel_noise.astype(np.float32),
            "duraction_input": duraction_input.astype(np.float32),
            "sample_rate": sr_alvo,
            "client_id": client_id_list[i],
            "sentence": sentence_list[i],
            "id": file_name,
        }

        np.save(save_path, data)


if __name__ == '__main__':
    print("********************** Treinamento **********************")
    create_dataset_CVMPT_offline(
        r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\train.tsv",
        data_type="treinamento")
    print("************************* Teste **************************")
    create_dataset_CVMPT_offline(
        r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\test.tsv",
        data_type="teste")

    data = np.load(
        r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\data\Treinamento_gp\common_voice_pt_20459935.npy",
        allow_pickle=True).item()
