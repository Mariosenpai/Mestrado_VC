from pathlib import Path
from src.main.collaters.nar_vc import NARVCCollater
from src.main.service.BaseService import BaseService

import json
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/home/mario/cuda_libdevice"

import tensorflow as tf

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from src.common.metricas import Mosnet


def _converter_json(obj: Any) -> Any:
    """
    Converte tipos nao nativos (numpy, etc) para tipos serializaveis em JSON.
    Usado como parametro 'default' do json.dump.

    Entrada:
        obj (Any): objeto que o json nao conseguiu serializar diretamente.

    Saida:
        Any: versao do objeto em tipo nativo do Python (float, int, list).
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Objeto do tipo {type(obj)} nao e serializavel em JSON")


def carregar_checkpoint(caminho_arquivo: str) -> dict[str, Any]:
    """
    Carrega o checkpoint salvo em disco, se existir.

    Entrada:
        caminho_arquivo (str): caminho do arquivo JSON de checkpoint.

    Saida:
        dict[str, Any]: dicionario com as chaves 'indice', 'list_pontuacao_noise'
        e 'list_pontuacao_org'. Se o arquivo nao existir, retorna um checkpoint
        vazio (indice 0 e listas vazias).
    """
    if os.path.exists(caminho_arquivo):
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "indice": 0,
        "list_pontuacao_noise": [],
        "list_pontuacao_org": [],
    }


def salvar_checkpoint(
        caminho_arquivo: str,
        indice: int,
        list_pontuacao_noise: list,
        list_pontuacao_org: list,
) -> None:
    """
    Salva o progresso atual em um arquivo JSON, de forma atomica
    (escreve em arquivo temporario e depois renomeia), para evitar
    corromper o checkpoint caso o processo seja interrompido durante a escrita.

    Entrada:
        caminho_arquivo (str): caminho do arquivo JSON de checkpoint.
        indice (int): indice do proximo batch a ser processado.
        list_pontuacao_noise (list): pontuacoes de audio com ruido ja calculadas.
        list_pontuacao_org (list): pontuacoes de audio original ja calculadas.

    Saida:
        None
    """
    dados = {
        "indice": indice,
        "list_pontuacao_noise": list_pontuacao_noise,
        "list_pontuacao_org": list_pontuacao_org,
    }

    caminho_tmp = caminho_arquivo + ".tmp"
    with open(caminho_tmp, "w", encoding="utf-8") as f:
        json.dump(dados, f, default=_converter_json)

    os.replace(caminho_tmp, caminho_arquivo)


def pontuacao_dataloader(
        dataloader: Any,
        name: str,
        caminho_checkpoint: str | None = None,
        intervalo_save: int = 50,
) -> tuple[list, list]:
    """
    Calcula a pontuacao mosnet de cada audio (original e com ruido) do
    dataloader, salvando o progresso periodicamente em um arquivo. Caso
    o processo seja interrompido, uma nova chamada retoma do ponto salvo
    em vez de recomecar do zero.

    Entrada:
        dataloader (Any): iteravel de batches, cada batch deve conter
            a chave "audio" contendo (audio, audio_noise).
        name (str): nome usado na barra de progresso (tqdm).
        caminho_checkpoint (str | None): caminho do arquivo de checkpoint.
            Se None, um nome padrao baseado em 'name' e usado.
        intervalo_save (int): a cada quantos batches processados o
            checkpoint e salvo em disco.

    Saida:
        tuple[list, list]: (list_pontuacao_noise, list_pontuacao_org) com
        a pontuacao de todos os audios processados.
    """
    if caminho_checkpoint is None:
        caminho_checkpoint = f"checkpoint_{name}.json"

    checkpoint = carregar_checkpoint(caminho_checkpoint)
    indice_inicial = checkpoint["indice"]
    list_pontuacao_noise = checkpoint["list_pontuacao_noise"]
    list_pontuacao_org = checkpoint["list_pontuacao_org"]

    total = len(dataloader)

    mosnet = Mosnet()

    for i, batch in enumerate(
            tqdm(dataloader, desc=f"Pontuacao total {name}", total=total, initial=indice_inicial)
    ):
        if i < indice_inicial:
            continue

        audio, audio_noise = batch["audio"][0], batch["audio_noise"][0]

        pontuacao_audio_org = mosnet.inference(audio)
        pontuacao_audio_noise = mosnet.inference(audio_noise)

        list_pontuacao_noise.append(pontuacao_audio_noise)
        list_pontuacao_org.append(pontuacao_audio_org)

        if (i + 1) % intervalo_save == 0:
            salvar_checkpoint(
                caminho_checkpoint, i + 1, list_pontuacao_noise, list_pontuacao_org
            )

    salvar_checkpoint(caminho_checkpoint, total, list_pontuacao_noise, list_pontuacao_org)

    return list_pontuacao_noise, list_pontuacao_org


def media_desvio_padrao(valores: list) -> tuple[float, float]:
    """
    Calcula a media e o desvio padrao de uma lista de valores numericos.

    Entrada:
        valores (list): lista de numeros (ex: pontuacoes mosnet).

    Saida:
        tuple[float, float]: (media, desvio_padrao) dos valores.
    """
    array_valores = np.array(valores, dtype=np.float64)

    media = float(np.mean(array_valores))
    desvio_padrao = float(np.std(array_valores))

    return media, desvio_padrao


def resumo_estatistico(
        points_noise_train: list,
        points_org_train: list,
        points_noise_val: list,
        points_org_val: list,
) -> dict[str, tuple[float, float]]:
    """
    Calcula media e desvio padrao para os quatro conjuntos de pontuacoes
    (audio com ruido e original, nos conjuntos de treino e validacao).

    Entrada:
        points_noise_train (list): pontuacoes de audio com ruido (treino).
        points_org_train (list): pontuacoes de audio original (treino).
        points_noise_val (list): pontuacoes de audio com ruido (validacao).
        points_org_val (list): pontuacoes de audio original (validacao).

    Saida:
        dict[str, tuple[float, float]]: dicionario onde cada chave e o nome
        do conjunto e o valor e uma tupla (media, desvio_padrao).
    """
    resumo = {
        "noise_train": media_desvio_padrao(points_noise_train),
        "org_train": media_desvio_padrao(points_org_train),
        "noise_val": media_desvio_padrao(points_noise_val),
        "org_val": media_desvio_padrao(points_org_val),
    }

    for nome, (media, desvio_padrao) in resumo.items():
        print(f"{nome}: media = {media:.4f}, desvio padrao = {desvio_padrao:.4f}")

    return resumo


if __name__ == "__main__":
    ROOT = Path.cwd().resolve()
    path_dataset = ROOT / "dataset" / "cv-corpus-mozilla-pt" / "data"

    print(path_dataset)
    train_load, val_load = BaseService(batch_size=1, path_dataset=path_dataset,
                                       collete_fn=NARVCCollater())._define_dataloader(path_dataset=path_dataset)

    points_noise_train, points_org_train = pontuacao_dataloader(train_load, "train")
    points_noise_val, points_org_val = pontuacao_dataloader(train_load, "val")

    resumo = resumo_estatistico(
        points_noise_train, points_org_train,
        points_noise_val, points_org_val,
    )
