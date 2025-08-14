from torch import nn
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import numpy as np
from sklearn.cluster import KMeans
import torch


def apply_kmeans_to_extract_tokens(embeddings, n_tokens):
    """
    Aplica K-Means para extrair tokens discretos a partir dos embeddings.

    Args: embeddings (np.array): Matriz de embeddings de forma [B, D], onde B é o número de exemplos e D é a
    dimensionalidade do embedding. N_tokens (int): Número de tokens discretos (clusters) desejados.

    Returns:
        np.array: vetores de tokens discretos de forma [B, 1], onde cada entrada é o índice do cluster correspondente.
    """
    # Treinar o K-Means
    kmeans = KMeans(n_clusters=n_tokens, random_state=42)
    kmeans.fit(embeddings)

    # Atribuir os tokens discretos para cada embedding (cluster mais próximo)
    tokens = kmeans.predict(embeddings)  # tokens será um vetor de indices de clusters

    return tokens


def wav_extrator(audio_wav, shape_saida: int = 128, sampling_rate=16000, embeddings_por_fatia: bool = True,
                 device="cuda") -> torch.Tensor:
    """

    :param embeddings_por_fatia:
    :param device:
    :param sampling_rate:
    :param audio_wav:
    :param shape_saida:
    :return: tensor [batch_size, shape_saida] exemplo : [1,128]
    """

    # dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
    model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

    # audio files are decoded on the fly
    audio = audio_wav
    inputs = feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=sampling_rate)
    embeddings = model(**inputs)

    if embeddings_por_fatia:
        conteudo_linguistico = embeddings.hidden_states
    else:
        conteudo_linguistico = embeddings.embeddings
        # Crie uma camada de projeção que mapeia de 512 para nova_dim
        projecao = nn.Linear(512, shape_saida).to(device)

        # Projete o embedding para a nova dimensão
        conteudo_linguistico = projecao(conteudo_linguistico)  # Esperado: [1, 128]

    return conteudo_linguistico


def content_extrator(audio_wav, shape_saida: int = 128, num_cluster: int = 1024, sampling_rate=16000,
                                  embeddings_por_fatia: bool = True,device="cuda"):

    embeddings = wav_extrator(audio_wav, shape_saida=shape_saida, sampling_rate=sampling_rate,
                                   device=device)
    if embeddings_por_fatia:
        # Concatena corretamente os tensores, remove a dimensão inicial, move para CPU e converte para numpy float64
        token_por_frame_tensor = torch.cat([t.detach().cpu() for t in embeddings], dim=1).squeeze(0).numpy().astype(
            np.float64)
        # Confira o tipo e formato
        # print(token_por_frame_tensor.shape, token_por_frame_tensor.dtype)  # Deve imprimir: (5837, 768) float64

        # Aplicar K-Means corretamente agora com float64
        return apply_kmeans_to_extract_tokens(token_por_frame_tensor, num_cluster)
    else:
        return embeddings
