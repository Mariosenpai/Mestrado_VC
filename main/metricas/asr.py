import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import librosa
from main.pre_processamento.noise import f0_constante


# Carregar o áudio
# audio, taxa_amostragem = torchaudio.load("caminho/para/seu/audio.wav")

def _transcricao(audio_np, sr):
    modelo = Wav2Vec2ForCTC.from_pretrained("lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2")
    processador = Wav2Vec2Processor.from_pretrained("lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2")
    audio_c = torch.tensor(audio_np).clone().detach().float()
    audio_trans = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio_c)
    # Processar o áudio
    inputs = processador(audio_trans.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
    logits = modelo(input_values=inputs.input_values).logits

    # Decodificar a transcrição
    transcricao = processador.batch_decode(torch.argmax(logits, dim=-1))

    return transcricao


def _wer(x, y, sr):
    from jiwer import wer
    referencia = _transcricao(x, sr)
    hipotese = _transcricao(y, sr)
    erro = wer(referencia, hipotese)

    return erro


if __name__ == '__main__':
    x, fs = librosa.load(
        r'/dataset/2024_AUDIOS_PROJETO_LARINGE/SEM_TRAQUEOSTOMIA/DALLETE_FONO/NATURAL_MP3/2_n.mp3',
        dtype=np.float64)
    y = f0_constante(x, fs)
    referencia = _transcricao(x, fs)
    hipotese = _transcricao(y, fs)
    print(referencia)
    print(hipotese)
