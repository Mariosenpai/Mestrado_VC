import librosa
import numpy as np
import pydub
from scipy import signal
from scipy.signal import convolve

from main.pre_processamento.spectograma import redimensionar_audio_para_tamanho_original
from main.utils.gerar import info_audio


def load_audio(file_name):
    audio = pydub.AudioSegment.from_file(file_name)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
    arr = arr / (1 << (8 * audio.sample_width - 1))
    return arr.astype(np.float32), audio.frame_rate

def add_ruido_audio(
        audio: np.ndarray,
        sr: np.array,
        intencidade: float = 0.01,
        ruido: str = "branco",
        nome_arquivo: str = "audio_com_ruido",
) -> np.array:
    '''
    Adiciona ruido em um audio que esteja em np.array
    Lista de ruidos: [branco,eco,metalico,distortion,flanger,robotic,vocoder-senoidal]
    :param nome_arquivo:
    :param audio: shape do audio (x,) exemplo (16000,)
    :param sr: int
    :param intencidade:
    :param ruido:
    :return:
    '''

    if ruido == "branco":
        noise = np.random.normal(0, intencidade, audio.shape)  # Ajuste o volume do ruído alterando 0.02
        # Adicionar ruído ao áudio original
        audio_com_ruido = audio + noise
    elif ruido == "eco":
        delay = int(sr * intencidade)  # Pequeno atraso (~2ms) para gerar ressonância metálica
        audio_com_ruido = audio.copy()
        for i in range(delay, len(audio)):
            audio_com_ruido[i] += 0.6 * audio_com_ruido[i - delay]  # Ajuste o fator para mais/menos eco
    elif ruido == "metalico":
        # Criar onda senoidal de modulação
        freq_mod = 150 * (1 + intencidade)  # Frequência de modulação (ajuste para mais ou menos efeito)
        t = np.arange(len(audio)) / sr
        modulator = np.sin(2 * np.pi * freq_mod * t)
        # Aplicar modulação de anel
        audio_com_ruido = audio * modulator
    elif ruido == "flanger":
        # Criar efeito de flanger
        delay = int(sr * intencidade)  # Atraso de 1ms
        depth = 0.5  # Intensidade do efeito
        flanger_audio = audio.copy()
        for i in range(delay, len(audio)):
            flanger_audio[i] += depth * audio[i - delay]
        # Normalizar áudio
        audio_com_ruido = flanger_audio / np.max(np.abs(flanger_audio))
    elif ruido == "distortion":
        distorted_audio = np.tanh(5 * audio * (1 + intencidade))  # Ajuste o fator para mais distorção
        from scipy.signal import butter, lfilter
        # Criar um filtro passa-alta
        def highpass_filter(data, cutoff=800, fs=44000, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=True)
            return lfilter(b, a, data)

        # Aplicar filtro passa-alta
        audio_com_ruido = highpass_filter(distorted_audio, cutoff=1200, fs=sr)

    elif ruido == "vocoder-senoidal":
        # Criar um sinal senoidal como portadora (tom robótico)
        carrier_freq = 120 * (1 + intencidade)  # Ajuste para alterar o efeito robótico
        t = np.arange(len(audio)) / sr
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        # Aplicar vocoder simples (multiplicação com carrier)
        vocoder_audio = audio * carrier

        # Normalizar
        audio_com_ruido = vocoder_audio / np.max(np.abs(vocoder_audio))

    elif ruido == "robotic":
        kernel = np.zeros(1000)
        kernel[::200] = 1  # Adiciona um atraso artificial

        # Aplicar convolução para criar o efeito robótico
        processed = convolve(audio, kernel, mode="same")

        # Normalizar e converter para o formato original
        audio_com_ruido = np.float32(processed / np.max(np.abs(processed)) * 32767 * (1 + intencidade))
    else:
        lista_ruidos = ["branco", "eco", "metalico", "distortion", "flanger", "robotic", "vocoder-senoidal"]
        print(f"Tipo {ruido} não é reconhecido")
        print(f"Lista de ruidos cadastrados {lista_ruidos}")
        return

    return audio_com_ruido
    # sf.write(f"{caminho}{nome_arquivo}.wav", audio_com_ruido, sr)


def transformar_audio_np_em_wav(audio, audio_original):
    """
    :param audio: (np.array, np.array) exemplo de entrada (500,500)
    :param audio_original: (float,) exemplo de entrada (16000,)
    :return:
    """

    phase, MIN, MAX, spectrogram_original = info_audio(audio_original)

    # Desnormalizar
    generated = audio * (MAX - MIN) + MIN

    # Converter de dB para amplitude
    generated = librosa.db_to_amplitude(generated)

    # Ajustar a fase para ter o mesmo tamanho de generated
    phase = phase[:generated.shape[0], :generated.shape[1]]  # Garante compatibilidade

    # Aplicar fase para reconstrução do espectrograma complexo
    generated = generated * np.exp(1j * phase)

    # Inverter para áudio
    generated = librosa.istft(np.array(generated), win_length=1024, window=signal.windows.hamming(1024), n_fft=1024)

    # Normalizar para evitar distorções
    generated = generated / np.max(np.abs(generated))
    return generated


def transformar_imagem_em_audio(audio_processado: np.array, audio_original: np.array) -> np.array:
    """

    :param audio_processado: (1024,1024)
    :param audio_original: (16000,)
    :return: audio da imagem em espectro
    """
    _, _, _, spectrogram_original = info_audio(audio_original)
    imagem_redimensionada_tam_original = redimensionar_audio_para_tamanho_original(audio_processado,
                                                                                   spectrogram_original.shape)
    generated = transformar_audio_np_em_wav(imagem_redimensionada_tam_original, audio_original)

    return generated