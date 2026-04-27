import os

import numpy as np
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
def calculate_similarity(imageA_path, imageB_path):
    """
    Calcula e exibe as métricas SSIM e PSNR entre duas imagens.

    Parâmetros:
        imageA_path (str): Caminho para a primeira imagem (ex.: espectrograma original).
        imageB_path (str): Caminho para a segunda imagem (ex.: espectrograma processado).

    Retorna:
        ssim_score (float): Valor do índice de similaridade estrutural.
        psnr_value (float): Valor da relação sinal-ruído de pico (dB).
    """
    # Carrega as imagens em escala de cinza
    imgA = imageA_path[0][0].cpu().numpy()  # cv2.imread(imageA_path, cv2.IMREAD_GRAYSCALE)
    imgB = imageB_path[0][0].cpu().numpy()  # cv2.imread(imageB_path, cv2.IMREAD_GRAYSCALE)

    # Verifica se as imagens foram carregadas corretamente
    if imgA is None or imgB is None:
        raise FileNotFoundError("Uma das imagens não foi encontrada ou não pode ser aberta.")

    # Verifica se as imagens possuem o mesmo tamanho
    if imgA.shape != imgB.shape:
        raise ValueError("As imagens devem ter as mesmas dimensões para comparação.")

    # Se necessário, converte para uint8 ou especifica data_range
    # Exemplo: se as imagens estiverem em ponto flutuante e na faixa de 0 a 255, data_range=255
    # Caso contrário, pode calcular automaticamente a diferença entre os valores máximo e mínimo
    data_range = imgB.max() - imgB.min()

    # Calcula o SSIM com o parâmetro data_range especificado
    ssim_score, diff = ssim(imgA, imgB, full=True, data_range=data_range)
    # print(f"SSIM: {ssim_score:.4f}")

    # Calcula o PSNR
    mse = np.mean((imgA - imgB) ** 2)
    if mse == 0:
        psnr_value = float('inf')
    else:
        PIXEL_MAX = 255.0
        psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    # print(f"PSNR: {psnr_value:.4f} dB")

    return ssim_score, psnr_value

def distancia_euclidiana(amplitude_original: np.array, amplitude_processado: np.array) -> float:
    return np.linalg.norm(amplitude_original - amplitude_processado)

def media_desvio_padrao(lista_valores: list) -> tuple:
    """

    :param lista_valores: lista de valores [float, float, float, ...]
    :return:
    """

    # Calcula média e desvio padrão das métricas
    mean = np.mean(lista_valores) if lista_valores else None
    std = np.std(lista_valores) if lista_valores else None

    return mean, std


def compute_amplitude_spectrum_array(data, sample_rate):
    """
    Calcula o espectro de amplitude a partir de um array NumPy de áudio.

    Parâmetros:
      data: np.array
          Array com os dados do áudio.
      sample_rate: int ou float
          Taxa de amostragem do áudio.

    Retorna:
      frequencies: np.array
          Array com as frequências correspondentes.
      amplitude_spectrum: np.array
          Array com a amplitude do espectro.
    """
    # Se o áudio for estéreo, utiliza apenas um dos canais
    if len(data.shape) > 1:
        data = data[:, 0]

    # Cálculo da FFT e do espectro de amplitude
    fft_data = np.fft.rfft(data)
    amplitude_spectrum = np.abs(fft_data)
    frequencies = np.fft.rfftfreq(len(data), d=1 / sample_rate)

    return frequencies, amplitude_spectrum


def plot_amplitude_spectrum_from_arrays(original_data, processed_data, sample_rate, titulo="Espectro de Amplitude"):
    """
    Plota o espectro de amplitude dos áudios original e processado a partir de arrays NumPy,
    exibindo o gráfico sem salvá-lo em disco.

    Parâmetros:
      original_data: np.array
          Array com o áudio original.
      processed_data: np.array
          Array com o áudio processado.
      sample_rate: int ou float
          Taxa de amostragem dos áudios.
      titulo: str
          Título do gráfico.
    """
    # Calcula os espectros de amplitude
    original_freqs, original_spectrum = compute_amplitude_spectrum_array(original_data, sample_rate)
    processed_freqs, processed_spectrum = compute_amplitude_spectrum_array(processed_data, sample_rate)

    # epsilon = 1e-10
    # original_spectrum_db = 20 * np.log10(original_spectrum + epsilon)
    # processed_spectrum_db = 20 * np.log10(processed_spectrum + epsilon)

    # Cria a figura do gráfico
    plt.figure(figsize=(15, 5))

    # Plota o espectro do áudio original com linha tracejada
    plt.plot(original_freqs[:500], original_spectrum[:500], '--', label='Original',
             color='b', linewidth=1, alpha=0.5)

    # Plota o espectro do áudio processado com marcadores e linha dash
    plt.plot(processed_freqs[:500], processed_spectrum[:500], '-o', label='Processado',
             color='r', linestyle='dashed', linewidth=0.6, alpha=0.5, markersize=2.5)

    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Amplitude')
    plt.title(titulo)
    plt.legend()
    plt.grid()

    # Exibe o gráfico na tela (sem salvá-lo)
    plt.show()


def compute_amplitude_spectrum(file_path):
    """Lê um arquivo de áudio e retorna o espectro de amplitude."""
    sample_rate, data = wav.read(file_path)

    # Se for estéreo, pega apenas um canal
    if len(data.shape) > 1:
        data = data[:, 0]

    # FFT e cálculo do espectro de amplitude
    fft_data = np.fft.rfft(data)
    amplitude_spectrum = np.abs(fft_data)
    frequencies = np.fft.rfftfreq(len(data), d=1 / sample_rate)

    return frequencies, amplitude_spectrum


def plot_amplitude_spectrum_para_pastas(original_folder, processed_folder, output_folder):
    """Plota o espectro de amplitude dos áudios originais e processados."""
    audio_files = [f for f in os.listdir(original_folder) if f.endswith('.wav')]
    print(audio_files)
    for file in audio_files:
        original_path = os.path.join(original_folder, file)
        processed_path = os.path.join(processed_folder, file)

        if not os.path.exists(processed_path):
            print(f"Arquivo {file} não encontrado na pasta processada. Pulando...")
            continue

        # Calcula espectros
        original_freqs, original_spectrum = compute_amplitude_spectrum(original_path)
        processed_freqs, processed_spectrum = compute_amplitude_spectrum(processed_path)

        # Plotando o espectro de amplitude
        plt.figure(figsize=(15, 5))
        # Linha original com menor espessura e transparência
        plt.plot(original_freqs[:500], original_spectrum[:500], '--', label='Original', color='b', linewidth=1,
                 alpha=0.5)

        # Linha processada com marcadores e menor espessura
        plt.plot(processed_freqs[:500], processed_spectrum[:500], '-o', label='Processado', color='r',
                 linestyle='dashed', linewidth=0.6, alpha=0.5, markersize=2.5)
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'Espectro de Amplitude - {file}')
        plt.legend()
        plt.grid()

        # Salvar o gráfico
        save_path = os.path.join(output_folder, f'spectrum_{file}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

