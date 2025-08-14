import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import os

# Caminho dos áudios
original_folder = r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\out\o'
processed_folder = r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\out\g'
output_folder = r'graficos/'
os.makedirs(output_folder, exist_ok=True)

def compute_amplitude_spectrum(file_path):
    """Lê um arquivo de áudio e retorna o espectro de amplitude."""
    sample_rate, data = wav.read(file_path)

    # Se for estéreo, pega apenas um canal
    if len(data.shape) > 1:
        data = data[:, 0]

    # FFT e cálculo do espectro de amplitude
    fft_data = np.fft.rfft(data)
    amplitude_spectrum = np.abs(fft_data)
    frequencies = np.fft.rfftfreq(len(data), d=1/sample_rate)

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
        plt.plot(original_freqs[:500], original_spectrum[:500], '--', label='Original', color='b', linewidth=1, alpha=0.5)

        # Linha processada com marcadores e menor espessura
        plt.plot(processed_freqs[:500], processed_spectrum[:500], '-o', label='Processado', color='r', linestyle='dashed', linewidth=0.6, alpha=0.5, markersize=2.5)
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'Espectro de Amplitude - {file}')
        plt.legend()
        plt.grid()
        
        # Salvar o gráfico
        save_path = os.path.join(output_folder, f'spectrum_{file}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

# Executa a função
# plot_amplitude_spectrum(original_folder, processed_folder, output_folder)
