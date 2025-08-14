import librosa
import numpy as np
import torch
from tqdm import tqdm

from main.metricas.LSD import calculate_lsd_tensor
from main.metricas.SNR import calculate_snr_tensor
from main.config import image_size
from scipy import signal

def validacao_melhoramento_auido(val_loader, generator, loss, num_epoca, device='cpu'):
    # Avaliação
    generator.eval()
    with torch.no_grad():
        total_snr, total_lsd, count = 0, 0, 0
        for i, data_batch in enumerate(tqdm(val_loader)):
            batch_lb, batch_hb = [], []

            for data in data_batch:
                data = data.squeeze(dim=0)

                # Transformar em espectrograma
                val_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096,
                                        window=signal.windows.hamming(4096))
                val_spectrogram = torch.tensor(librosa.amplitude_to_db(np.abs(val_stft)))
                val_spectrogram = (val_spectrogram - val_spectrogram.min()) / (
                            val_spectrogram.max() - val_spectrogram.min())

                lb = val_spectrogram[1:image_size+1, :image_size]
                hb = val_spectrogram[image_size+1:, :image_size]

                lb = lb.reshape(1, image_size, image_size)
                hb = hb.reshape(1, image_size, image_size)

                batch_lb.append(lb)
                batch_hb.append(hb)

            batch_lb = torch.stack(batch_lb).to(device)
            batch_hb = torch.stack(batch_hb).to(device)

            gen_hb = generator(batch_lb).to(device)

            # print(batch_lb)
            # print(type(batch_lb))
            # print(batch_lb.shape)

            # print(calculate_snr_tensor(batch_hb, gen_hb))
            batch_snr = calculate_snr_tensor(batch_hb, gen_hb)
            batch_lsd = calculate_lsd_tensor(batch_hb, gen_hb)

            # print(calculate_snr_tensor(batch_hb, gen_hb))
            total_snr += batch_snr.item()
            total_lsd += batch_lsd.item()
            count += 1

        avg_snr = total_snr / count
        avg_lsd = total_lsd / count

        informacoes = f"----EPOCA {num_epoca}----\nLoss:{loss}\nSNR médio (validação): {avg_snr:.2f} dB\nLSD médio (validação): {avg_lsd:.4f}\n"

        with open("auto_encoder/pesos/info_treinamento.txt", "a") as arquivo:
            arquivo.write(informacoes)

        print(f"SNR medio (validacao): {avg_snr:.2f} dB")
        print(f"LSD medio (validacao): {avg_lsd:.4f}")
