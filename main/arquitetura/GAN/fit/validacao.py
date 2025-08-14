import librosa
import numpy as np
import torch
from scipy import signal
from tqdm import tqdm

from main.arquitetura.GAN.fit.treinamento import skipe_for
from main.config import image_size
from main.utils import salva_e_exibir_info


def validacao(testloader, filename, generator, discriminator, loss_gen, loss_dis, beta=1.0, device='cpu'):
    generator.eval()
    discriminator.eval()

    total_gen_loss = 0
    total_dis_loss = 0
    num_samples_seen = 0

    with torch.no_grad():
        for data_batch in tqdm(testloader):
            batch_lf = []
            batch_hf = []

            for data in data_batch:
                data = data.squeeze(dim=0)

                test_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096,
                                         window=signal.windows.hamming(4096))
                test_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(test_stft)))

                rows = test_spectrogram.shape[0]
                cols = test_spectrogram.shape[1]

                test_spectrogram = test_spectrogram.reshape(1, rows, cols).float()
                test_spectrogram_resize_1 = test_spectrogram[:, 1:1025, :image_size]
                test_spectrogram_resize_2 = test_spectrogram[:, 1025:, :image_size]

                if skipe_for(test_spectrogram_resize_1, test_spectrogram_resize_2):
                    continue

                batch_lf.append(test_spectrogram_resize_1)
                batch_hf.append(test_spectrogram_resize_2)
                num_samples_seen += 1

            if not batch_lf or not batch_hf:
                continue

            batch_lf = torch.stack(batch_lf).to(device)
            batch_hf = torch.stack(batch_hf).to(device)

            # Generator forward pass
            generated_data = generator(batch_lf)
            generator_loss = loss_gen(batch_hf, generated_data)

            # Discriminator forward pass
            combined_data = torch.cat((batch_hf, generated_data), dim=0)
            labels = torch.cat((torch.ones(batch_hf.shape[0]), torch.zeros(generated_data.shape[0])), dim=0).to(device)

            discriminator_out = discriminator(combined_data).reshape(-1)
            discriminator_loss = loss_dis(discriminator_out, labels)

            total_gen_loss += generator_loss.detach()
            total_dis_loss += discriminator_loss.detach()

    mean_gen_loss = total_gen_loss / num_samples_seen
    mean_dis_loss = total_dis_loss / num_samples_seen

    salva_e_exibir_info(generator,filename, info_val(mean_gen_loss, mean_dis_loss, beta))

    return mean_gen_loss.item(), mean_dis_loss.item()

def info_val(mean_gen_loss,mean_dis_loss, beta):
    return f"""  *** Resultado Validação ***
          f"Generative Loss: {mean_gen_loss.item():.6f}
          f"Discriminative Loss: {mean_dis_loss.item():.6f}
          f"Weighted Loss: {beta * mean_dis_loss.item() + mean_gen_loss.item():.6f}"""