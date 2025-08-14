import librosa
import numpy as np
import torch
from scipy import signal
from tqdm import tqdm

from main.arquitetura.GAN.fit.validacao import validacao

from main.config import image_size

from main.utils import salva_e_exibir_info


def train(trainloader, validloader, generator, discriminator, optimizer_gen, optimizer_dis, loss_gen, loss_dis,
          epoches=1, beta=1.0, device='cpu'):
    filename = 'reshapeAfterVit_V1.txt'
    list_loss = []
    alpha = 1.5

    NUM_COLS = 1024

    # TrainSteps
    for epoch in range(epoches):
        print(f'Epocas : {1 + epoch} / {epoches} ')
        num_samples_seen = 0
        total_gen_loss = 0
        total_dis_loss = 0

        # Iter on batches
        for data_batch in tqdm(trainloader):
            # print('START BATCH PROCESS')
            batch_lf = []
            batch_hf = []

            for data in data_batch:
                data = data.squeeze(dim=0)

                train_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096,
                                          window=signal.windows.hamming(4096))
                train_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(train_stft)))

                rows = train_spectrogram.shape[0]
                cols = train_spectrogram.shape[1]

                train_spectrogram = train_spectrogram.reshape(1, rows, cols).float()

                train_spectrogram_resize_1 = train_spectrogram[:, 1:1025, :image_size]
                train_spectrogram_resize_2 = train_spectrogram[:, 1025:, :image_size]

                if skipe_for(train_spectrogram_resize_1, train_spectrogram_resize_2): continue

                batch_lf.append(train_spectrogram_resize_1)
                batch_hf.append(train_spectrogram_resize_2)

                num_samples_seen += 1

            if skipe_for(train_spectrogram_resize_1, train_spectrogram_resize_2): continue

            batch_lf = torch.stack(batch_lf).to(device)
            batch_hf = torch.stack(batch_hf).to(device)

            shuffled_indexes = np.random.permutation(batch_lf.shape[0])  # shuffle
            batch_lf = batch_lf[shuffled_indexes]
            batch_hf = batch_hf[shuffled_indexes]

            # Train the discriminator on the true/generated data
            generated_data = generator(batch_lf)
            combined_data = torch.cat((batch_hf.to(device), generated_data.detach()), dim=0)
            labels = torch.cat((torch.ones(batch_hf.shape[0]), torch.zeros(generated_data.shape[0])), dim=0)

            shuffled_indexes = np.random.permutation(combined_data.shape[0])  # shuffle
            combined_data = combined_data[shuffled_indexes]
            labels = labels[shuffled_indexes].to(device)

            optimizer_dis.zero_grad()
            discriminator_out = discriminator(combined_data).reshape(-1)
            discriminator_loss = loss_dis(discriminator_out, labels)
            # print("Discriminator "+str(discriminator_loss.item()))
            discriminator_loss.backward()
            optimizer_dis.step()

            # Train the generator
            optimizer_gen.zero_grad()
            generator_out = generator(batch_lf)
            generator_loss = loss_gen(batch_hf, generator_out)

            discriminator_out_gen = discriminator(generator_out).reshape(-1)
            discriminator_loss_gen = loss_dis(discriminator_out_gen.to('cpu'),
                                              torch.ones(size=(discriminator_out_gen.shape[0],)))  # bce

            total_dis_loss = total_dis_loss + discriminator_loss_gen.detach()
            total_gen_loss = total_gen_loss + generator_loss.detach()

            # print("Generator content "+str(generator_loss.item()))
            # print("Generator adv "+str(discriminator_loss_gen.item()))

            loss = alpha * generator_loss + beta * discriminator_loss_gen
            list_loss.append(loss)
            loss.backward()
            optimizer_gen.step()
            # clip loss pytcho

        # End Trainloader Loop

        mean_gen_loss = total_gen_loss / num_samples_seen
        mean_dis_loss = total_dis_loss / num_samples_seen

        gen_order = torch.floor(torch.log10(mean_gen_loss))
        dis_order = 0 if mean_dis_loss == 0 else torch.floor(torch.log10(mean_dis_loss))
        b_pow = gen_order - dis_order
        if b_pow > 0:
            b_pow = b_pow
        beta = pow(10.0, b_pow)

        salva_e_exibir_info(generator, filename,
                            info_train(epoch, alpha, beta, loss, mean_dis_loss, discriminator_loss_gen, mean_gen_loss))

        try:
            print(list_loss.shape)
        except Exception as e:
            print(e)

        validacao(validloader, filename, generator, discriminator, loss_gen, loss_dis, beta=1.0, device='cuda')


def skipe_for(train_spectrogram_resize_1, train_spectrogram_resize_2):
    if (train_spectrogram_resize_1.shape[1] and train_spectrogram_resize_1.shape[2]) != image_size:
        return True

    if (train_spectrogram_resize_2.shape[1] and train_spectrogram_resize_2.shape[2]) != image_size:
        return True



def info_train(epoch, alpha, beta, loss, mean_dis_loss, discriminator_loss_gen, mean_gen_loss):
    return f"""  *** Resultado Treino ***
    EPOCH  {str(epoch + 1)}
    \t Alpha = {alpha} , Beta = {beta}
    \t -> Discriminative Loss during D Training =  + {str(mean_dis_loss.item())} , during G Training =  {str(mean_gen_loss.item())}
    \t -> Generative Loss = {str(loss.item())}  ---> alpha *  {str(mean_gen_loss.item())} beta *  {str(mean_dis_loss.item())}"""
