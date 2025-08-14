
from matplotlib import pyplot as plt

from main.arquitetura.auto_encoder.fit.validacao.FMA import validacao_melhoramento_auido
from main.config import path_save_auto


import librosa
import numpy as np
import torch
from scipy import signal
from tqdm import tqdm

from main.config import image_size


from main.utils import save_model


def treinamento_melhoramento_audio(generator, loss_gen, optimizer_gen, epochs, train_loader, val_loader, device):

    for epoch in range(epochs):
        epoch += 1
        history = []
        print(f'Epocas : {epoch} / {epochs} ')
        total_loss = 0
        k = 0
        for i, data_batch in enumerate(tqdm(train_loader)):

            batch_lb = []
            batch_hb = []

            for data in data_batch:
                data = data.squeeze(dim=0)

                train_stft = librosa.stft(np.asarray(data), n_fft=4096, win_length=4096,
                                          window=signal.windows.hamming(4096))

                train_spectrogram = torch.tensor(librosa.amplitude_to_db(abs(train_stft))).to(device)
                train_spectrogram = (train_spectrogram - train_spectrogram.min()) / (
                        train_spectrogram.max() - train_spectrogram.min())

                lb = train_spectrogram[1:image_size + 1, :image_size]
                hb = train_spectrogram[image_size + 1:, :image_size]

                lb = lb.reshape(1, image_size, image_size)
                hb = hb.reshape(1, image_size, image_size)

                batch_lb.append(lb)
                batch_hb.append(hb)

            batch_lb = torch.stack(batch_lb).to(device)
            batch_hb = torch.stack(batch_hb).to(device)

            gen_hb = generator(batch_lb).to(device)

            # gen_hb = torch.nn.functional.log_softmax(gen_hb, dim=1).to(device)
            # batch_hb = torch.nn.functional.log_softmax(batch_hb, dim=1).to(device)

            optimizer_gen.zero_grad()
            loss = loss_gen(gen_hb, batch_hb)
            total_loss += loss.detach()
            k += 1
            loss.backward()

            optimizer_gen.step()

            # print(f'Loss: {loss.item()}')
            history.append(loss.item())

            if i > 100:
                if str(loss.item()) == "nan":
                    break

                    # if i in [1000,2000,3000,4000,5000,6000]:
        #     print(loss.item())
        #     plt.plot(history,label="loss")
        #     plt.show()

        total_loss = total_loss / k

        save_model(generator, path_save_auto, f"auto_encoder_epoca_{epoch}")

        plt.plot(history, label="loss")

        plt.show()

        print("Loss : " + str(total_loss.item()))
        # print("Mean loss"+str(total_loss))

        print("--------- VALIDACAO ---------")
        validacao_melhoramento_auido(val_loader, generator, str(total_loss.item()), epoch, device)

    return generator




