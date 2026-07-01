from pathlib import Path

import librosa
import numpy as np
import scipy
import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
from torch import nn
from tqdm import tqdm

'''
link: https://github.com/lochenchou/MOSNet/blob/master/custom_test.py
artigo : https://arxiv.org/pdf/1904.08352
'''

class CNN_BLSTM(object):

    def __init__(self):
        print('CNN_BLSTM init')

    def build(self):
        _input = keras.Input(shape=(None, 257))

        re_input = layers.Reshape((-1, 257, 1), input_shape=(-1, 257))(_input)

        # CNN
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv1)

        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv2)

        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv3)

        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3, 3), strides=(1, 3), activation='relu', padding='same'))(conv4)

        re_shape = layers.Reshape((-1, 4 * 128), input_shape=(-1, 4, 128))(conv4)

        # BLSTM
        blstm1 = Bidirectional(
            LSTM(128, return_sequences=True, dropout=0.3,
                 recurrent_dropout=0.3, recurrent_constraint=max_norm(0.00001)),
            merge_mode='concat')(re_shape)

        # DNN
        flatten = TimeDistributed(layers.Flatten())(blstm1)
        dense1 = TimeDistributed(Dense(128, activation='relu'))(flatten)
        dense1 = Dropout(0.3)(dense1)

        frame_score = TimeDistributed(Dense(1), name='frame')(dense1)

        average_score = layers.GlobalAveragePooling1D(name='avg')(frame_score)

        model = Model(outputs=[average_score, frame_score], inputs=_input)

        return model

FS = 16000
FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH=256
WIN_LENGTH=512

def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE):
    # Loading sound file
    y, _ = librosa.load(sound_file, sr=fs)  # or set sr to hp.sr.

    # Preemphasis
    # y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                          n_fft=fft_size,
                          hop_length=HOP_LENGTH,
                          win_length=WIN_LENGTH,
                          window=scipy.signal.windows.hamming,
                          )

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft/2, T)

    # shape in (T, 1+n_fft/2)
    return np.transpose(mag.astype(np.float32))


def load_model() -> nn.Module:
    MOSNet = CNN_BLSTM()
    model = MOSNet.build()
    ROOT = Path(__file__).resolve().parent.parent.parent.parent
    checkpoint = ROOT / "src" / "common" / "metricas" / "checkpoint_model_mos" / "mosnet_checkpoint.h5"
    model.load_weights(checkpoint)
    return model

def preprocesse_melgram(mag_sgram):
    timestep = mag_sgram.shape[0]
    mag_sgram = np.reshape(mag_sgram, (1, timestep, SGRAM_DIM))
    return mag_sgram

def _mosnet(audio) -> float:

    model = load_model()
    mag_sgram = get_spectrograms(audio)
    mag_sgram = preprocesse_melgram(mag_sgram)

    Average_score, _ = model.predict(mag_sgram, verbose=0, batch_size=1)

    return Average_score[0][0]


if __name__ == "__main__":

    audio_ruido = fr"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\2024_AUDIOS_PROJETO_LARINGE\SEM_TRAQUEOSTOMIA\DALLETE_FONO\LARINGE_ELETRONICA_MP3\2_d.mp3"
    audio_sem = fr"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\2024_AUDIOS_PROJETO_LARINGE\SEM_TRAQUEOSTOMIA\DALLETE_FONO\NATURAL_MP3\2_n.mp3"
    print(_mosnet(audio_sem))
    print(_mosnet(audio_ruido))