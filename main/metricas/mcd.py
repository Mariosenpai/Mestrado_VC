import numpy as np

from mel_cepstral_distance import compare_mel_spectrograms


def _mcd(C, C_hat):
    """C and C_hat are NumPy arrays of shape (T, D),
    representing mel-cepstral coefficients.
    """
    K = (10 * np.sqrt(2)) / np.log(10)
    return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))

def compare_mel(mel,mel_noise):
    distortions, pen = compare_mel_spectrograms(mel, mel_noise)
    return distortions

if __name__ == '__main__':
    mel = np.zeros((80, 259))
    mel_f0 = np.ones((80, 259))

    distortions, pen = compare_mel_spectrograms(mel, mel_f0)

    print("distortions :", distortions)
    print("Penality: ", pen)
