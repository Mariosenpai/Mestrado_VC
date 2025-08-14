import numpy as np


def lsd_mel(mel_clean, mel_noise):
    clean_log_spec = np.log(mel_clean.__abs__() + 1e-8)
    noisy_log_spec = np.log(mel_noise.__abs__() + 1e-8)
    lsd = np.sqrt(np.mean((clean_log_spec - noisy_log_spec) ** 2))
    return lsd


if __name__ == '__main__':
    mel = np.zeros((80,259))
    mel_f0 = np.zeros((80,259))
    lsd = lsd_mel(mel, mel_f0)
    print(f"LSD médio: {lsd:.2f}")
