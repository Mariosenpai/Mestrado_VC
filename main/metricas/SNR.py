import torch


def calculate_snr_tensor(original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """
    Calcula a métrica Signal-to-Noise Ratio (SNR) entre dois tensores.

    Parâmetros:
    - original: tensor com o espectrograma original (real).
    - generated: tensor com o espectrograma gerado (previsão do modelo).

    Retorna:
    - SNR: Tensor escalar representando a relação sinal-ruído em dB.
    """
    # eps = 1e-10  # Para evitar divisão por zero
    # signal_power = torch.sum(original ** 2, dim=(1, 2))  # Soma sobre frequência e tempo
    # noise_power = torch.sum((original - generated) ** 2, dim=(1, 2))
    # snr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    # return torch.mean(snr)  # Média para o batch

    signal = original  ## input orignal data
    mean_signal = torch.mean(signal)
    signal_diff = torch.subtract(signal, mean_signal)
    var_signal = torch.sum(torch.mean(signal_diff ** 2))  ## variance of orignal data

    noisy_signal = generated  ## input noisy data
    noise = torch.subtract(noisy_signal, signal)
    mean_noise = torch.mean(noise)
    noise_diff = torch.subtract(noise, mean_noise)
    var_noise = torch.sum(torch.mean(noise_diff ** 2))  ## variance of noise

    if var_noise == 0:
        snr = 100  ## clean image
    else:
        snr = (torch.log10(var_signal / var_noise)) * 10  ## SNR of the data

    return snr
