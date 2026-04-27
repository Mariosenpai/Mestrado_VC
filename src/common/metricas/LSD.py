import torch


def calculate_lsd_tensor(original: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """
    Calcula a métrica Log-Spectral Distance (LSD) entre dois tensores.

    Parâmetros:
    - original: Tensor com o espectrograma original (real).
    - generated: Tensor com o espectrograma gerado (previsão do modelo).

    Retorna:
    - LSD: Tensor escalar representando a distância espectral logarítmica.
    """
    # eps = 1e-10  # Para evitar log de zero
    # original = torch.maximum(original, torch.tensor(eps, device=original.device))
    # generated = torch.maximum(generated, torch.tensor(eps, device=generated.device))

    # log_original = 10 * torch.log10(original)
    # log_generated = 10 * torch.log10(generated)

    # lsd = torch.sqrt(torch.mean((log_original - log_generated) ** 2, dim=(1, 2)))
    # return torch.mean(lsd)  # Média para o batch

    est, target = original, generated
    assert est.shape == target.shape, "Spectrograms must have the same shape."
    est = est.squeeze(0).squeeze(0) ** 2
    target = target.squeeze(0).squeeze(0) ** 2
    # Compute the log of the magnitude spectrograms (adding a small epsilon to avoid log(0))
    epsilon = 1e-10
    log_spectrogram1 = torch.log10(target + epsilon)
    log_spectrogram2 = torch.log10(est + epsilon)
    squared_diff = (log_spectrogram1 - log_spectrogram2) ** 2
    squared_diff = torch.mean(squared_diff, dim=1) ** 0.5
    lsd = torch.mean(squared_diff, dim=0)

    return lsd