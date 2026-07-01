import numpy as np
import torch
from matplotlib import pyplot as plt
import soundfile as sf
import os
import random

class Relatorio_validacao:
    def __init__(self, mdc, wer, snr, psnr, mosnet, msd,f0_rmse, f0_rmse_log,loss_val,loss_train, pred, grouth_truth, audio, audio_noise, audio_pred, sr):
        self.mdc = mdc
        self.wer = wer
        self.snr = snr
        self.psnr = psnr
        self.mosnet = mosnet
        self.msd = msd
        self.f0_rmse = f0_rmse
        self.f0_rmse_log = f0_rmse_log
        self.loss_val = loss_val
        self.loss_train = loss_train
        self.pred = pred
        self.grouth_truth = grouth_truth
        self.audio = audio
        self.audio_noise = audio_noise
        self.audio_pred = audio_pred
        self.sr = sr

    def gerar_relatorio_visual(
        self,
        output_dir="relatorio_visual",
        n_samples=10,
        seed=42,
        vmin=None,
        vmax=None
    ):
        """
        Gera um relatório visual com:
        - Spectrograma GT vs Pred (lado a lado)
        - Áudios correspondentes salvos em .wav
        """

        os.makedirs(output_dir, exist_ok=True)
        random.seed(seed)

        total = len(self.pred)
        indices = random.sample(range(total), min(n_samples, total))

        for i, idx in enumerate(indices):
            sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
            os.makedirs(sample_dir, exist_ok=True)

            # ====== Spectrogramas ======
            gt = self.grouth_truth[idx]
            pred = self.pred[idx]

            if torch.is_tensor(gt):
                gt = gt.detach().cpu().numpy()
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy()

            gt = np.squeeze(gt)
            pred = np.squeeze(pred)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            im0 = axes[0].imshow(
                gt,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )
            axes[0].set_title("Ground Truth")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel("Mel bins")

            im1 = axes[1].imshow(
                pred,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )
            axes[1].set_title("Prediction")
            axes[1].set_xlabel("Time")

            fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.8)
            plt.tight_layout()

            fig_path = os.path.join(sample_dir, "spectrograma.png")
            plt.savefig(fig_path, dpi=200)
            plt.close()

            # ====== Áudios ======
            sf.write(
                os.path.join(sample_dir, "audio_gt.wav"),
                self.audio[idx],
                self.sr
            )

            if self.audio_noise is not None:
                sf.write(
                    os.path.join(sample_dir, "audio_noise.wav"),
                    self.audio_noise[idx],
                    self.sr
                )

            sf.write(
                os.path.join(sample_dir, "audio_pred.wav"),
                self.audio_pred[idx],
                self.sr
            )

        print(f"Relatório visual gerado em: {output_dir}")