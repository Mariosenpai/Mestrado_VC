import logging
import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from bibliotecas_externas.seq2seqvc.seq2seq_vc.trainers import ARVCTrainer, AASVCTrainer
from main.arquitetura.Seq2SeqVC.deprected.validation import fix_shape_min
from main.loggs.relatorio_validacao import Relatorio_validacao
from main.metricas import metricas_avalicao_model
from main.pre_processamento.noise import f0_constante
from main.pre_processamento.spectograma import mel_to_rgb
from main.vocoder.HiFiGAN import sr_hifigan, mel_for_audio


class Trainer(AASVCTrainer):
    def __init__(self, steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,
                 device=torch.device("cpu"), is_test: bool = False):
        super().__init__(steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,
                         is_test,
                         device)
        self.relatorio = None

    def run_with_relatorio(self):
        self.run()
        if self.steps % self.config["eval_interval_steps"] == 0:
            for eval_steps_per_epoch, batch in enumerate(
                    tqdm(self.data_loader["dev"], desc="[eval]"), 1
            ):
                print(self._relatorio_create_log(batch))

    def run_wandb(self, project, config):
        """Run training."""
        self.backward_steps = 0
        self.all_loss = 0.0
        self.tqdm = tqdm(
            initial=self.steps, total=self.config["train_max_steps"], desc="[train]"
        )
        with wandb.init(project=project, config=config) as run:
            while True:
                # train one epoch
                self._train_epoch()

                run.log(self.relatorio)
                # check whether training is finished
                if self.finish_train:
                    break

        self.tqdm.close()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        self._genearete_and_save_intermediate_result()

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
        )

        # restore mode
        self.model.train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws

        # check directory
        #dirname = self._get_and_check_directory()
        total_mcd = 0
        total_psnr = 0
        total_snr = 0
        total_loss = 0

        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        # generate
        # xs, _, ys, _, olens, spembs = tuple(
        #     [_.to(self.device) if _ is not None else _ for _ in batch]
        # )

        pbar = tqdm(self.data_loader["dev"], desc="Validacao", total=len(self.data_loader["dev"]))

        with torch.no_grad():

            for i, batch in enumerate(pbar):

                xs = batch["xs"].to(self.device)
                ilens = batch["ilens"].to(self.device)

                ys = batch["ys"].to(self.device)
                olens = batch["olens"].to(self.device)

                dp_inputs = batch["dp_inputs"].to(self.device)
                dplens = batch["dplens"].to(self.device)

                spembs = batch["spembs"]

                audios = batch["audio"]
                srs = batch["sr"]

                # ==========================
                # Forward
                # ==========================
                ret = self.model(
                    xs,
                    ilens,
                    ys,
                    olens,
                    dp_inputs,
                    dplens,
                    spembs
                )

                ds = ret["ds"]
                ilens_ = ret["ilens"]
                olens_ = ret["olens"]
                bin_loss = ret["bin_loss"]
                log_p_attn = ret["log_p_attn"]
                olens_reduced = ret["olens_reduced"]
                ys_gt = ret["ys"]

                # ==========================
                # Reconstruction loss
                # ==========================
                gen_loss = self._calculate_loss(ret, ds, olens_, ilens_, bin_loss, log_p_attn, olens_reduced)

                # loss = l1_loss + ret["bin_loss"]
                total_loss += gen_loss.item()

                # ==========================
                # Inference para métricas
                # ==========================

                x = xs[0] # [:ilens_]
                y = ys_gt[0]  # ys[0][:olens_]

                outs_ori, d_outs, *other = self.model.inference(
                    src_speech=x,
                    tgt_speech=y,
                    spembs=spembs,
                    dp_input=dp_inputs[0],
                    use_teacher_forcing=False
                )

                clean_audio_image = y

                # alinhar tamanhos
                clean_audio_image, outs = fix_shape_min(
                    clean_audio_image.unsqueeze(0),
                    outs_ori.unsqueeze(0)
                )

                outs = self.gpu_to_cpu(outs.squeeze(0))
                clean_audio_image = self.gpu_to_cpu(clean_audio_image.squeeze(0))

                metrica = metricas_avalicao_model(
                    clean_audio_image,
                    outs
                )

                total_mcd += metrica.mcd
                total_snr += metrica.snr
                total_psnr += metrica.psnr

                if i <= self.config["num_save_intermediate_results"]:
                    self._plot_and_save(
                        outs,
                        dirname + f"/{i}_out.png",
                        ref=clean_audio_image,
                        origin="lower",
                    )

                pbar.set_postfix(loss=total_loss / (i + 1))

                if self.is_test:
                    break

        # self.set_relatorio(
        #     self._create_relatorio(
        #         outs=outs,
        #         y=ys[0],
        #         audio=audios[0],
        #         idx=i,
        #         total_mcd=total_mcd,
        #         total_snr=total_snr,
        #         total_psnr=total_psnr,
        #         sr=srs[0]
        #
        #     )
        # )

        relatorio = self._create_relatorio(
            outs=outs,
            y=ys[0],
            audio=audios[0],
            idx=i,
            total_mcd=total_mcd,
            total_snr=total_snr,
            total_psnr=total_psnr,
            sr=srs[0]
        )

        print(relatorio)

    def fix_shape_min(self, obj1: torch.Tensor, obj2: torch.Tensor):
        T = min(obj1.size(1), obj2.size(1))
        b1 = obj1[:, :T, :]
        b2 = obj2[:, :T, :]
        return b1, b2

    def set_relatorio(self, relatorio):

        self.relatorio = {
            "MDC": relatorio.mdc,
            "WER": 0,
            "SNR": relatorio.snr,
            "PSNR": relatorio.psnr,
            "loss_train": {
                "L1": self.total_train_loss["train/l1_loss"],
                "BCE": self.total_train_loss["train/bce_loss"],
                "loss": self.total_train_loss["train/loss"],
            },
            "loss_val": relatorio.loss,
            "mel_exemple": {
                "ground_truth": wandb.Image(mel_to_rgb(relatorio.grouth_truth)),
                "prediction": wandb.Image(mel_to_rgb(relatorio.pred)),
            },
            "audio_exemple": {
                "ground_truth": wandb.Audio(relatorio.audio, sample_rate=int(relatorio.sr)),
                "noise": wandb.Audio(relatorio.audio_noise, sample_rate=int(relatorio.sr)),
                "prediction": wandb.Audio(relatorio.audio_pred, sample_rate=sr_hifigan()),
            },
            "epocas": self.epochs + 1
        }

    def _create_relatorio(self, outs, y, audio, idx, total_mcd, total_snr, total_psnr, sr):
        def gpu_to_cpu(obj):
            return obj.cpu().detach().numpy()

        mel_spec_final = torch.Tensor(outs).unsqueeze(0).permute(0, 2, 1)
        audio_pred = mel_for_audio(mel_spec_final.to('cuda'))
        audio_noise = f0_constante(audio.astype(np.float64), sr=sr)

        total_loss = 0  #total_loss / (idx + 1)
        total_mcd += total_mcd / (idx + 1)
        total_snr += total_snr / (idx + 1)
        total_psnr += total_psnr / (idx + 1)

        clean_audio_image = gpu_to_cpu(mel_spec_final[0].permute(1, 0))
        audio_pred = gpu_to_cpu(audio_pred[0][0])
        before_outs = gpu_to_cpu(mel_spec_final[0])

        return Relatorio_validacao(
            mdc=total_mcd,
            wer=0,
            snr=total_snr,
            psnr=total_psnr,
            loss=total_loss,
            pred=before_outs,
            sr=sr,
            grouth_truth=clean_audio_image,
            audio_noise=audio_noise,
            audio_pred=audio_pred,
            audio=audio
        )

    def _plot_and_save(self,
                       array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
                       ):
        shape = array.shape
        if len(shape) == 1:
            # for eos probability
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(array)
            plt.xlabel("Frame")
            plt.ylabel("Probability")
            plt.ylim([0, 1])
        elif len(shape) == 2:
            # for tacotron 2 attention weights, whose shape is (out_length, in_length)
            if ref is None:
                plt.figure(figsize=figsize, dpi=dpi)
                plt.imshow(array.T, aspect="auto", origin=origin)
                plt.xlabel("Input")
                plt.ylabel("Output")
            else:
                plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                plt.subplot(1, 2, 1)
                plt.imshow(array.T, aspect="auto", origin=origin)
                plt.xlabel("Input")
                plt.ylabel("Output")
                plt.subplot(1, 2, 2)
                plt.imshow(ref.T, aspect="auto", origin=origin)
                plt.xlabel("Input")
                plt.ylabel("Output")
        elif len(shape) == 4:
            # for transformer attention weights,
            # whose shape is (#leyers, #heads, out_length, in_length)
            plt.figure(
                figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
            )
            for idx1, xs in enumerate(array):
                for idx2, x in enumerate(xs, 1):
                    plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                    plt.imshow(x, aspect="auto")
                    plt.xlabel("Input")
                    plt.ylabel("Output")
        else:
            raise NotImplementedError("Support only from 1D to 4D array.")
        plt.tight_layout()
        if not os.path.exists(os.path.dirname(figname)):
            # NOTE: exist_ok = True is needed for parallel process decoding
            os.makedirs(os.path.dirname(figname), exist_ok=True)
        plt.savefig(figname)
        plt.close()

    def gpu_to_cpu(self, obj):
        return obj.cpu().detach().numpy()
