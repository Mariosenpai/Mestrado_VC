import logging
import os

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import soundfile as sf

from moduleExternal.seq2seqvc.seq2seq_vc.trainers import AASVCTrainer
#from src.main.model.voiceConversion.Seq2SeqVC.deprected.validation import fix_shape_min
from src.common.loggs.relatorio_validacao import Relatorio_validacao
from src.common.metricas import metricas_avalicao_model
from src.common.pre_processamento.noise import f0_constante
from src.common.pre_processamento.spectograma import mel_to_rgb
from src.common.vocoder.HiFiGAN import sr_hifigan


class AASVCTrainerInterface(AASVCTrainer):
    def __init__(self, steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,
                 device=torch.device("cuda"), is_test: bool = False):
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

        pbar = tqdm(self.data_loader["dev"], desc="[Validacao]", total=len(self.data_loader["dev"]))

        with torch.no_grad():

            for i, batch in enumerate(pbar):

                # ==========================
                # Fragmentação do Batch
                # ==========================

                xs = batch["xs"].to(self.device)
                ilens = batch["ilens"].to(self.device)

                ys = batch["ys"].to(self.device)
                olens = batch["olens"].to(self.device)

                duraction_predict = batch["dp_inputs"].to(self.device)
                dplens = batch["dplens"].to(self.device)

                spembs = batch["spembs"]

                audios = batch["audio"]
                srs = batch["sr"]

                # ==========================
                # Forward
                # ==========================
                output_model = self.model(
                    xs,
                    ilens,
                    ys,
                    olens,
                    duraction_predict,
                    dplens,
                    spembs
                )

                ds = output_model["ds"]
                after_outs = output_model["after_outs"]
                ilens_ = output_model["ilens"]
                olens_ = output_model["olens"]
                bin_loss = output_model["bin_loss"]
                log_p_attn = output_model["log_p_attn"]
                olens_reduced = output_model["olens_reduced"]
                ys_gt = output_model["ys"]

                # ==========================
                # Reconstruction loss
                # ==========================
                gen_loss = self._calculate_loss(output_model, ds, olens_, ilens_, bin_loss, log_p_attn, olens_reduced)

                # loss = l1_loss + ret["bin_loss"]
                total_loss += gen_loss.item()

                # ==========================
                # métricas
                # ==========================

                grouth_truth = ys_gt[0]
                x = xs[0]

                output_inference, d_outs, *other = self.model.inference(
                    src_speech=x,
                    tgt_speech=grouth_truth,
                    spembs=spembs,
                    dp_input=duraction_predict[0],
                    use_teacher_forcing=False
                )

                # alinhar tamanhos
                grouth_truth, output_inference = self.fix_shape_min(
                    grouth_truth.unsqueeze(0),
                    output_inference.unsqueeze(0)
                )

                output_inference_gpu = output_inference.squeeze(0).clone()
                output_inference = self.gpu_to_cpu(output_inference.squeeze(0))
                grouth_truth = self.gpu_to_cpu(grouth_truth.squeeze(0))

                metrica = metricas_avalicao_model(
                    grouth_truth,
                    output_inference
                )

                total_mcd += metrica.mcd
                total_snr += metrica.snr
                total_psnr += metrica.psnr

                # ==========================
                # Inference
                # ==========================
                if i <= self.config["num_save_intermediate_results"]:

                    self._plot_and_save(
                        after_outs[0].cpu().numpy(),
                        dirname + f"/{i}_out_model.png",
                        ref=grouth_truth,
                        origin="lower",
                    )

                    self._plot_and_save(
                        output_inference,
                        dirname + f"/{i}_out_inference.png",
                        ref=grouth_truth,
                        origin="lower",
                    )

                    if self.vocoder is not None:
                        if not os.path.exists(os.path.join(dirname, "wav")):
                            os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)

                        y = self.vocoder.inference(
                            output_inference_gpu.float().transpose(1,0).unsqueeze(0))  # tem q ser um tensor
                        audio = y.squeeze(0).squeeze(0).detach().cpu().numpy().astype("float32")

                        sf.write(
                            os.path.join(dirname, "wav", f"{i}_gen.wav"),
                            audio,
                            self.vocoder.sr_hifigan(),
                            "PCM_16",
                        )

                pbar.set_postfix(loss=total_loss / (i + 1))

                if self.is_test and i > self.config["num_save_intermediate_results"]:
                    break

        num_batches = i + 1

        avg_loss = total_loss / num_batches
        avg_mcd = total_mcd / num_batches
        avg_snr = total_snr / num_batches
        avg_psnr = total_psnr / num_batches

        relatorio = self._create_relatorio(
            outs=output_inference,
            grouth_truth=grouth_truth,
            loss_val=avg_loss,
            loss_train=self.loss_train,
            audio=audios[0],
            idx=i,
            total_mcd=avg_mcd,
            total_snr=avg_snr,
            total_psnr=avg_psnr,
            sr=srs[0]
        )

        data = {
            "mdc": float(relatorio.mdc),
            "snr": relatorio.snr.float(),
            "psnr": relatorio.psnr,
            "loss_train": self.loss_train,
            "loss_val": avg_loss,
        }

        with open(os.path.join(dirname, "relatorio.json"), "w") as f:
            f.write(f"---------------------------------------------------------------------------------------------\n"
                    f"Metricas: {data}\n"
                    f"Steps:    {self.steps}\n"
                    f"Epochs:   {self.epochs}"
                    f"\n---------------------------------------------------------------------------------------------")

        self.set_relatorio(relatorio)

    def inference(self, batch, output_path):
        x = torch.Tensor(batch["mel_noise"])
        ground_truth = torch.Tensor(batch["mel"])
        duraction_predict = torch.Tensor(batch["dp_inputs"])

        output_inference, d_outs, *other = self.model.inference(
            src_speech=x,
            tgt_speech=ground_truth,
            spembs=None,
            dp_input=duraction_predict[0],
            use_teacher_forcing=False
        )
        y = self.vocoder(output_inference.float().detach().numpy(), output_path, 0)

        return y, output_inference

    def _vocoder_inference(self, output_inference):
        if self.vocoder is not None:
            y = self.vocoder.inference(
                torch.Tensor(output_inference).float().transpose(1, 0).unsqueeze(0).cuda()
            )  # tem q ser um tensor
            audio = y.squeeze(0).squeeze(0).detach().cpu().numpy().astype("float32")
            return audio, self.vocoder.sr_hifigan()

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
                "loss_train": relatorio.loss_train,
                "loss_val": relatorio.loss_val
            },
            "mel_exemple": {
                "ground_truth": wandb.Image(mel_to_rgb(relatorio.grouth_truth)),
                "prediction": wandb.Image(mel_to_rgb(relatorio.pred)),
            },
            "audio_exemple": {
                "ground_truth": wandb.Audio(relatorio.audio, sample_rate=int(relatorio.sr)),
                "noise": wandb.Audio(relatorio.audio_noise, sample_rate=int(relatorio.sr)),
                "prediction": wandb.Audio(relatorio.audio_pred, sample_rate=sr_hifigan()),
            },
            "epocas": self.epochs,
            "steps": self.steps
        }

    def _create_relatorio(self, outs, grouth_truth, loss_val, loss_train, audio, idx, total_mcd, total_snr, total_psnr,
                          sr):
        def gpu_to_cpu(obj):
            return torch.tensor(obj).cpu().detach().numpy()

        mel_spec_final = torch.Tensor(outs).unsqueeze(0).permute(0, 2, 1)
        clean_audio_image = torch.Tensor(grouth_truth).unsqueeze(0).permute(0, 2, 1)

        audio_pred, _ = self._vocoder_inference(outs)
        audio_noise = f0_constante(audio.astype(np.float64), sr=sr)

        audio_pred = gpu_to_cpu(audio_pred)
        before_outs = gpu_to_cpu(mel_spec_final[0])
        clean_audio_image = gpu_to_cpu(clean_audio_image[0])

        return Relatorio_validacao(
            mdc=total_mcd,
            wer=0,
            snr=total_snr,
            psnr=total_psnr,
            loss_val=loss_val,
            loss_train=loss_train,
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
