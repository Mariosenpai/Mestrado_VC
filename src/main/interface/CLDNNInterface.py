import os
from typing import Tuple

from torch.nn import functional as F
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from moduleExternal.seq2seqvc.seq2seq_vc.trainers.base import Trainer
from src.common.loggs.relatorio_validacao import Relatorio_validacao
from src.common.metricas import metricas_avalicao_model
from src.common.pre_processamento.noise import f0_constante
from src.common.pre_processamento.spectograma import mel_to_rgb
from src.main.model.CLDNN import CondUNet, CLDNNEncoder, CLDNN


class CLDNNInterface(Trainer):

    def __init__(self, steps, epochs, data_loader, sampler, model:CLDNN, vocoder, criterion, optimizer, scheduler, config, device,is_test):
        super().__init__(steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,is_test)
        self.device = device
        self.model = model
        self.loss_train = 0.0
        self.relatorio = None

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

    def _train_step(self, batch):

        # mel_in, mel_target: (B, T, F)
        audio, mel, mel_noise, sr, sentence = batch
        cldnn = self.model.cldnn
        vel_model = self.model.condUnet

        pred, groud_truth = self._generate_mel(cldnn, mel, mel_noise, vel_model, self.device)

        loss = self.criterion["mse"](pred, groud_truth)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_train += loss.item()

        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    def _generate_mel(self,cldnn, mel_noise, mel, vel_model: CondUNet, device):
        '''
        Gera o mel espectograma a partir do mel_noise

        a função retorna o mel gerado (pred) e o mel esperado (groud_truth)
        :param mel_noise:
        :param mel:
        :param vel_model:
        :return:
        '''
        # mel_in, mel_target: (B, T, F)
        # pred_list = []
        # target_list = []
        # for i in range(len(mel_noise)):

        mel_in = mel_noise.to(device)
        mel_target = mel.to(device)

        # get cond embedding from CLDNN (per-frame or global)
        cond = cldnn(mel_in)  # (B, T_cond, cond_dim)
        # for simplicity make cond per-utterance via mean (FiLM handles this)
        # but cond can be per-frame if you upsample/align it
        # convert mel_target to image shape (B, C=1, F, T)
        x1 = mel_target.permute(0, 2, 1).unsqueeze(1).contiguous()
        x0 = torch.randn_like(x1)  # noise source

        B = x1.shape[0]
        t = torch.rand(B, device=device)  # random times in [0,1]

        x_t, groud_truth = self._sample_linear_path(x0, x1, t)
        pred = vel_model(x_t, t, cond)  # (B, C, F, T)

        target_T = groud_truth.size(-1)  # 259
        pred = pred[..., :target_T]

        # pred_list.append(pred)
        # target_list.append(target_T)

        return pred, x1

    def _sample_linear_path(self,x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear interpolation path:
          x_t = (1-t) * x0 + t * x1
        derivative wrt t:
          dx_dt = x1 - x0    (constant, same shape)
        Args:
          x0, x1: (B, C, F, T)
          t: (B,) in [0,1]
        Returns:
          x_t: (B, C, F, T)
          dx_dt: (B, C, F, T)
        """
        # expand t to spatial shape
        B = x0.shape[0]
        shape = [B] + [1] * (x0.dim() - 1)
        t_sh = t.view(shape)
        x_t = (1.0 - t_sh) * x0 + t_sh * x1
        dx_dt = x1 - x0
        return x_t, dx_dt
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        # logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        self.model.eval()

        self._genearete_and_save_intermediate_result()

        # logging.info(
        #     f"(Steps: {self.steps}) Finished evaluation "
        # )

        # restore mode
        self.model.train()
    def _genearete_and_save_intermediate_result(self):

        # dirname = self._get_and_check_directory()
        total_mcd = 0
        total_psnr = 0
        total_snr = 0
        total_loss = 0

        dirname = os.path.join(self.config["outdir"], f"predictions\\{self.steps}steps")
        # generate
        # xs, _, ys, _, olens, spembs = tuple(
        #     [_.to(self.device) if _ is not None else _ for _ in batch]
        # )

        pbar = tqdm(self.data_loader["dev"], desc="[Validacao]", total=len(self.data_loader["dev"]))

        with torch.no_grad():

            for i, batch in enumerate(pbar):

                # mel_in, mel_target: (B, T, F)
                audio, mel, mel_noise, sr, sentence = batch
                cldnn = self.model.cldnn
                vel_model = self.model.condUnet

                pred, grouth_truth = self._generate_mel(cldnn, mel, mel_noise, vel_model, self.device)
                loss = self.criterion["mse"](pred, grouth_truth)

                total_loss += loss.item()
                grouth_truth = grouth_truth.squeeze(0).squeeze(0).detach().cpu().numpy()
                pred = pred.squeeze(0).squeeze(0).detach().cpu().numpy()

                metrica = metricas_avalicao_model(
                    grouth_truth,
                    pred
                )

                total_mcd += metrica.mcd
                total_snr += metrica.snr
                total_psnr += metrica.psnr

                if self.is_test and i > self.config["num_save_intermediate_results"]:
                    break

        pred = torch.tensor(pred).transpose(0, 1).detach().cpu().numpy()

        num_batches = i + 1
        avg_loss = total_loss / num_batches
        avg_mcd = total_mcd / num_batches
        avg_snr = total_snr / num_batches
        avg_psnr = total_psnr / num_batches

        audio = torch.tensor(audio).numpy()

        relatorio = self._create_relatorio(
            outs=pred,
            grouth_truth=grouth_truth,
            loss_val=avg_loss,
            loss_train=self.loss_train,
            audio=audio[0],
            idx=i,
            total_mcd=avg_mcd,
            total_snr=avg_snr,
            total_psnr=avg_psnr,
            sr=sr[0]
        )


        data = {
            "mdc": float(relatorio.mdc),
            "snr": relatorio.snr.float(),
            "psnr": relatorio.psnr,
            "loss_train": self.loss_train,
            "loss_val": avg_loss,
        }
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "relatorio.json"), "w", encoding="utf-8") as f:
            f.write(f"---------------------------------------------------------------------------------------------\n"
                    f"Metricas: {data}\n"
                    f"Steps:    {self.steps}\n"
                    f"Epochs:   {self.epochs}"
                    f"\n---------------------------------------------------------------------------------------------")


        self.set_relatorio(relatorio)