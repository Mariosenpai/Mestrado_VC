import os
from typing import Tuple

from torch.nn import functional as F
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm

from moduleExternal.seq2seqvc.seq2seq_vc.trainers.base import Trainer
from src.common.loggs.relatorio_validacao import Relatorio_validacao
from src.common.metricas import  metricas_geral, Metricas
from src.common.pre_processamento.noise import f0_constante
from src.common.pre_processamento.spectograma import mel_to_rgb
from src.main.model.CLDNN import CondUNet, CLDNNEncoder, CLDNN


class CLDNNInterface(Trainer):

    def __init__(self, steps, epochs, data_loader, sampler, model:CLDNN, vocoder, criterion, optimizer, scheduler, config, device,is_test,mod="base"):
        super().__init__(steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,is_test)
        self.device = device
        self.model = model
        self.loss_train = 0.0
        self.relatorio = None

    def get_device(self):
        return self.device
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
                if self.finish_train or self.is_test:
                    break

        self.tqdm.close()

    def _train_step(self, batch):

        # mel_in, mel_target: (B, T, F)
        audio, mel, mel_noise, sr, sentence = batch
        cldnn = self.model.cldnn
        vel_model = self.model.condUnet

        pred, groud_truth = self._get_training_type(cldnn,mel, mel_noise,vel_model)

        loss = self.criterion["mse"](pred, groud_truth)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_train += loss.item()

        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_flow(self, cldnn, mel, mel_noise):
        flow = cldnn

        x_1 = mel
        x_0 = mel_noise
        t = torch.rand(x_1.shape[0], 1, 1, device=x_1.device)

        x_t = (1-t) * x_0 + t * x_1
        dx_t = x_1 - x_0
        t = t.expand(-1, x_t.shape[1], 1)

        pred = flow(mel.to(self.device), x_t.to(self.device), t.to(self.device))
        groud_truth = dx_t.to(self.device)

        return pred.to(self.device) , groud_truth.to(self.device)


    def _inference_flow(self,x , n_steps):
        time_steps = torch.linspace(0, 1.0, n_steps + 1).to(self.device)
        x = x.to(self.device)
        for i in range(n_steps):
            x = self.model.cldnn.step(x, time_steps[i], time_steps[i + 1])

        return x

    def get_inference_type(self,**kwargs ):

        if self.model.cldnn.use_flow_matching:
            pred = self._inference_flow(x= kwargs["mel_noise"],n_steps=kwargs['n_steps'])
        else:
            pred, _ = self._generate_mel(
                cldnn=self.model.cldnn,
                mel=kwargs["mel"],
                mel_noise=kwargs["mel_noise"],
                vel_model=self.model.condUnet,
                device=self.device,
            )
        return pred


    def _get_training_type(self, cldnn, mel, mel_noise, vel_model=None):
        if self.model.cldnn.use_flow_matching:
            pred, groud_truth = self._train_flow(cldnn, mel, mel_noise)
        else:
            pred, groud_truth = self._generate_mel(cldnn, mel, mel_noise, vel_model, self.device)
        return pred, groud_truth

    def _generate_mel(self,cldnn, mel_noise, mel, vel_model, device):
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
        cond = cldnn(mel_in, None,None)  # (B, T_cond, cond_dim)
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
        # total_mcd = 0
        # total_psnr = 0
        # total_snr = 0
        total_loss = 0
        # total_mosnet = 0
        # total_f0_rmse = 0
        # total_f0_rmse_log = 0
        # total_msd = 0
        metrica = Metricas()

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

                pred, grouth_truth = self._get_training_type(cldnn, mel, mel_noise, vel_model)

                loss = self.criterion["mse"](pred, grouth_truth)

                total_loss += loss.item()

                grouth_truth = mel

                mel_noise = torch.tensor(mel_noise).to(self.device)
                mel = torch.tensor(mel).to(self.device)
                pred_infe = self.get_inference_type(mel_noise=mel_noise, mel=mel, n_steps=8)

                if pred.dim() == 4:
                    grouth_truth = grouth_truth.squeeze(0).detach().cpu().numpy()

                pred_infe = pred_infe.detach().cpu().numpy()
                grouth_truth = grouth_truth.detach().cpu().numpy()

                metrica = self._metricas_avalicao(metrica,grouth_truth, pred_infe, audio, sr)

                if self.is_test and i > self.config["num_save_intermediate_results"]:
                    break

        grouth_truth = grouth_truth.squeeze(0)

        pred = torch.tensor(pred_infe).transpose(1,2).detach().cpu().numpy()

        num_batches = i + 1
        avg_loss = total_loss / num_batches
        avg_mcd = metrica.mcd / num_batches
        avg_snr = metrica.snr / num_batches
        avg_psnr = metrica.psnr / num_batches
        avg_f0_rmse = metrica.f0_rmse / num_batches
        avg_f0_rmse_log = metrica.f0_rmse_log / num_batches
        avg_msd = metrica.msd / num_batches
        avg_mosnet = metrica.mosnet / num_batches

        audio = torch.tensor(audio).numpy()

        relatorio = self._create_relatorio(
            outs=pred,
            grouth_truth=grouth_truth,
            loss_val=avg_loss,
            loss_train=self.loss_train,
            audio=audio[0],
            total_mcd=avg_mcd,
            total_snr=avg_snr,
            total_psnr=avg_psnr,
            total_msd=avg_msd,
            total_mosnet=avg_mosnet,
            total_f0_rmse=avg_f0_rmse,
            total_f0_rmse_log=avg_f0_rmse_log,
            sr=sr[0]
        )


        data = {
            "mdc": float(relatorio.mdc),
            "snr": relatorio.snr.float(),
            "psnr": relatorio.psnr,
            "mosnet": relatorio.mosnet,
            "f0_rmse": relatorio.f0_rmse,
            "f0_rmse_log": relatorio.f0_rmse_log,
            "msd": relatorio.msd,
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