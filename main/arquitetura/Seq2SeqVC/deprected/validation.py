from typing import List

import torch
import numpy as np
from tqdm import tqdm

from main.loggs.relatorio_validacao import Relatorio_validacao
from main.metricas import metricas_avalicao_model



def one_validation(model, device, valid_loader, criterion, is_test=False):

    model.eval()

    total_loss = 0
    total_mcd = 0
    total_psnr = 0
    total_snr = 0

    pbar = tqdm(valid_loader, desc="Validacao", total=len(valid_loader))

    with torch.no_grad():

        for i, batch in enumerate(pbar):

            xs = batch["xs"].to(device)
            ilens = batch["ilens"].to(device)

            ys = batch["ys"].to(device)
            olens = batch["olens"].to(device)

            dp_inputs = batch["dp_inputs"].to(device)
            dplens = batch["dplens"].to(device)

            spembs = batch["spembs"]

            audios = batch["audio"]
            sr = batch["sr"]

            # ==========================
            # Forward
            # ==========================
            ret = model(
                xs,
                ilens,
                ys,
                olens,
                dp_inputs,
                dplens,
                spembs
            )

            after_outs = ret["after_outs"]
            before_outs = ret["before_outs"]
            ys_gt = ret["ys"]
            olens_gt = ret["olens"]

            bin_loss = ret["bin_loss"]

            # ==========================
            # Reconstruction loss
            # ==========================
            l1_loss = criterion(
                after_outs,
                before_outs,
                ys_gt,
                olens_gt
            )

            loss = l1_loss + ret["bin_loss"]
            total_loss += loss.item()

            # ==========================
            # Inference para métricas
            # ==========================

            outs_ori, d_outs = model.inference(
                src_speech=xs[0],
                tgt_speech=None,
                spembs=None,
                dp_input=xs[0],
                use_teacher_forcing=False
            )

            clean_audio_image = ys_gt[0]

            # alinhar tamanhos
            clean_audio_image, outs = fix_shape_min(
                clean_audio_image.unsqueeze(0),
                outs_ori.unsqueeze(0)
            )

            outs = gpu_to_cpu(outs.squeeze(0))
            clean_audio_image = gpu_to_cpu(clean_audio_image.squeeze(0))

            metrica = metricas_avalicao_model(
                clean_audio_image,
                outs
            )

            total_mcd += metrica.mcd
            total_snr += metrica.snr
            total_psnr += metrica.psnr

            pbar.set_postfix(loss=total_loss / (i + 1))

            if is_test:
                break

    # ==========================
    # Médias finais
    # ==========================
    total_loss /= (i + 1)
    total_mcd /= (i + 1)
    total_snr /= (i + 1)
    total_psnr /= (i + 1)

    return Relatorio_validacao(
        mdc=total_mcd,
        wer=0,
        snr=total_snr,
        psnr=total_psnr,
        loss=total_loss,
        pred=outs,
        sr=sr[0],  # ajuste se necessário
        grouth_truth=ys[0].detach().cpu(),
        audio_noise=xs[0].detach().cpu(),
        audio_pred=clean_audio_image,
        audio=audios[0]
    )

def fix_shape_min(obj1: torch.Tensor, obj2: torch.Tensor):
    T = min(obj1.size(1), obj2.size(1))
    b1 = obj1[:, :T, :]
    b2 = obj2[:, :T, :]
    return b1, b2


def gpu_to_cpu(obj):
    return obj.cpu().detach().numpy()
