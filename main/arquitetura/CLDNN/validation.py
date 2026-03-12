
import numpy as np
import torch


from main.arquitetura.CLDNN.model import sample_linear_path
from main.arquitetura.CLDNN.train import generate_mel
from main.loggs.relatorio_validacao import Relatorio_validacao
from main.pre_processamento.noise import f0_constante
from tqdm.auto import tqdm
from main.metricas import mcd, wer, wer_with_trans, snr, psnr

from main.vocoder.HiFiGAN import mel_for_audio, sr_hifigan





def validation(model, vel_model, val_loader, device="cuda", log_every=100, on_wer=True) -> Relatorio_validacao:
    global dx_dt, pred, audio, sr, groud_truth

    model.eval()
    vel_model.eval()

    total_loss = 0
    total_mdc = 0
    total_index = 0
    total_wer = 0
    total_snr = 0
    total_psnr = 0

    pbar = tqdm(val_loader, desc="Validacao", total=len(val_loader))

    for i, batch in enumerate(pbar):

        audio, mel, mel_noise, sr, sentence = batch

        pred, groud_truth = generate_mel(model, mel, mel_noise, vel_model, device)

        loss = torch.nn.functional.mse_loss(pred, groud_truth)

        pred = change_shape(pred)
        groud_truth = change_shape(groud_truth)

        mcd_log, snr_log, psnr_log, wer_log = metrics_avaliation(groud_truth, pred, sentence, on_wer)

        total_loss += loss.item()
        total_mdc += mcd_log
        total_wer += wer_log
        total_snr += snr_log
        total_psnr += psnr_log

        total_index += 1


    audio_pred = mel_for_audio(torch.Tensor(pred).to('cuda')).cpu().detach()
    audio_noise = f0_constante(audio[0].astype(np.float64), sr=sr[0])
    audio_pred = audio_pred[0].cpu().detach().numpy()

    final_loss = total_loss / total_index
    final_mdc = total_mdc / total_index
    final_wer = total_wer / total_index
    final_snr = total_snr / total_index
    final_psnr = total_psnr / total_index

    relatorio = Relatorio_validacao(
        final_mdc,
        final_wer,
        final_snr,
        final_psnr,
        final_loss,
        pred,
        groud_truth,
        audio[0],
        audio_noise,
        audio_pred,
        sr[0]
    )

    return relatorio


def metrics_avaliation(dx_dt, pred, sentence, on_wer):
    '''
    Retorna as metricas de avalicao do modelo
    :param dx_dt:
    :param pred:
    :param sentence:
    :param on_wer:
    :return:
    '''

    mcd_log = mcd(dx_dt, pred)
    snr_log = snr(dx_dt, pred)
    psnr_log = psnr(dx_dt, pred)

    if on_wer:  # Ele demora muito pra fazer a validacao
        audio_pred = mel_for_audio(torch.Tensor(pred).to('cuda')).cpu().detach()
        wer_log = wer_with_trans(sentence, audio_pred, sr_y=sr_hifigan())
    else:
        wer_log = -1

    return mcd_log, snr_log, psnr_log, wer_log

def change_shape(mel):
    return mel.squeeze(0).squeeze(0).cpu().detach().numpy()