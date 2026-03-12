import wandb
from tqdm import tqdm

from main.arquitetura.Seq2SeqVC.deprected.collate import format_batch
from main.arquitetura.Seq2SeqVC.deprected.validation import one_validation
from main.pre_processamento.spectograma import mel_to_rgb
from main.vocoder.HiFiGAN import sr_hifigan


def one_epoch(model, device, train_loader, valid_loader, optimizer, criterion, ep, is_test=False):
    print("-------------------------------------------")
    print("Treinamento")
    loss = one_train(model, device, train_loader, optimizer, criterion, is_test)
    print("-------------------------------------------")
    print("Validacao")
    relatorio = one_validation(model, device, valid_loader, criterion, is_test)
    # boa tarde amigo tudo bo
    # não
    # qq ta pegano
    # passei 2 horas ouvindo uma vei falando
    # e n deu uma cadeirada nela pq?
    # professora de vanessa
    # ih rapa n pode flw kkkk
    return {
        "MDC": relatorio.mdc,
        "WER": 0,
        "SNR": relatorio.snr,
        "PSNR": relatorio.psnr,
        "loss_train": loss,
        "loss_val": relatorio.loss,
        "mel_exemple": {
            "ground_truth": wandb.Image(mel_to_rgb(relatorio.grouth_truth)),
            "prediction": wandb.Image(mel_to_rgb(relatorio.pred))
        },
        "audio_exemple": {
            "ground_truth": wandb.Audio(relatorio.audio, sample_rate=int(relatorio.sr)),
            "noise": wandb.Audio(relatorio.audio_noise, sample_rate=int(relatorio.sr)),
            "prediction": wandb.Audio(relatorio.audio_pred, sample_rate=sr_hifigan()),
        },
        "epocas": ep + 1
    }


def one_train(model, device, train_loader, optimizer, criterion, is_test=False):
    total_loss = 0
    total_loss_bce = 0
    i = 0

    model.train()
    pbar = tqdm(train_loader, desc="Treinamento", total=len(train_loader))

    for i, batch in enumerate(pbar):

        xs = batch["xs"].to(device)
        ilens = batch["ilens"].to(device)

        ys = batch["ys"].to(device)
        olens = batch["olens"].to(device)

        dp_inputs = batch["dp_inputs"].to(device)
        dplens = batch["dplens"].to(device)

        spembs = batch["spembs"]  # normalmente None

        ret = model(
            xs,
            ilens,
            ys,
            olens,
            dp_inputs,
            dplens,
            spembs
        )

        optimizer.zero_grad()

        # outputs do decoder
        after_outs = ret["after_outs"]
        before_outs = ret["before_outs"]

        # ground truth ajustado
        ys_gt = ret["ys"]
        olens_gt = ret["olens"]

        # perdas principais (L1/MSE)
        l1_loss = criterion(
            after_outs,
            before_outs,
            ys_gt,
            olens_gt
        )

        loss = l1_loss + ret["bin_loss"]

        # perda binária de alinhamento (Viterbi)
        bin_loss = ret["bin_loss"]


        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (i + 1))

        if is_test:
            break
    return total_loss / (i + 1)
