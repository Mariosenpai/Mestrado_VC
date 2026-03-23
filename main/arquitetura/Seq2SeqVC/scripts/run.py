import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from bibliotecas_externas.seq2seqvc.seq2seq_vc.losses import Seq2SeqLoss, DurationPredictorLoss, \
    StochasticDurationPredictorLoss

from main.arquitetura.Seq2SeqVC.models.model import seq2seq_AASVC
from main.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
import torch

import wandb
wandb.login(key="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")

yaml_model = r"/home/mario/Mestrado_VC/main/arquitetura/Seq2SeqVC/configs/AASVC_ENG/aas_vc.melmelmel.v1.yaml"
checkpoint_model_path = r"/home/mario/Mestrado_VC/main/arquitetura/Seq2SeqVC/configs/AASVC_JP/checkpoint-50000steps.pkl"

dataset_train_path = r"/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/treinamento_gp"
dataset_val_path = r"/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/teste_gp"

from main.arquitetura.Seq2SeqVC.datasets.collaters.nar_vc import NARVCCollater

device = "cuda" if torch.cuda.is_available() else "cpu"
model_seq2seq = seq2seq_AASVC(yaml_model,device)



l1_loss = torch.nn.L1Loss()
bce_loss = torch.nn.BCEWithLogitsLoss()


train_set = CVMPT_offline(path=dataset_train_path)
valid_set = CVMPT_offline(path=dataset_val_path)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
    collate_fn=NARVCCollater()
)

valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=1,
    shuffle=False,
    collate_fn=NARVCCollater()
)

from bibliotecas_externas.seq2seqvc.seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform

vocoder = Spectrogram2Waveform(
    n_fft=1024,
    n_shift=256,
    fs=22050,
    n_mels=80,
    griffin_lim_iters=32,
    take_norm_feat=False,
    #stats=trg_stats,  # stats do target
)

from main.arquitetura.Seq2SeqVC.trainers.TrainerAASVCMod import Trainer
from bibliotecas_externas.seq2seqvc.seq2seq_vc.losses import L1Loss, ForwardSumLoss

# bibliotecas_externas.seq2seqvc.seq2seq_vc.trainers import ARVCTrainer, AASVCTrainer

# contadores
steps = 0
epochs = 0

# dataloaders
data_loader = {
    "train": train_loader,
    "dev": valid_loader,
}

# sampler (sem DDP)
sampler = {}

# modelo
model = model_seq2seq.to(device)

# loss
criterion = {
    "L1Loss": L1Loss(),
    "ForwardSumLoss": ForwardSumLoss(),
    "StochasticDurationPredictorLoss": StochasticDurationPredictorLoss(),
}

optimizer = torch.optim.Adam(
    model_seq2seq.parameters(),
    lr=1e-5,
    betas=(0.9, 0.98),
    eps=1e-9
)

# scheduler (opcional)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=4000,
    gamma=1.0
)

inference_args = {
    "threshold": 0.5,
    "minlenratio": 6.0,
    "maxlenratio": 0.0,
}

# config
config = {
    "outdir": "./experiments/aas_vc_3_100k",
    "train_max_steps": 100000,
    "log_interval_steps": 10,
    "eval_interval_steps": 10000,
    "save_interval_steps": 10000,
    "distributed": False,
    "rank": 0,
    "gradient_accumulate_steps": 1,
    "grad_norm": 0,
    "num_save_intermediate_results": 5,
    "inference": inference_args,
    "criterions": ["L1Loss", "ForwardSumLoss", "StochasticDurationPredictorLoss"],
    "lambda_align": 2.0,
    "dp_train_start_steps": 0

}

# trainer
trainer = Trainer(
    steps=steps,
    epochs=epochs,
    data_loader=data_loader,
    sampler=sampler,
    model=model,
    vocoder=vocoder,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config,
    is_test=False,
    device=device,
)

trainer.load_checkpoint(checkpoint_model_path)

project="Laringe Eletronica Seq2Seq - Mestrado VC"

config = {
    "epocas": 100
}
trainer.run_wandb(project,config)