from pathlib import Path

import torch
from argon2 import Parameters


class CLDNNParameters:

    def __init__(self, model, data, epochs, name_experiment, vocoder=None):

        self.model = model
        self.steps= 0
        self.device = "cuda"
        self.sampler = {}
        self.data = data
        self.epochs = epochs
        self.name_experiment = name_experiment
        self.vocoder = vocoder

        self.data_loader = {
            "train": data[0],
            "dev": data[1],
        }

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        self.criterion = {
            "mse": torch.nn.MSELoss(),
        }
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=4000,
            gamma=1.0
        )
        inference_args = {
            "threshold": 0.5,
            "minlenratio": 6.0,
            "maxlenratio": 0.0,
        }
        ROOT = Path.cwd().resolve().parent.parent.parent
        self.config = {
            "outdir": ROOT / "experiments"/ f"{name_experiment}",
            "train_max_steps": 10000 * epochs,
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