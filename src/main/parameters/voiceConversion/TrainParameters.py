import torch

from moduleExternal.seq2seqvc.seq2seq_vc.vocoder.griffin_lim import Spectrogram2Waveform
from moduleExternal.seq2seqvc.seq2seq_vc.losses import L1Loss, ForwardSumLoss
from moduleExternal.seq2seqvc.seq2seq_vc.losses import StochasticDurationPredictorLoss

class Train_parameters:

    def __init__(self, model, data, epochs, name_experiment, vocoder = None):

        self.model = model
        self.steps = 0
        self.epochs = epochs

        self.device = "cuda"
        self.sampler = {}

        if vocoder is None:
            self.vocoder = Spectrogram2Waveform(
                n_fft=1024,
                n_shift=256,
                fs=22050,
                n_mels=80,
                griffin_lim_iters=32,
                take_norm_feat=False,
                # stats=trg_stats,  # stats do target
            )
        else:
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

        # scheduler (opcional)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=4000,
            gamma=1.0
        )

        self.criterion = {
            "L1Loss": L1Loss(),
            "ForwardSumLoss": ForwardSumLoss(),
            "StochasticDurationPredictorLoss": StochasticDurationPredictorLoss(),
        }

        inference_args = {
            "threshold": 0.5,
            "minlenratio": 6.0,
            "maxlenratio": 0.0,
        }
        self.config = {
            "outdir": f"/home/mario/Mestrado_VC/experiments/{name_experiment}",
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

