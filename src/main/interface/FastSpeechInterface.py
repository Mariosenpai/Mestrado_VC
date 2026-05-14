#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2023 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os

import numpy as np
import soundfile as sf
import time
import torch
import wandb
from tqdm.auto import tqdm

from moduleExternal.seq2seqvc.seq2seq_vc.trainers.base import Trainer

# set to avoid matplotlib error in CLI environment
import matplotlib

from moduleExternal.seq2seqvc.seq2seq_vc.utils.model_io import filter_modules, get_partial_state_dict, \
    transfer_verification, print_new_keys

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class NARVCTrainer(Trainer):

    def __init__(self, steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,
                 device=torch.device("cpu"), is_test: bool = False):
        super().__init__(steps, epochs, data_loader, sampler, model, vocoder, criterion, optimizer, scheduler, config,
                         is_test,
                         device)

    """Customized trainer module for non-autoregressive VC training."""

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

    def load_trained_modules(self, checkpoint_path, init_mods):
        if self.config["distributed"]:
            main_state_dict = self.model.module.state_dict()
        else:
            main_state_dict = self.model.state_dict()

        if os.path.isfile(checkpoint_path):
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

            # first make sure that all modules in `init_mods` are in `checkpoint_path`
            modules = filter_modules(model_state_dict, init_mods)

            # then, actually get the partial state_dict
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)

            if partial_state_dict:
                if transfer_verification(main_state_dict, partial_state_dict, modules):
                    print_new_keys(partial_state_dict, modules, checkpoint_path)
                    main_state_dict.update(partial_state_dict)
        else:
            logging.error(f"Specified model was not found: {checkpoint_path}")
            exit(1)

        if self.config["distributed"]:
            self.model.module.load_state_dict(main_state_dict)
        else:
            self.model.load_state_dict(main_state_dict)


        # =========================
        # 🔹 Forward padrão
        # =========================

    def forward_step(self, batch):
        xs = batch["xs"]
        ys = batch["ys"]
        ilens = batch["ilens"]
        olens = batch["olens"]
        durations = batch["durations"]
        duration_lens = batch["duration_lens"]
        dp_inputs = batch["dp_inputs"]
        dplens = batch["dplens"]

        # print(xs.shape)
        # print(xs)

        print(durations)
        print(durations.shape)

        return self.model(
            xs, ilens, ys, olens, durations=durations, durations_lengths=duration_lens, dp_inputs=dp_inputs, dp_lengths=dplens
        )

    # =========================
    # 🔹 Loss modular
    # =========================

    def compute_loss(self, outputs, batch):
        before_outs, after_outs, d_outs, ilens_, olens_, ys_ = outputs

        losses = {}

        # L1
        l1_loss = self.criterion["L1Loss"](after_outs, before_outs, ys_, olens_)
        losses["l1"] = l1_loss

        # Duration
        duration_loss = self.criterion["DurationPredictorLoss"](
            d_outs, batch["durations"], ilens_
        )
        losses["duration"] = duration_loss

        total_loss = sum(losses.values())

        return total_loss, losses

        # =========================
        # 🔹 Train step padronizado
        # =========================

    def _train_step(self, batch):

        batch["durations"] = torch.tensor([10, 10]).unsqueeze(0)
        batch["duration_lens"] = torch.tensor([10, 10]).unsqueeze(0)
        print(batch["durations"].shape)
        durations = batch["durations"].to(self.device)

        (before_outs, after_outs, d_outs, ilens_, olens_, ys_,) = self.forward_step(batch)

        l1_loss = self.criterion["L1Loss"](after_outs, before_outs, ys_, olens_)
        # duration prediction loss
        duration_loss = self.criterion["DurationPredictorLoss"](
            d_outs, durations, ilens_
        )

        gen_loss = l1_loss + duration_loss
        self.total_train_loss["train/l1_loss"] += l1_loss.item()
        self.total_train_loss["train/duration_loss"] += duration_loss.item()

        self.total_train_loss["train/loss"] += gen_loss.item()

        self.optimizer.zero_grad()
        gen_loss.backward()
        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def inference_step(self, batch, idx):
        """Padroniza a inferência para qualquer modelo"""

        x = batch["xs"][idx]
        dp_input = batch.get("dp_inputs", None)

        if dp_input is not None:
            dp_input = dp_input[idx]

        # chamada padrão
        outputs = self.model.inference(
            x,
            dp_input=dp_input,
            spembs=None
        )

        # garantir formato consistente
        if isinstance(outputs, tuple):
            mel = outputs[0]
            extra = outputs[1:]
        else:
            mel = outputs
            extra = None

        return {
            "mel": mel,
            "extra": extra
        }

    def save_results(self, mel, y_ref, idx, dirname, olen):
        import os
        import soundfile as sf
        import matplotlib.pyplot as plt

        # ===== plot =====
        plt.figure(figsize=(6, 4), dpi=150)
        plt.imshow(mel.cpu().numpy().T, aspect="auto", origin="lower")
        plt.xlabel("Time")
        plt.ylabel("Mel")
        plt.tight_layout()

        os.makedirs(os.path.join(dirname, "outs"), exist_ok=True)
        plt.savefig(os.path.join(dirname, f"outs/{idx}_out.png"))
        plt.close()

        # ===== vocoder =====
        if self.vocoder is not None:
            os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)

            y, sr = self.vocoder.decode(mel)

            sf.write(
                os.path.join(dirname, f"wav/{idx}_gen.wav"),
                y.cpu().numpy(),
                sr,
                "PCM_16",
            )

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):

        batch = {
            k: v.to(self.device) if v is not None else None
            for k, v in batch.items()
        }

        dirname = os.path.join(
            self.config["outdir"],
            f"predictions/{self.steps}steps"
        )
        os.makedirs(dirname, exist_ok=True)

        for idx in range(len(batch["xs"])):

            start_time = time.time()

            result = self.inference_step(batch, idx)

            mel = result["mel"]

            elapsed = time.time() - start_time
            logging.info(
                f"Inference speed: {int(mel.size(0)) / elapsed:.1f} frames/sec"
            )

            y_ref = batch["ys"][idx]
            olen = batch["olens"][idx]

            self.save_results(
                mel,
                y_ref,
                idx,
                dirname,
                olen
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break