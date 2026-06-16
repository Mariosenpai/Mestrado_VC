import os
import re
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
import soundfile as sf

from src.common.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
from src.main.interface.AASVCInterface import AASVCTrainerInterface
from src.main.collaters.nar_vc import NARVCCollater
from src.main.model.DuractionInputEncoder import DPInputEncoder
from src.main.model.AASVC import seq2seq_AASVC
from src.main.parameters.AASVCParameters import AASVCParameters
from src.common.vocoder.HiFiGAN import HiFiGAN
from src.main.service.BaseService import BaseService


class AASVCService(BaseService):
    def __init__(self, batch_size, path_dataset: str = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"):
        super().__init__(batch_size, path_dataset, collete_fn= NARVCCollater())

        self.vocoder = HiFiGAN()
        self.model = seq2seq_AASVC

    def _define_interface(self, model, data, vocoder):
        name_experiment = "inferance"
        epochs = 1
        parameters = AASVCParameters(model, data, epochs, name_experiment, vocoder)

        return AASVCTrainerInterface(
            steps=parameters.steps,
            epochs=parameters.epochs,
            data_loader=parameters.data_loader,
            sampler=parameters.sampler,
            model=parameters.model,
            vocoder=parameters.vocoder,
            criterion=parameters.criterion,
            optimizer=parameters.optimizer,
            scheduler=parameters.scheduler,
            config=parameters.config,
            is_test=False,
            device=parameters.device,
        )

    def _duraction_input_inference(self, data) -> list[np.array]:
        dp_model = DPInputEncoder()
        dp_inputs = []
        dp_inputs.append(dp_model(torch.Tensor(data["mel_noise"]).transpose(0, 1)).detach().squeeze(0).numpy())
        return dp_inputs

    def _define_trainer(self, model, data, epochs, name_experiment, is_test: bool = False) -> AASVCTrainerInterface:
        parameters = AASVCParameters(model, data, epochs, name_experiment)

        return AASVCTrainerInterface(
            steps=parameters.steps,
            epochs=parameters.epochs,
            data_loader=parameters.data_loader,
            sampler=parameters.sampler,
            model=parameters.model,
            vocoder=self.vocoder,
            criterion=parameters.criterion,
            optimizer=parameters.optimizer,
            scheduler=parameters.scheduler,
            config=parameters.config,
            is_test=is_test,
            device=parameters.device,
        )

    def _save_wav(
            self,
            audio: torch.Tensor,
            sample_rate=22050,
            folder="audios",
            file_name=None
    ):
        """
        Salva um áudio em formato .wav

        Parameters
        ----------
        audio : torch.Tensor
            Array contendo o áudio

        sample_rate : int
            Taxa de amostragem

        folder : str
            Pasta onde o áudio será salvo

        file_name : str
            Nome do arquivo (opcional)
        """
        audio = torch.Tensor(audio).squeeze(0).squeeze(0).detach().cpu().numpy()
        # Cria a pasta caso não exista
        os.makedirs(folder, exist_ok=True)


        # Gera nome automático
        if file_name is None:
            file_name = f"audio.wav"


        # Caminho completo
        file_path = os.path.join(folder, file_name)

        # Verifica se já existe
        if os.path.exists(file_path):
            print(f"Arquivo já existe: {file_path}")
            return file_path

        # Salva o áudio
        sf.write(file_path, audio, sample_rate)

        print(f"Áudio salvo em: {file_path}")

        return file_path


    def generate_wav_all_dataset_val(
            self,
            path_model_checkpoint,
            path_model_params,
            output_path: str=r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\inference\AASVC\audios_gerados"
    ):
        model = self._define_model(model=self.model, path_model_params=path_model_params,device=torch.device("cpu"))

        inferance = self._define_interface(model, self.data, self.vocoder)

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            inferance.load_checkpoint(path_model_checkpoint)

        dataset_val = self.val_loader.dataset

        for i, batch in enumerate(tqdm(dataset_val)):
            data = dataset_val[i]
            data["dp_inputs"] = self._duraction_input_inference(data)

            audio, _ = inferance.inference(data)
            sentence = re.sub(r'[^\w\-.]', '_', data["sentence"][:10])

            self._save_wav(
                audio=audio,
                sample_rate=self.vocoder.sr_vocoder(),
                folder=output_path,
                file_name=f"audio_{data["client_id"][:10]}_{sentence}.wav"
            )



    def inference(
            self,
            path_model_checkpoint: str,
            output_path: str = r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\inference\AASVC"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna um audios aleatorio do dataset de validação

        :param path_model_checkpoint:
        :param output_path:
        :return: audio (T,) , mel (Frequencia, Time)
        """

        model = self._define_model(model=self.model, device=torch.device("cpu"))

        inferance = self._define_interface(model, self.data, self.vocoder)

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            inferance.load_checkpoint(path_model_checkpoint)

        data = self.val_loader.dataset[0]
        data["dp_inputs"] = self._duraction_input_inference(data)

        audio, mel = inferance.inference(data)

        return audio, mel

    def all_mels(self,path_model_checkpoint,path_model_params) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor]:
        """

        :param path_model_checkpoint:
        :param path_model_params:
        :return: mel_inference (Freq, Time), mel_original (Freq, Time) , mel_ruido (Freq, Time)
        """
        model = self._define_model(self.model,device=torch.device("cpu"),path_model_params=path_model_params)

        interface = self._define_interface(model, self.data, self.vocoder)
        if path_model_checkpoint is not None or path_model_checkpoint == "":
            interface.load_checkpoint(path_model_checkpoint)

        data = self.val_loader.dataset[0]
        mel_original = data["mel"]
        mel_noise = data["mel_noise"]

        data["dp_inputs"] = self._duraction_input_inference(data)

        _ , mel_inference = interface.inference(data)

        return mel_inference[0], torch.Tensor(mel_original), torch.Tensor(mel_noise)