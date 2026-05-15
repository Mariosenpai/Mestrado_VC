import numpy as np
import torch
import wandb

from src.common.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
from src.main.interface.AASVCInterface import AASVCTrainerInterface
from src.main.collaters.nar_vc import NARVCCollater
from src.main.model.DuractionInputEncoder import DPInputEncoder
from src.main.model.AASVC import seq2seq_AASVC
from src.main.parameters.voiceConversion.TrainParameters import Train_parameters
from src.main.vocoder.HiFiGAN import HiFiGAN
from src.main.vocoder.VocoderBase import VocoderBase


class AASVCService:
    def __init__(self, batch_size, path_dataset: str = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"):

        wandb.login(key="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")

        self.vocoder = HiFiGAN()
        self.collate_fn = NARVCCollater()

        self.data = self._define_dataloader(batch_size=batch_size, path_dataset=path_dataset)
        self.train_loader = self.data[0]
        self.val_loader = self.data[1]

    def _define_dataloader(
            self,
            batch_size: int = 2,
            path_dataset: str = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

        train_set = CVMPT_offline(path=path_dataset + "treinamento")
        valid_set = CVMPT_offline(path=path_dataset + "teste")

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return train_loader, valid_loader

    def _define_model(self,
                      path_model_params: str,
                      device=torch.device("cpu"),
                      ) -> torch.nn.Module:
        """
        Name_model: AASVC, FASTSPEECH
        """
        yaml_model = path_model_params
        model_seq2seq = seq2seq_AASVC(yaml_model, device)

        return model_seq2seq.to(device)

    def _define_trainer(self, model, data, epochs, name_experiment, is_test: bool = False) -> AASVCTrainerInterface:
        parameters = Train_parameters(model, data, epochs, name_experiment)
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

    def _define_interface(self, model, data, vocoder):
        name_experiment = "inferance"
        epochs = 1
        parameters = Train_parameters(model, data, epochs, name_experiment, vocoder)

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

    def trainer(self, path_model_checkpoint, path_model_params, epochs, name_experiment, is_test):
        print("Instanciando o modelo ...")
        model = self._define_model(device=torch.device("cuda"), path_model_params=path_model_params)
        trainer = self._define_trainer(model, self.data, epochs, name_experiment, is_test=is_test)

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            print("Carregando o modelo pre-treinado ...")
            trainer.load_checkpoint(path_model_checkpoint)

        print("Iniciando o treinamento:")
        trainer.run_wandb(name_experiment, config={"epochs": 5})

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

        model = self._define_model(device=torch.device("cpu"))

        inferance = self._define_interface(model, self.data, self.vocoder)

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            inferance.load_checkpoint(path_model_checkpoint)

        dp_model = DPInputEncoder()
        dp_inputs = []
        data_dict = self.val_loader[0].dataset[0]

        dp_inputs.append(dp_model(torch.Tensor(data_dict["mel_noise"])).detach().squeeze(0).numpy())

        data_dict["dp_inputs"] = dp_inputs

        audio, mel = inferance.inference(data_dict, output_path)

        return audio, mel
