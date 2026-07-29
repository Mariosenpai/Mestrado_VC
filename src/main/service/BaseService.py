from pathlib import Path

import torch
import wandb

from src.common.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
from src.main.interface.AASVCInterface import AASVCTrainerInterface
from src.main.parameters.AASVCParameters import AASVCParameters


class BaseService:


    def __init__(self, batch_size,path_dataset, collete_fn):

        wandb.login(key="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")
        self.root  = Path(__file__).resolve().parent.parent.parent
        self.model = None
        self.collate_fn = collete_fn
        self.vocoder = None

        self.data = self._define_dataloader(batch_size=batch_size, path_dataset=path_dataset)
        self.train_loader = self.data[0]
        self.val_loader = self.data[1]

    def trainer(self, path_model_checkpoint, path_model_params, epochs, name_experiment, is_test):
        print("Instanciando o modelo ...")
        model = self._define_model(model=self.model ,device=torch.device("cuda"), path_model_params=path_model_params)
        trainer = self._define_trainer(model, self.data, epochs, name_experiment, is_test=is_test)

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            print("Carregando o modelo pre-treinado ...")
            trainer.load_checkpoint(path_model_checkpoint)

        print("Iniciando o treinamento:")
        trainer.run_wandb(name_experiment, config={"epochs": 5})

    def _define_trainer(self, model, data, epochs, name_experiment, is_test):
        pass

    def _define_model(self,
                      model: torch.nn.Module,
                      path_model_params: str,
                      device=torch.device("cpu"),
                      ) -> torch.nn.Module:
        """
        """
        if path_model_params is not None:
            yaml_model = path_model_params
            model = model(yaml_model, device)

        return model.to(device)


    def _define_dataloader(
            self,
            batch_size: int = 2,
            path_dataset: str = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_set = CVMPT_offline(path=path_dataset / "treinamento")
        valid_set = CVMPT_offline(path=path_dataset / "teste")

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

