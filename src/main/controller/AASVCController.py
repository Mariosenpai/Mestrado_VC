from pathlib import Path

from src.main.controller.BaseController import BaseController
from src.main.service.AASVCService import AASVCService


class AASVCController(BaseController):

    def __init__(self, batch_size, path_dataset,path_checkpoint, path_model_params):
        super().__init__(batch_size, path_dataset,path_checkpoint, path_model_params)
        self.service = AASVCService(batch_size=batch_size, path_dataset=path_dataset)

    def all_mels(self):
        return self.service.all_mels(self.path_checkpoint,self.path_model_params)

    def generate_wav_all_dataset_val(self):
        self.service.generate_wav_all_dataset_val(self.path_checkpoint, self.path_model_params)

    def inference(self):
        audio, mel = self.service.inference(self.path_checkpoint, self.path_dataset)

        return audio, mel


if __name__ == "__main__":

    ROOT = Path.cwd().resolve().parent.parent.parent
    path_dataset = ROOT / "dataset" / "cv-corpus-mozilla-pt" / "data"
    path_model_checkpoint = ROOT / "src" / "config" / "modelCheckpoint" / "AASVC" / "checkpoint-100000steps.pkl"
    path_model_params = ROOT / "src" / "config" / "yaml" / "aas_vc.melmelmel.v1.yaml"
    epochs = 5
    name_experiment = "aasvc-mel-pytorch"
    is_test = False

    controller = AASVCController(
        batch_size=2,
        path_dataset=path_dataset,
        path_checkpoint=path_model_checkpoint,
        path_model_params=path_model_params
    )

    controller.trainer(epochs, name_experiment, is_test)
    # controller.generate_wav_all_dataset_val()

    #print(controller.inference(path_model_checkpoint, path_dataset))

    # controller.traine r(
    #     path_model_checkpoint,
    #     path_model_params,
    #     epochs,
    #     name_experiment,
    #     is_test
    # )
