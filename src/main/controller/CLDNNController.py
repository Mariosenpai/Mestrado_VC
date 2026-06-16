from pathlib import Path

from src.main.controller.BaseController import BaseController
from src.main.service.CLDNNService import CLDNNService


class CLDNNController(BaseController):

    def __init__(self,batch_size, path_dataset,path_checkpoint, path_model_params):
        super().__init__(batch_size, path_dataset,path_checkpoint, path_model_params)
        self.service = CLDNNService(batch_size=batch_size, path_dataset=path_dataset)

if __name__ == "__main__":

    ROOT = Path.cwd().resolve().parent.parent.parent
    path_dataset = ROOT / "dataset" / "cv-corpus-mozilla-pt" / "data"
    path_model_checkpoint = None # ROOT / "src" / "config" / "modelCheckpoint" / "AASVC" / "checkpoint-100000steps.pkl"
    path_model_params = None #ROOT / "src" / "config" / "yaml" / "cldnn_vc.mel.v1.yaml"
    epochs = 5
    name_experiment = "cldnn-mel-pytorch"
    is_test = True

    controller = CLDNNController(
        batch_size=2,
        path_dataset=path_dataset,
        path_checkpoint=path_model_checkpoint,
        path_model_params=path_model_params
    )

    controller.trainer(epochs,name_experiment, is_test)