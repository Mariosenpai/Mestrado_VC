from pathlib import Path

from src.main.controller.BaseController import BaseController
from src.main.service.FastSpeechService import FastSpeechService


class FastSpeechController(BaseController):


    def __init__(self, batch_size, path_dataset):
        super().__init__()
        self.service = FastSpeechService(batch_size=batch_size, path_dataset=path_dataset)

if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent.parent.parent

    path_dataset = ROOT / "dataset" / "cv-corpus-mozilla-pt" / "data"
    path_model_checkpoint = ROOT / "src" / "config" / "modelCheckpoint"/ "FastSpeech" / "checkpoint-0steps.pkl"
    path_model_params = ROOT / "src" / "config" / "yaml" / "fs2_vc.melmelmel.v1.yaml"
    epochs = 5
    name_experiment = "fastspeech-teste"
    is_test = False

    print(path_dataset)

    controller = FastSpeechController(batch_size=2, path_dataset=path_dataset)

    # print(controller.inference(path_model_checkpoint, path_dataset))

    controller.trainer(
        path_model_checkpoint,
        path_model_params,
        epochs,
        name_experiment,
        is_test
    )