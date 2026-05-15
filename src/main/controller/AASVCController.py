from src.main.service.AASVCService import AASVCService


class AASVCController:

    def __init__(self, batch_size, path_dataset):
        self.service = AASVCService(batch_size=batch_size, path_dataset=path_dataset)

    def trainer(self, path_model_checkpoint, path_model_params, epochs, name_experiment, is_test):
        self.service.trainer(
            path_model_checkpoint,
            path_model_params,
            epochs,
            name_experiment,
            is_test
        )

    def inference(self, path_checkpoint, path_dataset):
        audio, mel = self.service.inference(path_checkpoint, path_dataset)

        return audio, mel


if __name__ == "__main__":
    path_dataset = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    path_model_checkpoint = "/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/AASVC_JP/checkpoint-50000steps.pkl"
    path_model_params = "/home/mario/Mestrado_VC/src/config/AASVC_ENG/aas_vc.melmelmel.v1.yaml"
    epochs = 5
    name_experiment = "aasvc-mel-pytorch"
    is_test = False

    controller = AASVCController(batch_size=2, path_dataset=path_dataset)

    #print(controller.inference(path_model_checkpoint, path_dataset))

    controller.trainer(
        path_model_checkpoint,
        path_model_params,
        epochs,
        name_experiment,
        is_test
    )
