from src.main.service.AASVCService import AASVCService


class AASVCController:

    def __init__(self,batch_size):
        self.service = AASVCService(batch_size=batch_size)

    def trainer(self, path_model_checkpoint, epochs, name_experiment, is_test):
        self.service.trainer(
            path_model_checkpoint,
            epochs,
            name_experiment,
            is_test
        )

    def inference(self, path_checkpoint, path_dataset):
        audio, mel = self.service.inference(path_checkpoint, path_dataset)

        return audio, mel


if __name__ == "__main__":
    controller = AASVCController(batch_size = 2)
    path_dataset = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    path_model_checkpoint = "/home/mario/Mestrado_VC/experiments/aas_vc_3_100k/checkpoint-250000steps.pkl"  #"/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/FastSpeechVC/checkpoint-0steps.pkl"
    epochs = 5
    name_experiment = "fastspeech_2"
    is_test = True

    print(controller.inference(path_model_checkpoint, path_dataset))

    # controller.trainer(
    #     path_dataset,
    #     path_model_checkpoint,
    #     batch_size,
    #     epochs,
    #     name_experiment,
    #     is_test
    # )
