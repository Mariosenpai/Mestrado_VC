from src.common.vocoder.HiFiGAN import HiFiGAN
from src.main.collaters.nar_vc import NARVCCollater
from src.main.interface.FastSpeechInterface import FastSpeechInterface
from src.main.model.FastSpeech import seq2seq_FastSpeech
from src.main.parameters.FastSpeechParameters import FastSpeechParameters
from src.main.service.BaseService import BaseService


class FastSpeechService(BaseService):

    def __init__(self, batch_size, path_dataset):
        super().__init__(batch_size, path_dataset, collete_fn= NARVCCollater())

        self.vocoder = HiFiGAN()
        self.collate_fn = NARVCCollater()
        self.model = seq2seq_FastSpeech

    def _define_trainer(self, model, data, epochs, name_experiment, is_test: bool = False) -> FastSpeechInterface:
        parameters = FastSpeechParameters(model, data, epochs, name_experiment)

        return FastSpeechInterface(
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