from src.common.vocoder.HiFiGAN import HiFiGAN
from src.main.collaters.CLDNNCollater import CLDNNCollater
from src.main.collaters.nar_vc import NARVCCollater
from src.main.interface.CLDNNInterface import CLDNNInterface
from src.main.model.CLDNN import CLDNN
from src.main.parameters.CLDNNParameters import CLDNNParameters

from src.main.service.BaseService import BaseService


class CLDNNService(BaseService):

    def __init__(self, batch_size, path_dataset: str = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"):
        BaseService.__init__(self,batch_size, path_dataset, collete_fn= CLDNNCollater())
        self.vocoder = HiFiGAN()
        self.model = CLDNN(mod="base")

    def set_mod_model(self, mod):
        self.model = CLDNN(mod=mod)

    def _define_trainer(self, model, data, epochs, name_experiment, is_test: bool = False) -> CLDNNInterface:
        parameters = CLDNNParameters(model, data, epochs, name_experiment)

        return CLDNNInterface(
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
