from random import random

import torch

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
        self.model = CLDNN()

    def define_mods(self,mods):
        dic_mods = {
            "transformer":False,
            "flow_matching": False
        }
        for mod in mods:
            if mod == "transformer":
                dic_mods["transformer"] = True
            elif mod == "flow_matching":
                dic_mods["flow_matching"] = True

        return dic_mods

    def set_cldnn_mods(self, dic_mods, device="cuda"):

        self.model = CLDNN(
            use_transformer=dic_mods["transformer"],
            use_flow_matching=dic_mods["flow_matching"],
            device=device
        )



    def _define_trainer(self, model, data, epochs, name_experiment, is_test: bool = False, device="cuda") -> CLDNNInterface:
        parameters = CLDNNParameters(model, data, epochs, name_experiment,device=device)

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

    def inference(self,path_model_checkpoint, n_steps:int=8, id_audio_dataset:int=-1 , device="cuda"):
        '''

        :param path_model_checkpoint:
        :param n_steps:
        :param id_audio_dataset:
        :return:
        '''

        model = self.model
        data = self.data
        inference = self._define_trainer(model,data=data,epochs=1,name_experiment="inference",device=device)
        device = inference.get_device()

        if path_model_checkpoint is not None or path_model_checkpoint == "":
            inference.load_checkpoint(path_model_checkpoint)

        if id_audio_dataset == -1:
            random_num = random.randint(0,len(data)-1)
        else:
            random_num = id_audio_dataset

        data = self.val_loader.dataset[random_num]
        mel = torch.Tensor(data["mel"]).transpose(0,1).unsqueeze(0).to(device)
        mel_noise = torch.Tensor(data["mel_noise"]).transpose(0,1).unsqueeze(0).to(device)

        mel_pred = inference.get_inference_type(mel_noise=mel_noise,n_steps=n_steps)
        mel_pred = mel_pred.transpose(1,2).detach().to("cuda")
        audio_pred = inference.vocoder.inference(mel_pred)

        return audio_pred, mel_pred, mel, mel_noise