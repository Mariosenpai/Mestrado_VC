from pathlib import Path

from src.main.controller.BaseController import BaseController
from src.main.service.CLDNNService import CLDNNService


class CLDNNController(BaseController):

    def __init__(self,batch_size, path_dataset,path_checkpoint, path_model_params):
        super().__init__(batch_size, path_dataset,path_checkpoint, path_model_params)
        self.service = CLDNNService(batch_size=batch_size, path_dataset=path_dataset)

    def set_mod(self, mods:list, device="cuda"):
        '''
            Lista de mods
                - transformer
                - flow_matching
        :param mods:
        :return:
        '''

        dic_mods = self.service.define_mods(mods)
        self.service.set_cldnn_mods(dic_mods, device)
    def inference(self, n_steps:int, id_audio_dataset:int,device="cuda") -> tuple:

        audio_pred, mel_pred, mel_orig, mel_noise= self.service.inference(self.path_checkpoint, n_steps,id_audio_dataset,device=device)
        return audio_pred, mel_pred, mel_orig,mel_noise


if __name__ == "__main__":

    ROOT = Path.cwd().resolve().parent.parent.parent
    path_dataset = ROOT / "dataset" / "cv-corpus-mozilla-pt" / "data"
    path_model_checkpoint = None # ROOT / "src" / "config" / "modelCheckpoint" / "CLDNN" / "flow-matching-checkpoint-50000steps.pkl"
    path_model_params = None # ROOT / "src" / "config" / "yaml" / "cldnn_vc.mel.v1.yaml"
    epochs = 6
    list_mods = []
    is_test = False

    name_experiment = "cldnn-mod"
    mod_name = ""
    if len(list_mods) == 0:
        name_experiment = "cldnn-mod-none"
    else:
        for mod in list_mods:
            mod_name = mod_name +"-"+mod
        name_experiment += mod_name


    controller = CLDNNController(
        batch_size=2,
        path_dataset=path_dataset,
        path_checkpoint=path_model_checkpoint,
        path_model_params=path_model_params
    )
    #controller.set_mod(list_mods)
    controller.trainer(epochs,name_experiment, is_test)