import torch

from src.common.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
from src.main.model.voiceConversion.Seq2SeqVC.datasets.collaters.nar_vc import NARVCCollater
from src.main.model.voiceConversion.Seq2SeqVC.models.model import seq2seq_AASVC, seq2seq_FastSpeech
from src.main.model.voiceConversion.Seq2SeqVC.trainers import TrainerAASVCMod
from src.main.parameters.voiceConversion.TrainParameters import Train_parameters


def define_dataloader(
        collate_fn,
        batch_size: int=2,
        path_dataset:str="/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    train_set = CVMPT_offline(path=path_dataset+"treinamento")
    valid_set = CVMPT_offline(path=path_dataset+"teste")

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader

def define_model(name_model:str="AASVC",device=torch.device("cpu")):
    """
    Name_model: AASVC, FASTSPEECH
    """

    if name_model == "AASVC":
        yaml_model = r"/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/AASVC_ENG/aas_vc.melmelmel.v1.yaml"
        model_seq2seq = seq2seq_AASVC(yaml_model,device)

        return model_seq2seq.to(device)
    elif name_model == "FASTSPEECH":
        yaml_model = r"/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/FastSpeechVC/fs2_vc.melmelmel.v1.yaml"
        model_seq2seq = seq2seq_FastSpeech(yaml_model,device)

        return model_seq2seq.to(device)

    else:
        return None



def define_trainer(model, data, epochs, name_experiment, is_test:bool=False):

    parameters = Train_parameters(model,data, epochs, name_experiment)

    return TrainerAASVCMod.Trainer(
        steps=parameters.steps,
        epochs=parameters.epochs,
        data_loader=parameters.data_loader,
        sampler=parameters.sampler,
        model=parameters.model,
        vocoder=parameters.vocoder,
        criterion=parameters.criterion,
        optimizer=parameters.optimizer,
        scheduler=parameters.scheduler,
        config=parameters.config,
        is_test=is_test,
        device=parameters.device,
    )

def controller(path_dataset, path_model_checkpoint,batch_size, epochs, model_type,name_experiment, is_test ):

    data = define_dataloader(path_dataset=path_dataset,collate_fn=NARVCCollater(), batch_size=batch_size)
    model = define_model(model_type)
    trainer = define_trainer(model, data, epochs, name_experiment, is_test=is_test)

    if path_model_checkpoint is not None or path_model_checkpoint == "":
        trainer.load_checkpoint(path_model_checkpoint)

    trainer.run_wandb(name_experiment, config={"epochs":5})



if __name__ == '__main__':

    path_dataset = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    path_model_checkpoint = "/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/FastSpeechVC/checkpoint-0steps.pkl"
    batch_size = 2
    epochs = 5
    model_type = "FASTSPEECH"
    name_experiment = "fastspeech_1"
    is_test = True

    controller(
        path_dataset,
        path_model_checkpoint,
        batch_size,
        epochs,
        model_type,
        name_experiment,
        is_test
    )





