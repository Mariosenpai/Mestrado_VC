import torch

from src.common.dataloader.CVMPT.CVMPT_offline import CVMPT_offline
from src.common.vocoder.WaveNet import generate_wav
from src.main.interface.AASVCInterface import AASVCTrainerInterface
from src.main.model.voiceConversion.Seq2SeqVC.datasets.collaters.nar_vc import NARVCCollater
from src.main.model.voiceConversion.Seq2SeqVC.models.EncoderDP import DPInputEncoder
from src.main.model.voiceConversion.Seq2SeqVC.models.model import seq2seq_AASVC, seq2seq_FastSpeech
from src.main.parameters.voiceConversion.TrainParameters import Train_parameters



def _define_dataloader(
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

def _define_model(device=torch.device("cpu")):
    """
    Name_model: AASVC, FASTSPEECH
    """

    yaml_model = r"/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/AASVC_ENG/aas_vc.melmelmel.v1.yaml"
    model_seq2seq = seq2seq_AASVC(yaml_model,device)

    return model_seq2seq.to(device)


def _define_trainer(model, data, epochs, name_experiment, is_test:bool=False):

    parameters = Train_parameters(model,data, epochs, name_experiment)
    return AASVCTrainerInterface(
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

def _define_interface(model,data, vocoder):

    name_experiment = "inferance"
    epochs = 1
    parameters = Train_parameters(model,data, epochs, name_experiment,vocoder )

    return AASVCTrainerInterface(
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
        is_test=False,
        device=parameters.device,
    )

def _define_vocoder():
    return generate_wav



def trainer(path_dataset, path_model_checkpoint,batch_size, epochs, name_experiment, is_test ):

    data = _define_dataloader(path_dataset=path_dataset,collate_fn=NARVCCollater(), batch_size=batch_size)
    model = _define_model()
    trainer = _define_trainer(model, data, epochs, name_experiment, is_test=is_test)

    if path_model_checkpoint is not None or path_model_checkpoint == "":
        trainer.load_checkpoint(path_model_checkpoint)

    trainer.run_wandb(name_experiment, config={"epochs":5})


def inference(path_model_checkpoint, path_dataset):

    data = _define_dataloader(path_dataset=path_dataset, collate_fn=NARVCCollater(), batch_size=1)
    model = _define_model(device=torch.device("cpu"))
    vocoder = _define_vocoder()

    inferance = _define_interface(model, data,vocoder)

    if path_model_checkpoint is not None or path_model_checkpoint == "":
        inferance.load_checkpoint(path_model_checkpoint)

    dp_model = DPInputEncoder()
    dp_inputs = []
    data_dict = data[0].dataset[0]

    dp_inputs.append(dp_model(torch.Tensor(data_dict["mel_noise"])).detach().squeeze(0).numpy())

    data_dict["dp_inputs"] = dp_inputs

    audio, mel = inferance.inference(data_dict)

    return audio, mel


if __name__ == '__main__':

    path_dataset = "/home/mario/Mestrado_VC/dataset/cv-corpus-mozilla-pt/data/"
    path_model_checkpoint = "/home/mario/Mestrado_VC/src/main/model/voiceConversion/Seq2SeqVC/configs/FastSpeechVC/checkpoint-0steps.pkl"
    batch_size = 2
    epochs = 5
    name_experiment = "fastspeech_2"
    is_test = True

    trainer(
        path_dataset,
        path_model_checkpoint,
        batch_size,
        epochs,
        name_experiment,
        is_test
    )





