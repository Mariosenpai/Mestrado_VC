import torch
from main.data_loader.CVMPT import cv_mozilla_pt_dataset_com_ruido


def criar_dataloader(dataset, batch_size, collate_fn=None):
    data_generator = torch.Generator(device='cpu')
    data_generator.manual_seed(13)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=data_generator,
        collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    # Cada item em `batch` é uma tupla (audio_natural, audio_eletronico)
    # Retorne como uma lista de tuplas
    return [(audio_natural, audio_eletronico) for audio_natural, audio_eletronico in batch]
