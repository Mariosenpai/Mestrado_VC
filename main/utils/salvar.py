import os
import datetime

import torch


def save_model(model, path,info=""):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/generator_' + str(info) + '.pt'
    torch.save(model.state_dict(), filename)


def salva_e_exibir_info(model,filename, info):
    # Escrever informações em um arquivo
    save_model(model, "/kaggle/working/model/")
    file = open(filename, 'a')
    file.write(info)
    file.flush()
    file.close()

    print(info)

