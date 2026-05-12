import yaml

from bibliotecas_externas.seq2seqvc.seq2seq_vc.models import AASVC


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def seq2seq_AASVC(yaml_path, device) -> AASVC:

    config = load_config(yaml_path)
    model_params = config["model_params"]

    model = AASVC(
        **model_params,  # aqui está a mágica
    ).to(device)
    return model
