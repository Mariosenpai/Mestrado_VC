import yaml

from moduleExternal.seq2seqvc.seq2seq_vc.models import FastSpeechVC


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def seq2seq_FastSpeech(yaml_path, device) -> FastSpeechVC:

    config = load_config(yaml_path)
    model_params = config["model_params"]

    model = FastSpeechVC(
        **model_params,
    ).to(device)

    return model