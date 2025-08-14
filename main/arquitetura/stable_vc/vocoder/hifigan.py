from speechbrain.inference.vocoders import HIFIGAN


def vocoder(spectrogram, model_type_rate: str = "22050"):
    if model_type_rate == "22050":
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz",
                                        savedir="pretrained_models/tts-hifigan-libritts-22050Hz")
    elif model_type_rate == "16000":
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                        savedir="pretrained_models/tts-hifigan-libritts-16kHz")
    else:
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                        savedir="pretrained_models/tts-hifigan-ljspeech")

    return hifi_gan.decode_batch(spectrogram)
