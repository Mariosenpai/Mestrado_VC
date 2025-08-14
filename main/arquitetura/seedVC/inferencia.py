from argparse import Namespace

from bibliotecas_externas.seed_vc import inference


def voice_conversion_inference(
        source="./examples/source/source_s1.wav",
        target="./examples/reference/s1p1.wav",
        output=r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\output\reconstructed",
        diffusion_steps=30,
        length_adjust=1.0,
        inference_cfg_rate=0.7,
        f0_condition=False,
        auto_f0_adjust=False,
        semi_tone_shift=0,
        checkpoint=None,
        config=None,
        fp16=True,
        audio_in_code=False,
        inference_test=False,
        noise_gaussian=False
):
    '''

    :param source: pode ser um np.array ou um string com o caminho
    :param target: pode ser um np.array ou um string com o caminho
    :param output: local do output
    :param diffusion_steps:
    :param length_adjust:
    :param inference_cfg_rate:
    :param f0_condition:
    :param auto_f0_adjust:
    :param semi_tone_shift:
    :param checkpoint:
    :param config:
    :param fp16:
    :param audio_in_code:
    :param inference_test:
    :return:
    '''
    args = Namespace(
        source=source,
        target=target,
        output=output,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        semi_tone_shift=semi_tone_shift,
        checkpoint=checkpoint,
        config=config,
        fp16=fp16,
        audio_in_code=audio_in_code,
        inference_test=inference_test,
        noise_gaussian=noise_gaussian
    )

    return inference.main(args)


if __name__ == '__main__':

    audios = [r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_19273358.mp3",
              r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\cv-corpus-mozilla-pt\pt\clips\common_voice_pt_19273361.mp3",
              r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\2024_AUDIOS_PROJETO_LARINGE\SEM_TRAQUEOSTOMIA\DALLETE_FONO\NATURAL_MP3\1_n.mp3"]

    audio_laringe = [r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\2024_AUDIOS_PROJETO_LARINGE\SEM_TRAQUEOSTOMIA\DALLETE_FONO\LARINGE_ELETRONICA_MP3\1_d.mp3",
                     r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\dataset\2024_AUDIOS_PROJETO_LARINGE\SEM_TRAQUEOSTOMIA\DALLETE_FONO\LARINGE_ELETRONICA_MP3\2_d.mp3"]

    models = [
        r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_10_8_0.1_gaussian_True_no_pre_train_whisper\ft_model.pth",
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_5_8_0.1_no_pre_train_xlsr_trimbre_f0_2\ft_model.pth',
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\output\runs\cv_mozila_pt_10_8_0.1_gaussian_True_timbre_f0_no_pre_train_xlsr\ft_model.pth',
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\output\runs\cv_mozila_pt_10_8_0.1_gaussian_True_no_pre_train\ft_model.pth',
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_5_8_0.1_no_pre_train_xlsr_2\ft_model.pth',
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_5_8_0.1_no_pre_train_xlsr_3\ft_model.pth',
        r'C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_5_8_0.1_no_pre_train_xlsr_4\ft_model.pth'
    ]

    output = r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\output\reconstructed"

    configs = [r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\main\arquitetura\seedVC\output\runs\cv_mozila_pt_10_8_0.1_gaussian_True_no_pre_train_whisper\config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
               r"C:\Users\USER\Documents\Mestrado\codigo\Mestrado_VC\output\runs\cv_mozila_pt_10_8_0.1_gaussian_True_timbre_f0_no_pre_train_xlsr\config_dit_mel_seed_uvit_xlsr_tiny.yml"]

    load_model = models[4]

    voice_conversion_inference(
        source=audios[0],
        target=audios[2],
        checkpoint=load_model,
        config=configs[1],
        noise_gaussian=True
    )


    voice_conversion_inference(
        source=audio_laringe[0],
        target=audios[2],
        checkpoint=load_model,
        config=configs[1],
        noise_gaussian=True
    )

    voice_conversion_inference(
        source=audios[2],
        target=audios[1],
        checkpoint=load_model,
        config= configs[1],
        noise_gaussian=True
    )