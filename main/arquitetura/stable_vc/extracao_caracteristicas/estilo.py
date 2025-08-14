import numpy as np
import torch
from torch import nn

from bibliotecas_externas.Amphion.models import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download


def fa_enc_dec_define(in_channels: int = 256, out_channels: int = 256, ngf: int = 32,
                      upsample_initial_channel: int = 1024):
    fa_encoder = FACodecEncoder(
        ngf=ngf,
        up_ratios=[2, 4, 5, 5],
        out_channels=out_channels,
    )

    fa_decoder = FACodecDecoder(
        in_channels=in_channels,
        upsample_initial_channel=upsample_initial_channel,
        ngf=ngf,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))

    fa_encoder.eval()
    fa_decoder.eval()

    return fa_encoder, fa_decoder


def extracao_estilo(audio_wav: np.array, d_style: int, fa_encoder, fa_decoder, device: str = "cuda") -> torch.Tensor:
    '''

    :param fa_encoder:
    :param device:
    :param d_style:
    :param audio_wav:
    :param fa_decoder: decoder do FACodecDecoder
    :return: tensor [batch_size, shape_saida] exemplo: [1,128]
    '''

    wav = torch.from_numpy(audio_wav).float()
    wav = wav.unsqueeze(0).unsqueeze(0)

    # encode
    enc_out = fa_encoder(wav)

    vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)

    # get prosody code
    prosody_code = vq_id[:1]

    projecao = nn.Linear(prosody_code.shape[2], d_style).to(device)

    prosody_code = projecao(prosody_code[0].cuda().float())

    return prosody_code
