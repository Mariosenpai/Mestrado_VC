import torch

def collate_fn(batch):

    audio_v = []
    mel_v = []
    mel_noise_v = []
    sr_v = []
    sentence_v = []
    mel_lens_v = []
    meln_lens_v = []

    max_len_mel_noise = 0
    max_len_mel = 0

    # ==========================
    # 1. Coleta
    # ==========================
    for item in batch:
        audio = item['audio']
        sr = item['sample_rate']
        sentence = item['sentence']

        # garantir formato (T, 80)
        mel = torch.tensor(item['mel'])
        mel_noise = torch.tensor(item['mel_noise'])

        mel_len = mel.shape[0]        # tempo é dim 0
        meln_len = mel_noise.shape[0] # tempo é dim 0

        audio_v.append(audio)
        mel_v.append(mel)
        mel_noise_v.append(mel_noise)
        sr_v.append(sr)
        sentence_v.append(sentence)
        mel_lens_v.append(mel_len)
        meln_lens_v.append(meln_len)

        max_len_mel = max(max_len_mel, mel_len)
        max_len_mel_noise = max(max_len_mel_noise, meln_len)

    # ==========================
    # 2. Padding mel
    # ==========================
    padded_mel = []
    for mel in mel_v:
        T, C = mel.shape
        pad_len = max_len_mel - T

        if pad_len > 0:
            pad = torch.zeros(pad_len, C, dtype=mel.dtype)
            mel = torch.cat([mel, pad], dim=0)

        padded_mel.append(mel)

    # ==========================
    # 3. Padding mel_noise
    # ==========================
    padded_mel_noise = []
    for mel_noise in mel_noise_v:
        T, C = mel_noise.shape
        pad_len = max_len_mel_noise - T

        if pad_len > 0:
            pad = torch.zeros(pad_len, C, dtype=mel_noise.dtype)
            mel_noise = torch.cat([mel_noise, pad], dim=0)

        padded_mel_noise.append(mel_noise)

    # ==========================
    # 4. Stack
    # ==========================
    mel_v = torch.stack(padded_mel, dim=0)
    mel_noise_v = torch.stack(padded_mel_noise, dim=0)

    mel_lens_v = torch.tensor(mel_lens_v, dtype=torch.long)
    meln_lens_v = torch.tensor(meln_lens_v, dtype=torch.long)

    return (
        audio_v,
        mel_v,              # (B, T_max, 80)
        mel_noise_v,        # (B, Tn_max, 80)
        sr_v,
        sentence_v,
        mel_lens_v,
        meln_lens_v
    )

def collate_fn_aasvc(batch):

    xs = []         # src_feat
    ys = []         # trg_feat
    dp_inputs = []  # duration predictor input

    ilens = []
    olens = []
    dplens = []

    audio_v =[]
    sr_v =[]

    max_src_len = 0
    max_trg_len = 0

    # ==========================
    # 1. Coleta
    # ==========================
    for item in batch:

        audio = item['audio']
        sr = item['sample_rate']
        audio_v.append(audio)
        sr_v.append(sr)

        # Garantir formato (T, C)
        src = torch.tensor(item["mel_noise"]).float()
        trg = torch.tensor(item["mel"]).float()

        src_len = src.shape[0]
        trg_len = trg.shape[0]

        xs.append(src)
        ys.append(trg)
        dp_inputs.append(src)  # normalmente igual ao src

        ilens.append(src_len)
        olens.append(trg_len)
        dplens.append(src_len)

        max_src_len = max(max_src_len, src_len)
        max_trg_len = max(max_trg_len, trg_len)

    # ==========================
    # 2. Padding src
    # ==========================
    padded_xs = []
    for x in xs:
        T, C = x.shape
        pad_len = max_src_len - T
        if pad_len > 0:
            pad = torch.zeros(pad_len, C, dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        padded_xs.append(x)

    # ==========================
    # 3. Padding trg
    # ==========================
    padded_ys = []
    for y in ys:
        T, C = y.shape
        pad_len = max_trg_len - T
        if pad_len > 0:
            pad = torch.zeros(pad_len, C, dtype=y.dtype)
            y = torch.cat([y, pad], dim=0)
        padded_ys.append(y)

    # ==========================
    # 4. Stack
    # ==========================
    xs = torch.stack(padded_xs, dim=0)      # (B, T_src_max, C)
    ys = torch.stack(padded_ys, dim=0)      # (B, T_trg_max, C)
    dp_inputs = xs.clone()                  # normalmente igual ao src

    ilens = torch.tensor(ilens, dtype=torch.long)
    olens = torch.tensor(olens, dtype=torch.long)
    dplens = torch.tensor(dplens, dtype=torch.long)

    return {
        "xs": xs,
        "ilens": ilens,
        "ys": ys,
        "olens": olens,
        "dp_inputs": dp_inputs,
        "dplens": dplens,
        "spembs": None,
        "audio": audio_v,
        "sr": sr_v,
    }

def get_info_audio(batch):
    audio, _, _, sr, sentence, _, _ = batch
    return audio, sr, sentence

def format_batch(batch, device):
    _, mel, mel_noise, _, _, mel_lens, meln_lens = batch

    B = len(mel)
    T_tgt = mel.size(1)

    labels = torch.zeros(B, T_tgt, device=device)
    labels[torch.arange(B), mel_lens[0] - 1] = 1.0

    xs = mel_noise.to(device)
    ilens = torch.tensor(meln_lens, dtype=torch.long).to(device)
    ys = mel.to(device)
    olens = mel_lens.to(device)

    return xs, ys, ilens, olens, labels