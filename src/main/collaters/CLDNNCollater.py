import numpy as np
import torch


class CLDNNCollater(object):
    """Customized collater for Pytorch DataLoader in non-autoregressive VC training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""

    def __call__(self, batch):
        def pad_list(xs, pad_value):
            """Perform padding for the list of tensors.

            Args:
                xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
                pad_value (float): Value for padding.

            Returns:
                Tensor: Padded tensor (B, Tmax, `*`).

            Examples:
                >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
                >>> x
                [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
                >>> pad_list(x, 0)
                tensor([[1., 1., 1., 1.],
                        [1., 1., 0., 0.],
                        [1., 0., 0., 0.]])

            """
            n_batch = len(xs)
            max_len = max(x.size(0) for x in xs)
            pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

            for i in range(n_batch):
                pad[i, : xs[i].size(0)] = xs[i]

            return pad

        audios = []
        audios_noise = []
        xs = []
        ys = []
        srs = []
        sentences = []

        for b in batch:
            # Definir um tamanho padrão

            x = b["mel_noise"]
            y = b["mel"]


            if x.shape[1] > y.shape[1]:
                x = x[:, : y.shape[1]]
            else:
                y = y[:, : x.shape[1]]

            xs.append(x)
            ys.append(y)

            audios.append(b["audio"])
            audios_noise.append(b["audio_noise"])
            sentences.append(b["sentence"])
            srs.append(b["sample_rate"])

        xs = pad_list([torch.from_numpy(x).float().transpose(1, 0) for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float().transpose(1, 0) for y in ys], 0)

        return audios,audios_noise, ys, xs, srs, sentences