import numpy as np
import torch


class ARVCCollater(object):
    """Customized collater for Pytorch DataLoader in autoregressive VC training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""

    def __call__(self, batch):
        """Convert into batch tensors."""

        def _ensure_td(x):
            """
            Garante que o tensor esteja em (T, D)
            """
            if isinstance(x, torch.Tensor):
                x = x.detach()
            else:
                x = torch.from_numpy(x)

            x = x.squeeze()

            if x.ndim != 2:
                raise ValueError(f"Expected 2D tensor (T, D), got {x.shape}")

            # Heurística segura para mel-spectrograma
            # D costuma ser pequeno (ex: 80)
            if x.shape[0] == 80 and x.shape[1] != 80:
                x = x.transpose(0, 1)

            return x.float()

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

        xs, ys, srs,audios = [b["mel_noise"] for b in batch], [b["mel"] for b in batch], [b["sample_rate"] for b in batch], [b["audio"] for b in batch]

        # get list of lengths (must be tensor for DataParallel)
        ilens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()
        olens = torch.from_numpy(np.array([y.shape[0] for y in ys])).long()

        # perform padding and conversion to tensor
        xs = pad_list([_ensure_td(x) for x in xs], 0)
        ys = pad_list([_ensure_td(y) for y in ys], 0)

        # make labels for stop prediction
        labels = ys.new_zeros(ys.size(0), ys.size(1))
        for i, l in enumerate(olens):
            labels[i, l - 1:] = 1.0

        items = {
            "xs": xs,
            "ilens": ilens,
            "ys": ys,
            "olens": olens,
            "labels": labels,
            "spembs": None,
            "sr": srs,
            "audios":audios,
        }

        return items
