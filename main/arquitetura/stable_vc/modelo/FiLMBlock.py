import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, n_res_blocks, n_channels):
        super(FiLM, self).__init__()

        dim_question = 11
        # Linear에서 나온 결과의 절반은 beta, 절반은 gamma
        # beta, gamma 모두 ResBlock 하나당 n_channels개씩 feed
        self.film_generator = nn.Linear(dim_question, 2 * n_res_blocks * n_channels)
        self.feature_extractor = FeatureExtractor()
        self.res_blocks = nn.ModuleList()

        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))

        #self.classifier = Classifier(n_channels, n_classes)

        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels

    def forward(self, x, features):
        batch_size = x.size(0)

        x = self.feature_extractor(x)
        film_vector = self.film_generator(features).view(
            batch_size, self.n_res_blocks, 2, self.n_channels)

        d = x.size(2)
        coordinate = torch.arange(-1, 1 + 0.00001, 2 / (d - 1)).cuda()
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)

        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]

            x = torch.cat([x, coordinate_x, coordinate_y], 1)
            x = res_block(x, beta, gamma)

        # feature = x
        #x = self.classifier(x)

        return x  # , feature


class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x


def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = nn.Sequential(
            # conv(3, 128, 5, 1, 2),
            # conv(128, 128, 3, 1, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            conv(3, 128, 5, 2, 2),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
            # conv(3, 128, 4, 2, 2),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)

class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x