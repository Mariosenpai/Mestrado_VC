from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, device='cpu'):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=1)
        ).to(device)

        self.ext_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, stride=2)
        ).to(device)

    def forward(self, inputs):
        extended_input = self.ext_block(inputs)
        convolved_input = self.block(inputs)
        return convolved_input + extended_input


class ConvResidualBlock(nn.Module):

    def __init__(self, device='cpu'):
        super(ConvResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding="same"),
            nn.Conv2d(1, 1, 3, stride=1, padding="same")
        ).to(device)

        self.conv_1d = nn.Conv2d(1, 1, 3, stride=1, padding="same").to(device)

    def forward(self, inputs):
        convolved_input = self.block(inputs)
        skip_con = self.conv_1d(inputs)
        return convolved_input + skip_con