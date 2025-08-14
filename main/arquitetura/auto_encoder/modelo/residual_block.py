from torch import nn


class ResidualBlockAuto(nn.Module):

    def __init__(self, device='cpu'):
        super(ResidualBlockAuto, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1)
        ).to(device)

    def forward(self, inputs):
        convolved_input = self.block(inputs)
        return convolved_input + inputs