from torch import nn


class DiscriminativeNetwork(nn.Module):

    def __init__(self, device='cpu'):
        super(DiscriminativeNetwork, self).__init__()
        self.device = device
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(2),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),  # 3x1
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_features=14400, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()

        ).to(self.device)

    def forward(self, inputs):
        if inputs.device == 'cpu':
            inputs = inputs.to(self.device)
        return self.classifier(inputs)