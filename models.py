import torch.nn as nn
import utils


class Shared_encoder(nn.Module):

    def __init__(self):
        super(Shared_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.encoder(x)
        return y


class Cloud_model(nn.Module):

    def __init__(self):
        super(Cloud_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 224, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(224), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(224, 224, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(224), nn.ReLU(inplace=True),
            nn.Conv2d(224, 224, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(224), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 224, out_features=10, bias=True), )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class Co_submodel(nn.Module):

    def __init__(self):
        super(Co_submodel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2),
            nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 32, out_features=10, bias=True), )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class Control_model(nn.Module):

    def __init__(self):
        super(Control_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2),
            nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(in_features=8 * 8 * 32, out_features=10, bias=True), )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class VGG_shared_encoder(nn.Module):

    def __init__(self):
        super(VGG_shared_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x):
        y = self.encoder(x)
        return y


class VGG_cloud_model(nn.Module):

    def __init__(self):
        super(VGG_cloud_model, self).__init__()
        self.layers = [
            56, 'M', 112, 112, 'M', 224, 224, 224, 'M', 448, 448, 448, 'M',
            448, 448, 448, 'M'
        ]

        self.encoder = utils.VGGG_make_layers(self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(1 * 1 * 448, 1 * 1 * 448),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1 * 1 * 448, 1 * 1 * 448),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1 * 1 * 448, 10),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class VGG_co_submodel(nn.Module):

    def __init__(self):
        super(VGG_co_submodel, self).__init__()
        self.layers = [
            8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64,
            'M'
        ]

        self.encoder = utils.VGGG_make_layers(self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(1 * 1 * 64, 1 * 1 * 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1 * 1 * 64, 1 * 1 * 64),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1 * 1 * 64, 10),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class VGG_control_model(nn.Module):

    def __init__(self):
        super(VGG_control_model, self).__init__()
        self.layers = [
            8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64,
            'M'
        ]

        self.encoder = utils.VGGG_make_layers(self.layers)
        self.classifier = nn.Sequential(
            nn.Linear(1 * 1 * 64, 1 * 1 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1 * 1 * 64, 1 * 1 * 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1 * 1 * 64, 10),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y


class VGG_base_model(nn.Module):

    def __init__(self):
        super(VGG_base_model, self).__init__()
        self.layers = [
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512, 'M'
        ]

        self.encoder = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(1 * 1 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        y = self.encoder(x)
        y = self.classifier(y)
        return y

    def _make_layers(self, batch_norm=True):
        layers = []

        input_channel = 3
        for l in self.layers:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l
        layers += [nn.Flatten()]

        return nn.Sequential(*layers)