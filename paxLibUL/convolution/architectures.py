import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        layers = [nn.Conv2d(3, 100, 3, padding=1), nn.ReLU()]
        self.num_layers = num_layers
        for layer_number in range(2, num_layers + 1):
            layers.append(nn.Conv2d(100, 100, 3, padding=1))
            layers.append(nn.ReLU())
            if layer_number in [3, 5]:
                layers.append(nn.MaxPool2d(2))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(100 * 8 * 8, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        for layer in self.model:
            x = layer(x)

        return x

    def eval_weight_distribution(self, x):
        k = 2000  # Chiffre de valeurs aléatoires
        entree = {}
        indice = 0
        for layer in self.model:
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                indice += 1
                if indice in (1, self.num_layers - 1):  # Première ou dernière couche
                    echantillon = nn.Flatten(0)(x.detach().cpu())
                    entree[f"Conv_{indice}"] = np.random.choice(echantillon, size=k)

            x = layer(x)

        return x, entree


class ResidualLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(100, 100, 3, padding=1)

        self.conv2 = nn.Conv2d(100, 100, 3, padding=1)

    def forward(self, x):
        res = x
        x = F.relu((self.conv1(x)))
        x = self.conv2(x)
        x = F.relu(x + res)
        return x


class ResNet(nn.Module):
    def __init__(self, nb_bloc_residuelle):
        super().__init__()
        assert nb_bloc_residuelle > 2

        self.conv1 = nn.Conv2d(3, 100, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(100)

        self.layers = nn.ModuleList([ResidualLayer() for _ in range(nb_bloc_residuelle)])
        self.fc1 = nn.Linear(100 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i <= 1:
                x = F.max_pool2d(x, 2)

        x = x.flatten(1)
        x = self.fc1(x)
        return x


class UNetSkipConnectionBlock(nn.Module):
    """Defines the U-Net submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
    ):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int): The number of filters in the outer conv layer.
            inner_nc (int): The number of filters in the inner conv layer.
            input_nc (int): The number of channels in input images/features.
            submodule (UnetSkipConnectionBlock): Previously defined submodules.
            outermost (bool): If this module is the outermost module.
            innermost (bool): If this module is the innermost module.
            norm_layer: Normalization layer.
            use_dropout (bool): If use dropout layers.
        """
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # Add skip connections
            return torch.cat([x, self.model(x)], 1)
