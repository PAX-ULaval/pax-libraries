import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Default architecture use for the formations.
"""


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
        entree = dict()
        indice = 0
        for layer in self.model:
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                indice += 1
                if (
                        indice == 1 or indice == self.num_layers - 1
                ):  # Première ou dernière couche

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

        self.layers = nn.ModuleList(
            [ResidualLayer() for _ in range(nb_bloc_residuelle)]
        )

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
