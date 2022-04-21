from functools import partial

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


"""
The code below was copied from the "Image-to-Image Translation in PyTorch" repository 

COPYRIGHT

All contributions from the https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/cf4191a3a4cc77fdffa5c0a8246c346049958e78/models/networks.py#L468.
 authors.
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


--------------------------- LICENSE FOR pix2pix --------------------------------
BSD License

For pix2pix software
Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

----------------------------- LICENSE FOR DCGAN --------------------------------
BSD License

For dcgan.torch software

Copyright (c) 2015, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the 
following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following 
disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name Facebook nor the names of its contributors may be used to endorse or promote products derived from 
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
DAMAGE.
"""


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
        if isinstance(norm_layer, partial):
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
