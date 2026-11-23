"""SiameseFIT encoder used as reference in this project.

The module published in SiameseFit-pub (Module22xP_1.py) has 1,582,560 parameters and
evaluates two unused branches. The official notebooks, however, report 100,705 parameters.
This file reconstructs that network: four main convolutions, one unused 128->1 convolution
and four skip convolutions, defined in this order. With seed 0 it reproduces the first-epoch
loss of the official SiameseFit_Flat_Overthrust notebook (0.00129976 vs 0.00129977).
"""
import torch.nn as nn
import torch.nn.init as init


def weights_init(m, leak_value):
    """Glorot (uniform) initialization with the given gain, zero bias."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.xavier_uniform_(m.weight, leak_value)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


class SiameseNetwork(nn.Module):
    """Four-block encoder: downsampling of 16 x 16 and 8*D1 latent feature maps."""

    def __init__(self, D1):
        super(SiameseNetwork, self).__init__()
        self.D1 = D1
        ch = [1, D1, 2 * D1, 4 * D1, 8 * D1]

        # main path (the definition order matters for the random initialization)
        self.cnn1 = nn.Conv2d(ch[0], ch[1], kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv2d(ch[1], ch[2], kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1, padding=1)

        # layer present in the official network but not used in the forward pass
        self.cnn5 = nn.Conv2d(ch[4], 1, kernel_size=3, stride=1, padding=1)

        # skip branches, fed directly by the input
        self.cnnXX1 = nn.Conv2d(1, ch[1], kernel_size=3, stride=1, padding=1)
        self.cnnXX2 = nn.Conv2d(1, ch[2], kernel_size=3, stride=1, padding=1)
        self.cnnXX3 = nn.Conv2d(1, ch[3], kernel_size=3, stride=1, padding=1)
        self.cnnXX4 = nn.Conv2d(1, ch[4], kernel_size=3, stride=1, padding=1)

        self.act = nn.LeakyReLU(0.5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.pool_skip = nn.ModuleList([nn.MaxPool2d(2 ** k, stride=2 ** k) for k in range(1, 5)])

    def forward_once(self, x):
        """Return the latent representation of one shot gather."""
        xx1 = self.act(self.cnnXX1(x))
        xx2 = self.act(self.cnnXX2(x))
        xx3 = self.act(self.cnnXX3(x))
        xx4 = self.act(self.cnnXX4(x))

        output1 = self.pool(self.act(self.cnn1(x))) + self.pool_skip[0](xx1)
        output2 = self.pool(self.act(self.cnn2(output1))) + self.pool_skip[1](xx2)
        output3 = self.pool(self.act(self.cnn3(output2))) + self.pool_skip[2](xx3)
        output4 = self.pool(self.act(self.cnn4(output3))) + self.pool_skip[3](xx4)

        return output4

    def forward(self, input1):
        """Each branch of the Siamese pair is evaluated by one call."""
        return self.forward_once(input1)
