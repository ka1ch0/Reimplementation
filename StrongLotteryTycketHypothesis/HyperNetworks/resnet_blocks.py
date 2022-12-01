import torch.nn as nn
import torch.nn.functional as F


class Identity_layer(nn.Module):
    """The layer to identity mapping. This layer used for skip connection."""
    def forward(self, x):
        return x



class ResNetBlock(nn.Module):
    def __init__(self, in_size: int = 16, out_size: int = 16, downsample: bool = False):
        """
        Parameters
        ----------
        in_size: int
            In channels.
        out_size : int
            Out channels.
        downsample : bool
            Set True if feature map size change.
        """
        super().__init__()
        self.out_size = out_size
        self.in_size = in_size

        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = Identity_layer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w):
        residual = self.reslayer(x)

        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=self.stride1, padding=1)), inplace=True)
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual

        out = F.relu(out)

        return out
