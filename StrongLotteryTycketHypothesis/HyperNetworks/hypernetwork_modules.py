import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd as autograd


class HyperNetwork(nn.Module):
    def __init__(self, f_size: int = 3, z_dim : int = 64, out_size: int = 16, in_size: int = 16):
        """
        Parameters
        ----------
        f_size : int
            The kernel size of a CNN.
        z_dim : int
            Dimension of embedded parameteres for each layers of a CNN.
        out_size : int
            Out chennel of the feature map for each layers of a CNN.
        in_size : int
            In chennel of the feature map for each layers of a CNN.
        """
        super().__init__()

        self.z_dim = z_dim  # N_z
        self.f_size = f_size  # f_size
        self.out_size = out_size  # N_out
        self.in_size = in_size  # N_in

        # Parameters of first layer
        self.w1 = Parameter(torch.fmod(torch.randn(self.z_dim, self.z_dim * self.in_size).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn(self.z_dim * self.in_size).cuda(), 2))  # d = N_z * N_in

        # Parameters of sedond layer
        self.w2 = Parameter(torch.fmod(torch.randn(self.z_dim, self.f_size * self.f_size * self.out_size).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn(self.f_size * self.f_size * self.out_size).cuda(), 2))

    def forward(self, z):
        # z: (B, z_dim)

        # Linear projection 1
        # (B, z_dim) x (z_dim, z_dim * in_size) -> (B, z_dim * in_size)
        h_in = torch.matmul(z, self.w1) + self.b1

        # (B, z_dim * in_size) -> (B, in_size, z_dim)
        h_in = h_in.reshape(self.in_size, self.z_dim)

        # Linear projection 2
        # (B, in_size, z_dim) -> (B, in_size, out_size * f_size * f_size)
        h_final = torch.matmul(h_in, self.w2) + self.b2

        # (B, in_size, out_size * f_size * f_size) -> (B, out_size, in_size, f_size, f_size)
        # The shape of output is equal the shape of parameters of a convolutional layer. 
        out = h_final.reshape(self.out_size, self.in_size, self.f_size, self.f_size)

        return out


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


if __name__ == "__main__":
    z = torch.randn(64).cuda()
    hypernet = HyperNetwork()

    score = hypernet(z)
    mask = GetSubnet.apply(score, 0.5)
    loss = mask.sum()

    loss.backward()

    print("score grad:")
    print("=" * 50)
    print(score.grad)

    print("hypernet grad:")
    print("=" * 50)
    print("w1")
    print(hypernet.w1.grad)
    print("=" * 50)
    print("w2")
    print(hypernet.w2.grad)
    print("=" * 50)
    print("b1")
    print(hypernet.b1.grad)
    print("=" * 50)
    print("b2")
    print(hypernet.b2.grad)
