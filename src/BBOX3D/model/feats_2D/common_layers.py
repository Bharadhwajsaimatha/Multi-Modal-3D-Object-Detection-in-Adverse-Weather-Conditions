import torch.nn as nn

class FCNLayer(nn.Module):
    def __init__(
            self,
            channel_in,
            channel_out,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            bntrack = True
            ):
        super().__init__()
        self.linear = nn.Linear(channel_in,channel_out,bias=True)
        self.ReLU = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d(channel_out, eps=eps, momentum=momentum, affine=affine,track_running_stats=bntrack)

    def forward(self,x):
        return self.BatchNorm(self.ReLU(self.linear(x)))
    
class CRB2D(nn.Module):
    def __init__(
            self,
            channel_in,
            channel_out,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            bntrack = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(channel_in,channel_out,kernel_size=kernel_size,stride=stride,padding=padding)
        self.ReLU = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d(channel_out, eps=eps, momentum=momentum, affine=affine,track_running_stats=bntrack)

    def forward(self,x):
        return self.BatchNorm(self.ReLU(self.conv(x)))

class DeCRB2D(nn.Module):
    def __init__(
            self,
            channel_in,
            channel_out,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            eps = 1e-5,
            momentum = 0.1,
            affine = True,
            bntrack = True
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channel_in,channel_out,kernel_size=kernel_size,stride=stride,padding=padding)
        self.ReLU = nn.ReLU()
        self.BatchNorm = nn.BatchNorm2d(channel_out, eps=eps, momentum=momentum, affine=affine,track_running_stats=bntrack)

    def forward(self,x):
        return self.BatchNorm(self.ReLU(self.deconv(x)))