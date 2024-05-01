import torch.nn as nn
import torch
class MaskedConvolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, device, padding = (0, 0)):
        super(MaskedConvolution2D, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias = False)
        self.mask = self.conv2d.weight.clone().to(device)
        self.mask.fill_(1)
        _, _, kH, kW = self.conv2d.weight.size()
        self.mask[:, :, kH // 2, kW // 2+(mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.conv2d.weight.data*=self.mask
        out = self.conv2d(x)
        return out