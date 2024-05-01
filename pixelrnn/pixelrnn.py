from .common.masked_convolution import MaskedConvolution2D
from .common.residual_blocks import ResidualBlock
import torch.nn.functional as F
import torch.nn as nn
class PixelRNN(nn.Module):
  def __init__(self, h, batch_size, device, input_size = 28, blocks = 7):
    super(PixelRNN, self).__init__()
    self.masked_conv = MaskedConvolution2D(in_channels=1, out_channels=2*h, padding=3, kernel_size=7, mask_type='A', device=device)
    self.n_blocks = blocks
    self.residual_blocks = nn.ModuleList([ResidualBlock(h = h, in_channels=2*h, out_channels=4*h, in_channels_s=h, out_channels_s=4*h, batch_size=batch_size, device=device, input_size = input_size) for i in range(self.n_blocks)] )

    self.masked_conv2 = MaskedConvolution2D(in_channels=2*h, out_channels=2*h, kernel_size=1, padding=0, mask_type='B', device=device)
    self.masked_conv3 = MaskedConvolution2D(in_channels=2*h, out_channels=2*h, kernel_size=1, padding=0, mask_type='B', device=device)
    self.masked_conv4 = MaskedConvolution2D(in_channels=2*h, out_channels=1, kernel_size=1, padding=0, mask_type='B', device=device)

  def forward(self, x):
    block_out = self.masked_conv(x)
    for i in range(self.n_blocks):
      block_out = self.residual_blocks[i](block_out)
    masked_conv2_out = F.relu(self.masked_conv2(block_out))
    masked_conv3_out = F.relu(self.masked_conv3(masked_conv2_out))
    masked_conv4_out = F.sigmoid(self.masked_conv4(masked_conv3_out))
    return masked_conv4_out
