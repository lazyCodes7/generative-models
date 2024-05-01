from ..lstm.diagonal_bilstm import DiagonalBiLSTM
from ..lstm.row_lstm import RowLSTM
import torch.nn as nn
class ResidualBlock(nn.Module):
    def __init__(self, h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, device, input_size = 28, lstm_type = 'row_lstm'):
        super(ResidualBlock, self).__init__()
        if(lstm_type == 'row_lstm'):
            self.lstm = RowLSTM(h=h, in_channels=in_channels, out_channels=out_channels, in_channels_s=in_channels_s, out_channels_s=out_channels_s, batch_size=batch_size, device=device, input_size=input_size)
        else:
            self.lstm = DiagonalBiLSTM(h=h, in_channels=in_channels, out_channels=out_channels, in_channels_s=in_channels_s, out_channels_s=out_channels_s, batch_size=batch_size, device=device, input_size=input_size)
        self.conv = nn.Conv2d(in_channels=h, out_channels=2*h, kernel_size=1)

    def forward(self, image):
        output = self.lstm(image)
        return image + self.conv(output)