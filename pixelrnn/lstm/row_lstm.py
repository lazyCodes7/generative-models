import torch.nn as nn
import torch
from ..common.masked_convolution import MaskedConvolution2D
class RowLSTM(nn.Module):
    def __init__(self,h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, input_size, device):
        super(RowLSTM, self).__init__()
        self.itos = MaskedConvolution2D(in_channels=in_channels, out_channels=out_channels, padding=(1,0), kernel_size=(3,1), mask_type='B', device=device)
        self.stos = nn.Conv2d(in_channels=in_channels_s, out_channels=out_channels_s, padding=(1,0), kernel_size=(3,1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.num_h = h
        self.device = device
        h0, c0 = self.init_hidden_state(input_size)
        self.h0 = nn.Parameter(h0, requires_grad = True)
        self.c0 = nn.Parameter(c0, requires_grad = True)

    def forward(self, image):
        batch_size, _, n_seq, _ = image.shape
        init_hidden = self.h0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        init_cell = self.c0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        feature_states = self.itos(image)
        hidden_states = [init_hidden]
        cell_states = [init_cell]
        for j in range(n_seq):
            hidden_out = self.stos(hidden_states[j]) + feature_states[:,:,j,:].unsqueeze(3)
            i, g, f, o = torch.split(hidden_out, (4 * self.num_h)//4, dim=1)
            cell_state = torch.mul(torch.sigmoid(f), cell_states[j]) + torch.mul(torch.sigmoid(i), torch.tanh(g))
            cell_states.append(cell_state)
            hidden_state = torch.mul(cell_states[j], torch.sigmoid(o))
            hidden_states.append(hidden_state)

        hidden_states = torch.stack(hidden_states[1:], dim = 2).squeeze()
        return hidden_states


    def init_hidden_state(self, input_size):
        h0 = torch.randn((self.num_h, input_size, 1)).to(self.device)
        c0 = torch.randn((self.num_h, input_size, 1)).to(self.device)
        return h0, c0