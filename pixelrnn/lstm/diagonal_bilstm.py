from ..common.masked_convolution import MaskedConvolution2D
import torch.nn as nn
import torch
class DiagonalLSTM(nn.Module):
  def __init__(self, h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, input_size, device):
    super(DiagonalLSTM, self).__init__()
    self.itos = MaskedConvolution2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), mask_type='B', device=device)
    self.stos = MaskedConvolution2D(in_channels=in_channels_s, out_channels=out_channels_s, kernel_size=(2,1), mask_type='B', device=device)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.num_h = h
    self.device = device
    h0, c0 = self.init_hidden_state(input_size)
    self.h0 = nn.Parameter(h0, requires_grad = True)
    self.c0 = nn.Parameter(c0, requires_grad = True)

  def skew(self, inputs):
    batch_size, channels, h, _ = inputs.shape
    skewed_inputs = torch.zeros((batch_size, channels, h, (2*h)-1))
    for i in range(h):
      skewed_inputs[:, :, i, i:i+h] = inputs[:, :, i]
    return skewed_inputs.to(self.device)


  def unskew(self, skewed_inputs):
    batch_size, channels, h, _ = skewed_inputs.shape
    unskewed_inputs = []
    for i in range(h):
      unskewed_inputs.append(skewed_inputs[:, :, i, i:i+h])
    return torch.stack(unskewed_inputs, dim = 2).to(self.device)

  def forward(self, image):
    image = self.skew(image)
    batch_size, _, h, n_seq = image.shape
    init_hidden = self.h0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    init_cell = self.c0.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    feature_states = self.itos(image)
    hidden_states = [init_hidden]
    cell_states = [init_cell]
    for j in range(n_seq):
        hidden_out = self.stos(
            torch.concat(
                (
                    torch.zeros(batch_size, self.num_h, 1, 1).to(self.device),
                    hidden_states[j]
                ),
                dim = 2
            )
        ) + feature_states[:,:,:,j].unsqueeze(3)
        i, g, f, o = torch.split(hidden_out, (4 * self.num_h)//4, dim=1)
        cell_state = torch.mul(torch.sigmoid(f), cell_states[j]) + torch.mul(torch.sigmoid(i), torch.tanh(g))
        cell_states.append(cell_state)
        hidden_state = torch.mul(cell_states[j], torch.sigmoid(o))
        hidden_states.append(hidden_state)

    hidden_states = torch.stack(hidden_states[1:], dim = 2).squeeze().permute(0,1,3,2)
    return self.unskew(hidden_states)
  def init_hidden_state(self, input_size):
    h0 = torch.randn((self.num_h, input_size, 1)).to(self.device)
    c0 = torch.randn((self.num_h, input_size, 1)).to(self.device)
    return h0, c0
  

class DiagonalBiLSTM(nn.Module):
  def __init__(self, h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, input_size, device):
    super(DiagonalBiLSTM, self).__init__()
    self.device = device
    self.h = h
    self.forward_lstm = DiagonalLSTM(h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, input_size, device)
    self.backward_lstm = DiagonalLSTM(h, in_channels, out_channels, in_channels_s, out_channels_s, batch_size, input_size, device)

  def forward(self, image):
    batch_size, _ , height, width = image.shape
    forward = self.forward_lstm(image)
    backward = self.backward_lstm(image.flip(dims=[3])).flip(dims=[3])
    return forward + torch.cat((torch.zeros(batch_size, self.h, 1, width).to(self.device), backward[:,:,:-1,:]), dim = 2)
