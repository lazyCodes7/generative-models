import torch.nn as nn
import torch
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual = False
    ):
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),

            # GroupNorm as it is mentioned in the paper
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        # MHA
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)

        
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        image_size = x.shape[-2]
        x = x.view(-1, self.channels, image_size * image_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        
        # Typical Layernorm + output from MHA
        attention_value = attention_value + x

        # Passing through a Feed forward
        attention_value = self.ff_self(attention_value) + attention_value

        return attention_value.swapaxes(2, 1).view(-1, self.channels, image_size, image_size)


class DownBlock(nn.Module):
    def __init__(self, features = None, in_channels = None, emb_dim = 256):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.downsample = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.timestep_emb = nn.ModuleList()
        for feature in features:
            self.downsample.append(
                nn.Sequential(
                    DoubleConv(
                        in_channels,
                        in_channels,
                        residual = True
                    ),
                    DoubleConv(
                        in_channels,
                        feature
                    )
                )

            )
            self.timestep_emb.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        emb_dim,
                        feature
                    ),
                )
            )
            self.attention.append(
                SelfAttention(feature)
            )
            in_channels = feature
    
    def forward(self, image, timestep):
        skip_connections = []
        for i in range(len(self.downsample)):
            image = self.downsample[i](image)
            emb = self.timestep_emb[i](timestep)[:, :, None, None].repeat(1, 1, image.shape[-2], image.shape[-1])
            image = image + emb
            image = self.attention[i](image)
            skip_connections.append(image)
            image = self.pool(image)
        return image, skip_connections

class UpBlock(nn.Module):
    def __init__(self, features = None, emb_dim = 256):
        super(UpBlock, self).__init__()
        self.upsample = nn.ModuleList()
        self.timestep_emb = nn.ModuleList()
        self.attention = nn.ModuleList()
        for feature in reversed(features):
            self.upsample.append(
                nn.ConvTranspose2d(
                    feature*2,
                    feature,
                    kernel_size = 2,
                    stride = 2

                )
            )
            self.upsample.append(
                nn.Sequential(
                    DoubleConv(feature*2, feature*2, residual=True),
                    DoubleConv(feature*2, feature)
                )
            )
            self.attention.append(
                SelfAttention(feature)
            )
            self.timestep_emb.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(
                        emb_dim,
                        feature
                    ),
                )
            )

    def forward(self, image, timestep, skip_connections):
        skip_connections = skip_connections[::-1]
        
        for i in range(0, len(self.upsample), 2):

            image = self.upsample[i](image)
            image = torch.cat((image, skip_connections[i//2]), dim = 1)
            image = self.upsample[i+1](image)
            emb = self.timestep_emb[i//2](timestep)[:, :, None, None].repeat(1, 1, image.shape[-2], image.shape[-1])
            image = image + emb
            image = self.attention[i//2](image)
        
        return image
            

class UNet(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        out_channels = 3,
        features = [
            64, 128, 256, 512
        ],
        time_dim = 256,
        device = 'cpu'

    ):
        super(UNet, self).__init__()
        self.down = DownBlock(
            features = features,
            in_channels = 3
        )
        self.up = UpBlock(
            features = features
        )
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.bottleneck = DoubleConv(
            features[-1],
            features[-1]*2
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)
        self.time_dim = time_dim
        self.device = device
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, image, timestep):
        timestep = timestep.unsqueeze(-1).float()
        timestep = self.pos_encoding(timestep, self.time_dim)
        image, skip_connections = self.down(image, timestep)
        image = self.bottleneck(image) 
        image = self.up(image, timestep, skip_connections)
        image = self.final_conv(image)
        return image

class ConditionalUNet(UNet):
    def __init__(
        self,
        in_channels = 3,
        out_channels = 3,
        features = [
            64, 128, 256, 512
        ],
        time_dim = 256,
        device = 'cpu',
        num_classes = 10
    ):
        super().__init__(in_channels, out_channels, features, time_dim, device)
        self.embedding = nn.Embedding(num_classes, time_dim)
    

    def forward(self, image, timestep, label):
        timestep = timestep.unsqueeze(-1).float()
        timestep = self.pos_encoding(timestep, self.time_dim)
        timestep = self.embedding(label) + timestep
        image, skip_connections = self.down(image, timestep)
        image = self.bottleneck(image) 
        image = self.up(image, timestep, skip_connections)
        image = self.final_conv(image)
        return image
