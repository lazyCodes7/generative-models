import torch.nn as nn
import torch.nn.functional as F
import torch

# Implementation of a vanilla VAE which is not conditional in nature
class CVAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim, n_classes, embedding_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.embed_cond = nn.Embedding(num_embeddings=n_classes,
                                       embedding_dim=embedding_dim,
                                       max_norm=True)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + embedding_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, int(self.h_dim//2)),
            nn.LeakyReLU(0.2),

        )


        self.hidden_2mu = nn.Linear(int(self.h_dim/2), self.z_dim)
        self.hidden_2log_var = nn.Linear(int(self.h_dim/2), self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim + embedding_dim, int(self.h_dim/2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.h_dim/2), self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, y = None):
        # encoding function outputs p(z|x)
        if(y!=None):
          y = self.embed_cond(y)
          new_x = torch.cat((x, y), dim = 1)
        encoded_image = self.encoder(new_x)
        mu, log_var = self.hidden_2mu(encoded_image), self.hidden_2log_var(encoded_image)
        return mu, log_var


    def decode(self, z, y = None):
        if(y!=None):
          y = self.embed_cond(y)
          z = torch.cat([z, y], dim = 1)
        #decoding function outputs p(x|z)
        image_reconstructed = self.decoder(z)
        return image_reconstructed

    def reparametrize(self, mean, var):
      epsilon = torch.randn_like(var).to(device)
      z = mean + var*epsilon
      return z


    def forward(self, image, y = None):
        # mean and |variance
        mu, log_var = self.encode(image, y)
        std = torch.exp(0.5 * log_var)

        # reparametrization trick

        z_reparameterized = self.reparametrize(mu, std)

        # reconstruction
        image_reconstructed = self.decode(z_reparameterized, y)
        return mu, log_var, image_reconstructed