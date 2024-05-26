import torch.nn as nn
import torch.nn.functional as F
import torch

# Implementation of a vanilla VAE which is not conditional in nature
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.img_2hidden = nn.Linear(self.input_dim, self.h_dim)
        self.hidden_2mu = nn.Linear(self.h_dim, self.z_dim)
        self.hidden_2sigma = nn.Linear(self.h_dim, self.z_dim)

        self.z_2hidden = nn.Linear(self.z_dim, self.h_dim)
        self.hidden_2img = nn.Linear(self.h_dim, self.input_dim)


    def encoder(self, x):
        # encoding function outputs p(z|x)
        image_hidden = F.relu(self.img_2hidden(x))
        mu, sigma = self.hidden_2mu(image_hidden), self.hidden_2sigma(image_hidden)
        return mu, sigma


    def decoder(self, z):
        #decoding function outputs p(x|z)
        image_hidden = F.relu(self.z_2hidden(z))
        image_reconstructed = self.hidden_2img(image_hidden)
        return image_reconstructed
        
    

    def forward(self, image):
        # mean and variance 
        mu, sigma = self.encoder(image)

        # reparametrization trick
        z_reparameterized = mu + sigma * torch.randn_like(sigma)

        # reconstruction
        image_reconstructed = self.decoder(z_reparameterized)
        return mu, sigma, image_reconstructed
