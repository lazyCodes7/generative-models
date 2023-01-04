import torch
import torch.nn as nn
class Diffusion(nn.Module):
    def __init__(
        self,
        device,
        beta_start=1e-4, 
        beta_end=0.02,
        img_size = 32,
        timesteps = 1000
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.device = device
        self.noise_scheduler = self.init_noise_scheduler().to(self.device)
        self.alpha = 1 - self.noise_scheduler
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)

        self.img_size = img_size

    def sample_timesteps(self, batch_size):
        t = torch.randint(0, self.timesteps, size = (batch_size, ))
        return t
    
    def init_noise_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
    
    def sample_noise_image(self, image, timestep):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[timestep])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[timestep])[:, None, None, None]
        eps = torch.randn_like(image).to(self.device)
        noise_image = sqrt_alpha_bar*image + sqrt_one_minus_alpha_bar*eps
        return noise_image, eps
    
    def sample_initial_image(self, model, batch_size):
        with torch.no_grad():
          model.eval()
          x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
          for timestep in range(self.timesteps-1, -1, -1):
              t = (torch.ones(batch_size) * timestep).long().to(self.device)
              alpha = self.alpha[t][:, None, None, None]
              alpha_bar = self.alpha_bar[t][:, None, None, None]
              predicted_noise = model(x, t)
              sigma = self.noise_scheduler[t][:, None, None, None]
              if timestep > 1:
                  noise = torch.randn_like(x)
              else:
                  noise = torch.zeros_like(x)
              x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(sigma) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x