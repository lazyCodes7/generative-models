from  .diffusion import Diffusion
import torch
import torch.nn as nn

class ConditionalDiffusion(Diffusion):
    def __init__(
        self,
        device,
        beta_start=1e-4, 
        beta_end=0.02,
        img_size = 32,
        timesteps = 1000
    ):
        super(ConditionalDiffusion, self).__init__(
            device,
            beta_start,
            beta_end,
            img_size,
            timesteps
        )
    
    def sample_initial_image(self, model, batch_size, labels):
        with torch.no_grad():
          model.eval()
          x = torch.randn((batch_size, 3, self.img_size, self.img_size)).to(self.device)
          for timestep in range(self.timesteps-1, -1, -1):
              t = (torch.ones(batch_size) * timestep).long().to(self.device)
              alpha = self.alpha[t][:, None, None, None]
              alpha_bar = self.alpha_bar[t][:, None, None, None]
              predicted_noise = model(x, t, labels)
              sigma = self.noise_scheduler[t][:, None, None, None]
              if timestep > 1:
                  noise = torch.randn_like(x)
              else:
                  noise = torch.zeros_like(x)
              x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(sigma) * noise
        model.train()
        x = x.clamp(-1, 1)
        x = self.invTrans(x)
        return x