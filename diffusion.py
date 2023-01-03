class Diffusion(nn.Module):
    def __init__(
        self,
        beta_start,
        beta_end,
        timesteps = 1000
    ):
        self.timesteps = timesteps
        self.noise_scheduler = self.init_noise_scheduler()
        self.alpha = 1 - self.noise_scheduler
        self.alpha_bar = torch.cumprod(self.noise_scheduler)

    def sample_timesteps(self, batch_size):
        t = 0
        while(t>1):
            t = torch.randint(0, self.timesteps, size = (batch_size, ))
        return t
    
    def init_noise_scheduler(self):
        return torch.linspace(beta_start, beta_end, self.timesteps)
    
    def sample_noise_image(self, image, timestep):
        sqrt_alpha_bar = torch.sqrt(self.alpha[t])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])
        eps = torch.randn_like(image)
        noise_image = sqrt_alpha_bar*image + sqrt_one_minus_alpha_bar*eps
        return noise_image
    
    def sample_initial_image(self, model, timestep):
        model.eval()
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        for t in range(timestep, -1, -1):
            alpha_bar = self.alpha[t]
            predicted_noise = model(x, t)
            sigma = self.noise_scheduler[t]
            if t > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha_bar) * (x - ((1 - alpha_bar) / (torch.sqrt(1 - alpha_bar))) * predicted_noise) + torch.sqrt(sigma) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
            




