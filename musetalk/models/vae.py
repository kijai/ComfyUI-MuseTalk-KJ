from diffusers import AutoencoderKL
import torch

class VAE():
    """
    VAE (Variational Autoencoder) class for image processing.
    """

    def __init__(self, config, sd, dtype=torch.bfloat16):
        """
        Initialize the VAE instance.

        """
        self.vae = AutoencoderKL(**config)
        self.vae.load_state_dict(sd, strict=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)
        self.vae = self.vae.to(dtype=dtype)
        self.scaling_factor = self.vae.config.scaling_factor

    def encode_latents(self,image):
        """
        Encode an image into latent variables.

        :param image: The image tensor to encode.
        :return: The encoded latent variables.
        """
        with torch.no_grad():
            init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents
    
    def decode_latents(self, latents):
        """
        Decode latent variables back into an image.
        :param latents: The latent variables to decode.
        :return: A NumPy array representing the decoded image.
        """
        latents = (1/  self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image