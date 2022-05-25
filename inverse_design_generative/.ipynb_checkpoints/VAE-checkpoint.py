import torch
from torch import nn
from torch.backends import cudnn
cudnn.benchmark = True

# -----------------------------------------

class VAE_encode(nn.Module):

    def __init__(self, composition_len, layer_1d, latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAE_encode, self).__init__()

        # Reduce dimension upto second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(composition_len, layer_1d),
            nn.ReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_1d, latent_dimension)

        # Latent space variance 
        self.encode_log_var = nn.Linear(layer_1d, latent_dimension)

    def reparameterize(self, mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass through the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class VAE_decode(nn.Module):

    def __init__(self, latent_dimension, layer_1d, composition_len):
        """
        Through Decoder
        """
        super(VAE_decode, self).__init__()

        self.decode_nn = nn.Sequential(
            nn.Linear(latent_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, composition_len),
            nn.Softmax(1),
        )

    def forward(self, z):
        """
        A forward pass through the entire model.
        """
        # Decode
        decoded = self.decode_nn(z)

        return decoded
