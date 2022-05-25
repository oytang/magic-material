
import numpy as np
import torch
from torch import nn
from .VAE import VAE_encode, VAE_decode
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda. is_available() else "cpu")


class inverse_design_cvae():

    def __init__(self, elements, properties, epoch
                 , elements_len=12, properties_len=6, layer_en=6, layer_de=6, latent_dimension=10,
                 ):
        self.EPOCH = epoch
        self.COMPOSITION = elements_len
        self.PROPERTY = properties_len
        self.X, self.Y = elements, properties
        self.LA_E, self.LA_D, self.LA_L = layer_en, layer_de, latent_dimension

        self.model, self.model_encode, self.model_decode = None, None, None

    def run_cvae(self):
        self.model_encode = VAE_encode(self.COMPOSITION + self.PROPERTY, self.LA_E, self.LA_L).to(device)
        self.model_decode = VAE_decode(self.LA_L + self.PROPERTY, self.LA_D, self.COMPOSITION).to(device)
        optimizer_encoder = torch.optim.Adam(self.model_encode.parameters(), lr=1e-3)
        optimizer_decoder = torch.optim.Adam(self.model_decode.parameters(), lr=1e-3)
        KLD_alpha = 1

        results = {
            'loss': [],
            'kld': [],
            'latent_points': []
        }
        for i in tqdm(range(self.EPOCH)):
            loss, recon_loss, kld = 0., 0., 0.
            input_ = torch.concat([torch.tensor(self.X).to(device).float(), torch.tensor(self.Y).to(device).float()], 1)
            latent_points, mus, log_vars = self.model_encode(input_)
            kld += -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
            latent_ = torch.concat([latent_points, torch.tensor(self.Y).to(device).float()], 1)
            decoded = self.model_decode(latent_)
            recon_loss += torch.nn.CrossEntropyLoss()(torch.tensor(self.X).to(device).float(), decoded)
            loss += recon_loss + KLD_alpha * kld
            results['loss'].append(loss.cpu().detach().numpy())
            results['kld'].append(kld.cpu().detach().numpy())
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.model_decode.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()
        return results, self.model_encode, self.model_decode

    def run_design(self, target):
        with torch.no_grad():
            exa = np.random.randn(self.LA_L, 1)
            pre_ = torch.concat([torch.tensor(exa).T.to(device), torch.tensor(target).reshape(1, -1).to(device)], 1)
            res = self.model_decode(pre_.to(device).float()).to("cpu").detach().numpy()
        return res.squeeze()