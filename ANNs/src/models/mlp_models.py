import torch.nn as nn
from torchviz import make_dot
import torch
from torchinfo import summary

class DECODER_1(nn.Module):
    def __init__(self, X_size, mlp_layers, decoder_channels):
        super(DECODER_1, self).__init__()

        # Construction dynamique du MLP
        mlp = []
        in_size = X_size
        for out_size in mlp_layers:
            mlp.append(nn.Linear(in_size, out_size))
            mlp.append(nn.ReLU())
            in_size = out_size
        self.mlp = nn.Sequential(*mlp)

        # Taille finale de sortie MLP (dernier élément de mlp_layers)
        final_mlp_output = mlp_layers[-1]
        sqrt_size = int(final_mlp_output ** 0.5)
        assert sqrt_size * sqrt_size == final_mlp_output, \
            "La sortie du MLP doit être carrée (ex: 9, 16, 25, etc.)"
        
        self.output_shape = (1, sqrt_size, sqrt_size)

        # Construction dynamique du décodeur
        decoder = []
        in_channels = self.output_shape[0]
        for out_channels in decoder_channels[:-1]:
            decoder.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            decoder.append(nn.ReLU())
            decoder.append(nn.Upsample(scale_factor=2))
            in_channels = out_channels
        # Dernière couche sans Upsample
        decoder.append(nn.Conv2d(in_channels, decoder_channels[-1], kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(-1, *self.output_shape)  # reshape en (batch, 1, H, W)
        x = self.decoder(x)
        return x


class INV_CNN_1(nn.Module):
    def __init__(self, X_size):
        super(INV_CNN_1, self).__init__()

        # MLP pour transformer l'entrée de taille 5 en taille 9
        self.mlp = nn.Sequential(
            nn.Linear(X_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
            nn.ReLU()
        )

        # Décodeur convolutionnel
        self.decoder = nn.Sequential(
            # Entrée : (batch_size, 1, 3, 3)
            nn.ConvTranspose2d(1, 16, kernel_size=2, stride=1, padding=0),  
            nn.ReLU(),

            nn.ConvTranspose2d(16, 32, kernel_size=2, stride=1, padding=0),  
            nn.ReLU(),

            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),

            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=1, padding=0),  # -> (batch, 1, 24, 24)
        )

    def forward(self, x):
        x = self.mlp(x)                 
        x = x.view(-1, 1, 3, 3)          
        x = self.decoder(x)             
        return x


def get_model(name, X_size, mlp_layers = "Default", decoder_channels = "Default"):
    if name == "DECODER_1":
        if mlp_layers == "Default" :
            mlp_layers =[32, 64, 32, 9]
        if decoder_channels == "Default" :
            decoder_channels=[32, 64, 32, 1]
        return DECODER_1(X_size, mlp_layers, decoder_channels)
    
    elif name == "INV_CNN_1":
        return INV_CNN_1(X_size)
    else:
        raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    decoder_1 = get_model("DECODER_1", X_size=6, mlp_layers=[112, 80, 128, 48, 9], decoder_channels=[112, 112, 96, 1])
    
    summary(decoder_1, input_size=(1, 6)) 

