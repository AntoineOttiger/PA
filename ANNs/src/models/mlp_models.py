import torch.nn as nn

class DECODER_1(nn.Module):
    def __init__(self, X_size):
        super(DECODER_1, self).__init__()

        # MLP pour transformer l'entrée de taille 5 en taille 9
        self.mlp = nn.Sequential(
            nn.Linear(X_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 9),
            nn.ReLU()
        )

        # Décodeur convolutionnel
        self.decoder = nn.Sequential(
            # Entrée : (batch_size, 1, 3, 3)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),

        )

    def forward(self, x):
        x = self.mlp(x)                  # (batch, 9)
        x = x.view(-1, 1, 3, 3)          # (batch, 1, 3, 3)
        x = self.decoder(x)             # (batch, 1, 24, 24)
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


def get_model(name, X_size):
    if name == "DECODER_1":
        return DECODER_1(X_size)
    elif name == "INV_CNN_1":
        return INV_CNN_1(X_size)
    else:
        raise ValueError(f"Unknown model name: {name}")
