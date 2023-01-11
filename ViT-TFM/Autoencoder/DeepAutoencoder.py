import torch
from torch import nn

class DeepLinearAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(32 * 32 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32 * 32 * 2),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        pred = torch.unflatten(decoded, dim=1, sizes=(2, 32, 32))
        return pred


class DeepConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.ReLU(True)
        )
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 64, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 32)
            #nn.Linear(32, 2),
            #nn.ReLU(True)
        )
        self.decoder_lin = nn.Sequential(
            #nn.Linear(2, 32),
            #nn.ReLU(True),
            nn.Linear(32, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, (2 * 2 * 256)),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 2, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        #print(f"shape of x before anything {x.shape}")
        x = self.encoder_cnn(x)
        #print(f"shape of x after encoding {x.shape}")
        x = self.flatten(x)
        #print(f"shape of x after flattening {x.shape}")
        x = self.encoder_lin(x)
        #print(f"shape of x after linear encoding {x.shape}")
        x = self.decoder_lin(x)
        #print(f"shape of x after linear decoding {x.shape}")
        x = self.unflatten(x)
        #print(f"shape of x after unflattening {x.shape}")
        x = self.decoder_conv(x)
        #print(f"shape of x after decoding {x.shape}")

        return x