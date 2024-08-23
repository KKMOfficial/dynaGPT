import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from quantizer import get_quantizer

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.en_skip_conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=183, stride=1, padding=0)

        # Vector Quantization
        self.quantizer = get_quantizer(num_embeddings, embedding_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=7, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=input_dim, out_channels=input_dim, kernel_size=16, stride=2, padding=1),
        )

        self.dc_skip_conv = nn.ConvTranspose1d(in_channels=embedding_dim, out_channels=input_dim, kernel_size=183, stride=1, padding=0)

    def encode(self, x):
        z_e = self.en_skip_conv(x) + self.encoder(x)
        z_e = z_e.permute(0, 2, 1)  # Change shape for quantization
        return z_e

    def decode(self, z_q):
        return self.dc_skip_conv(z_q) + self.decoder(z_q)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, q_loss ,indices = self.quantizer(z_e)
        x_recon = self.decode(z_q.permute((0,2,1)))
        return x_recon, q_loss, indices
    
    def decode_indices(self, indices):
        z_q = torch.index_select(self.quantizer.embedding.weight, 0, indices.view(-1)).reshape((indices.shape[0],indices.shape[1],-1))
        x_recon = self.decode(z_q.permute((0,2,1)))
        return x_recon



def train_vqvae(model, train_data_loader, validation_data_loader, num_epochs, learning_rate, device, check_point_path, model_path, commitment_cost=0.25):
    best_validation_loss = -1
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_history = []

    try:
      for epoch in range(num_epochs):
          model.train()
          total_loss = 0

          for bidex,batch in enumerate(train_data_loader):
              batch = batch.to(device)
              optimizer.zero_grad()
              x_recon, quantize_loss, indices = model(batch)
              recon_loss      = F.mse_loss(x_recon, batch)
              loss = recon_loss + quantize_loss
              loss.backward()
              optimizer.step()
              total_loss += recon_loss.item()

              avg_loss = total_loss / len(train_data_loader)
              print(f'Batch Index [{bidex + 1}/{num_epochs}], Partial Loss: {recon_loss.item():.4f}')

              train_history.append(avg_loss)

          avg_loss = total_loss / len(train_data_loader)
          print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

          model.eval()
          total_val_loss = 0
          with torch.no_grad():
              for batch in validation_data_loader:
                  batch = batch.to(device)
                  x_recon, quantize_loss, indices = model(batch)
                  recon_loss      = F.mse_loss(x_recon, batch)
                  loss = recon_loss + quantize_loss
                  total_val_loss += recon_loss.item()
                  print(f'Partial Validation Loss: {recon_loss.item():.4f}')  

              avg_val_loss = total_val_loss / len(validation_data_loader)
              print(f'Validation Loss: {avg_val_loss:.4f}')

              if avg_val_loss < best_validation_loss:
                  print(f'saving model...')
                  torch.save(model.state_dict(), model_path)
                  best_validation_loss = avg_val_loss
              else:
                  print(f'saving check-point...')
                  torch.save(model.state_dict(), check_point_path)
              # log information
              # train_history.append([avg_loss, avg_val_loss])
    except:
        print(f'saving model...')
        torch.save(model.state_dict(), model_path)
        return train_history
    return train_history