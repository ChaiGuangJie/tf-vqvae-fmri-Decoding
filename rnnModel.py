import torch
import torch.nn as nn

class GRU_Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, time_step):
        super().__init__()
        self.cell = nn.GRUCell(input_size, hidden_size).cuda()
        # self.decoder_cell = nn.GRUCell(decoder_input_size, hidden_size).cuda()
        # self.z = nn.Linear(hidden_size, encoder_input_size).cuda()
        self.hidden_size = hidden_size
        self.time_step = time_step

    def encoder(self, x):
        hx = torch.full((x.shape[0], self.hidden_size), 0.1).cuda()
        for i in range(self.time_step):
            hx = self.encoder_cell(x[:, i, :], hx)
        return hx

    def decoder(self, hx):
        z_list = []
        for i in range(self.time_step):
            hx = self.decoder_cell(hx, hx)
            z = self.z(hx) #torch.tanh(self.z(hx))
            z_list.append(z)

        return torch.stack(z_list).permute(1, 0, 2)  # 这样交换维度可以？

    def forward(self, x):
        hx = self.encoder(x)
        return self.decoder(hx)