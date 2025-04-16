import torch
import torch.nn as nn
import torch.nn.functional as F

class MixerBlock(nn.Module):
    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x

class TriU(nn.Module):
    def __init__(self, time_step: int):
        super().__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList([
            nn.Linear(i + 1, 1) for i in range(time_step)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x

class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y

class NoGraphMixer(nn.Module):
    def __init__(self, stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x

class MultTime2dMixer(nn.Module):
    def __init__(self, time_step: int, channel: int, scale_dim: int = 8):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)

class StockMixer(nn.Module):
    def __init__(
        self,
        stocks: int,
        time_steps: int,
        channels: int,
        market: int = 20,
        scale: int = 8
    ):
        super().__init__()
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, time_steps, channels]
        x = inputs.permute(0, 2, 1)  # [batch_size, channels, time_steps]
        x = self.conv(x)  # [batch_size, channels, time_steps/2]
        x = x.permute(0, 2, 1)  # [batch_size, time_steps/2, channels]
        
        y = self.mixer(inputs, x)  # [batch_size, time_steps*2 + scale, channels]
        y = self.channel_fc(y).squeeze(-1)  # [batch_size, time_steps*2 + scale]

        z = self.stock_mixer(y)  # [batch_size, time_steps*2 + scale]
        y = self.time_fc(y)  # [batch_size, 1]
        z = self.time_fc_(z)  # [batch_size, 1]
        
        return y + z  # [batch_size, 1]
