import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.checkpoint import checkpoint
from utils import utils
from .TFA import TFA


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return x * self.sigmoid(attn)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class PhysNet_padding_ED_peak(nn.Module):
    """
    Changes vs original:
      * use_tfa (default True): set False to drop the SpyNet/TFA motion branch.
        For controlled, low-motion captures (subject sitting still) TFA mostly
        adds parameters + overfitting; ablating it is often faster AND more
        accurate. NOTE: toggling this changes the parameter set, so a checkpoint
        trained with one setting cannot be strictly loaded into the other.
      * use_checkpoint (default False): gradient-checkpoint the heavy 3D conv
        blocks + TFA to fit frames=160 (or longer) on an 8 GB GPU. Only active
        in training mode.
      * tfa_blocks: shrink the TFA trunks for small datasets.
    """
    def __init__(self, frames=160, device_ids=[0], hidden_layer=128,
                 use_tfa=True, use_checkpoint=False, tfa_blocks=15,
                 spynet_path='./weights/spynet_.pth', freeze_spynet=True):
        super(PhysNet_padding_ED_peak, self).__init__()
        imnet_in_dim = 16 * 9 + 2
        self.device_ids = device_ids
        self.hidden_layer = hidden_layer
        self.use_tfa = use_tfa
        self.use_checkpoint = use_checkpoint

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.ConvBlock1_1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True))
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True))
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True))

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU())
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, [4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64), nn.ELU())

        self.spatial_attention = SpatialAttention()
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.mlp_x = MLP(in_dim=imnet_in_dim, out_dim=16, hidden_list=[self.hidden_layer, self.hidden_layer])
        self.mlp_y = MLP(in_dim=imnet_in_dim, out_dim=16, hidden_list=[self.hidden_layer, self.hidden_layer])

        if self.use_tfa:
            self.tfa = TFA(num_feat=16, num_block=tfa_blocks,
                           spynet_path=spynet_path, freeze_spynet=freeze_spynet)

        self.register_buffer('_coord', utils.make_coord([64, 64]))
        _cell = torch.ones(64 * 64, 2)
        _cell[:, 0] *= 2.0 / 64
        _cell[:, 1] *= 2.0 / 64
        self.register_buffer('_cell', _cell)

    def _ckpt(self, module, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x, y):
        x_visual = x

        if self.use_tfa:
            z_x = self._ckpt(self.tfa, x.permute(0, 2, 1, 3, 4).contiguous())
            z_y = self._ckpt(self.tfa, y.permute(0, 2, 1, 3, 4).contiguous())

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        y = self.ConvBlock1_1(y)
        y = self.MaxpoolSpa(y)

        x = self.pfe_heart(x, 64, self.mlp_x)
        y = self.pfe_heart(y, 64, self.mlp_y)

        if self.use_tfa:
            x = x + z_x
            y = y + z_y

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        y = y.permute(0, 2, 1, 3, 4).contiguous()
        x = torch.cat([x, y], dim=0)

        x = self._ckpt(self.ConvBlock2, x)
        x_visual6464 = self._ckpt(self.ConvBlock3, x)
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self._ckpt(self.ConvBlock4, x)
        x_visual3232 = self._ckpt(self.ConvBlock5, x)
        x = self.MaxpoolSpaTem(x_visual3232)

        x = self._ckpt(self.ConvBlock6, x)
        x_visual1616 = self._ckpt(self.ConvBlock7, x)
        x = self.MaxpoolSpa(x_visual1616)

        x = self._ckpt(self.ConvBlock8, x)
        x = self._ckpt(self.ConvBlock9, x)
        x = self.upsample(x)
        x = self.upsample2(x)

        x = self.spatial_attention(x)
        x = self.poolspa(x)
        x = self.ConvBlock10(x)

        rPPG_peak = x.squeeze(-1).squeeze(-1)
        return rPPG_peak, x_visual, x_visual3232, x_visual1616

    def pfe_heart(self, x, output_size, imnet):
        batch, channel, length, width, height = x.shape
        coord = self._coord.expand(batch * length, -1, -1)
        cell = self._cell.expand(batch * length, -1, -1)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, channel, width, height)
        ret = self.query_rgb(x, coord, cell=cell, imnet=imnet)
        ret = ret.permute(0, 2, 1).contiguous()
        ret = ret.view(batch * length, channel, output_size, output_size)
        ret = ret.view(batch, length, channel, output_size, output_size)
        return ret

    def query_rgb(self, x, coord, cell=None, imnet=None):
        feat = x
        feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
        coord_ = coord.clone()
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]
        inp = torch.cat([q_feat, rel_cell], dim=-1)
        bs, q = coord.shape[:2]
        pred = imnet(inp.view(bs * q, -1)).view(bs, q, -1)
        return pred