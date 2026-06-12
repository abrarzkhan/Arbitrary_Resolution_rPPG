import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import math


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, 7, 1, 3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])
        if load_path:
            if os.path.exists(load_path):
                self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])
            else:
                print(f"[TFA] WARNING: SpyNet weights not found at '{load_path}'. "
                      f"Using random (frozen) flow -- TFA will be weak. "
                      f"Either provide weights or train with --use_tfa 0.")
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        # input is [-1, 1] YUV -> convert to [0,1] RGB (BT.601) then ImageNet-normalise
        yuv_01 = (tensor_input + 1.0) / 2.0
        y = yuv_01[:, 0:1]
        cb = yuv_01[:, 1:2] - 0.5
        cr = yuv_01[:, 2:3] - 0.5
        r = torch.clamp(y + 1.402 * cr, 0, 1)
        g = torch.clamp(y - 0.344136 * cb - 0.714136 * cr, 0, 1)
        b = torch.clamp(y + 1.772 * cb, 0, 1)
        rgb = torch.cat([r, g, b], dim=1)
        return (rgb - self.mean) / self.std

    def process(self, ref, supp):
        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        for level in range(3):
            ref.insert(0, F.avg_pool2d(ref[0], 2, 2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(supp[0], 2, 2, count_include_pad=False))
        flow = ref[0].new_zeros(
            [ref[0].size(0), 2,
             int(math.floor(ref[0].size(2) / 2.0)),
             int(math.floor(ref[0].size(3) / 2.0))])
        for level in range(len(ref)):
            upsampled_flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(upsampled_flow, [0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(upsampled_flow, [0, 1, 0, 0], mode='replicate')
            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(supp[level], upsampled_flow.permute(0, 2, 3, 1),
                          interp_mode='bilinear', padding_mode='border'),
                upsampled_flow], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()
        h, w = ref.size(2), ref.size(3)
        flow = F.interpolate(self.process(ref, supp), size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= float(w) / float(16)
        flow[:, 1, :, :] *= float(h) / float(16)
        return flow


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


def make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


class ConvResidualBlocks(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class TFA(nn.Module):
    """
    Temporal Feature Alignment.

    Changes vs original:
      * `freeze_spynet` (default True): the pretrained flow net is frozen and run
        under no_grad. On a small dataset, fine-tuning SpyNet just overfits and
        wastes memory/compute. The warping field still backprops into features.
      * `num_block` is now configurable so the trunks can be shrunk for small data.
    """
    def __init__(self, num_feat=16, num_block=15, spynet_path=None, freeze_spynet=True):
        super().__init__()
        self.num_feat = num_feat
        self.spynet = SpyNet(spynet_path)
        self.freeze_spynet = freeze_spynet
        if freeze_spynet:
            for p in self.spynet.parameters():
                p.requires_grad = False
            self.spynet.eval()
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.video_upsample_1 = nn.Upsample(size=[3, 16, 16], mode='trilinear', align_corners=False)
        self.video_upsample_2 = nn.Upsample(size=[16, 64, 64], mode='trilinear', align_corners=False)
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_spynet:
            self.spynet.eval()  # keep frozen subnet in eval regardless
        return self

    def _spynet_pair(self, x_1, x_2, b, n, h, w):
        return self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)

    def get_flow(self, x):
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        if self.freeze_spynet:
            with torch.no_grad():
                flows_backward = self._spynet_pair(x_1, x_2, b, n, h, w)
                flows_forward = self._spynet_pair(x_2, x_1, b, n, h, w)
        else:
            flows_backward = self._spynet_pair(x_1, x_2, b, n, h, w)
            flows_forward = self._spynet_pair(x_2, x_1, b, n, h, w)
        return flows_forward, flows_backward

    def forward(self, x):
        x = self.video_upsample_1(x)
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out_l[i] = out
        out_put = torch.stack(out_l, dim=1)
        out_put = self.video_upsample_2(out_put)
        return out_put