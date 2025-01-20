import math
import torch
from torch import nn as nn

from crisp.models.resnet import resnet34, resnet50, resnet101
from crisp.models.dpt import Interpolate, ReassembleBlocks, FeatureFusionBlock
from torchvision.models.resnet import BasicBlock, conv1x1


def _make_output_norm_op(output_normalization, shape_code_norm_scale):
    """Helper function for potentially normalizing the output shape latent code"""
    output_op = nn.Identity()
    if output_normalization is not None:
        if output_normalization == "sphere":
            output_op = lambda x: nn.functional.normalize(x, dim=1) * shape_code_norm_scale
        elif output_normalization == "max_norm":

            def max_norm(x):
                norm = torch.norm(x, dim=1, keepdim=True)
                if norm > shape_code_norm_scale:
                    return x / norm * shape_code_norm_scale
                else:
                    return x

            output_op = max_norm
    return output_op


class ReconsMappingModule_DPT(nn.Module):
    def __init__(
        self,
        vit_feature_dim,
        output_dim,
        output_normalization=None,
        shape_code_norm_scale=1,
        channels=64,
        post_process_channels=None,
        readout_type="ignore",
        patch_size=14,
        expand_channels=False,
        fusion_norm_type="gn",
        **kwargs
    ):
        super().__init__()

        if post_process_channels is None:
            post_process_channels = [48, 96, 192, 384]

        self.output_dim = output_dim
        self.channels = channels
        self.in_channels = vit_feature_dim
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(vit_feature_dim, post_process_channels, readout_type, patch_size)
        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel for i, channel in enumerate(post_process_channels)
        ]

        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(nn.Conv2d(channel, self.channels, kernel_size=3, padding=1, bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, norm_type=fusion_norm_type))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

        # final ResNet block layer
        norm = lambda c: nn.GroupNorm(4, c)
        ds = lambda inplanes, planes: nn.Sequential(
            conv1x1(inplanes, planes, 2),
            norm(planes),
        )

        planes = [64, 48, 24, 12]
        inplanes = [64, 64, 48, 24]
        self.final_resnet = nn.Sequential()
        for i in range(len(planes)):
            self.final_resnet.append(
                BasicBlock(
                    inplanes[i],
                    planes[i],
                    stride=2,
                    norm_layer=norm,
                    downsample=ds(inplanes[i], planes[i]),
                )
            )
        # flattened final feature map of size (15, 15)
        self.final_fc = nn.Linear(planes[-1] * int((240 / (2 ** len(planes))) ** 2), self.output_dim)

    def forward(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        # out = x8 size in H and W
        out = self.project(out)
        out = self.final_resnet(out)
        out = torch.flatten(out, 1)
        out = self.final_fc(out)
        return out


class ReconsMappingModule_ResNet34(nn.Module):
    def __init__(self, vit_feature_dim, output_dim, output_normalization=None, shape_code_norm_scale=1, **kwargs):
        super().__init__()
        # 4 layers of patch embedding + cls token
        self.in_channels = vit_feature_dim * 5
        self.output_dim = output_dim
        norm_layer = lambda x: nn.GroupNorm(32, x)
        self.net = resnet34(in_channels=self.in_channels, num_classes=output_dim, norm_layer=norm_layer)
        self.output_op = _make_output_norm_op(output_normalization, shape_code_norm_scale)

    def forward(self, x):
        # concat cls token with features
        with torch.no_grad():
            patch_input = torch.cat([patch for patch, _ in x], dim=1)
            last_cls = x[-1][1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_input.shape[-2], patch_input.shape[-1])
            patch_input = torch.cat([patch_input, last_cls], dim=1)
        return self.output_op(self.net(patch_input))


class ReconsMappingModule_ResNet101(nn.Module):
    def __init__(self, vit_feature_dim, output_dim, output_normalization=None, shape_code_norm_scale=1, **kwargs):
        super().__init__()
        # 4 layers of patch embedding + cls token
        self.in_channels = vit_feature_dim * 5
        self.output_dim = output_dim
        norm_layer = lambda x: nn.GroupNorm(32, x)
        self.net = resnet101(in_channels=self.in_channels, num_classes=output_dim, norm_layer=norm_layer)
        self.output_op = _make_output_norm_op(output_normalization, shape_code_norm_scale)

    def forward(self, x):
        # concat cls token with features
        with torch.no_grad():
            patch_input = torch.cat([patch for patch, _ in x], dim=1)
            last_cls = x[-1][1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_input.shape[-2], patch_input.shape[-1])
            patch_input = torch.cat([patch_input, last_cls], dim=1)
        return self.output_op(self.net(patch_input))


class ReconsMappingModule_ResNet50(nn.Module):
    def __init__(self, vit_feature_dim, output_dim, output_normalization=None, shape_code_norm_scale=1, **kwargs):
        super().__init__()
        # 4 layers of patch embedding + cls token
        self.in_channels = vit_feature_dim * 5
        self.output_dim = output_dim
        norm_layer = lambda x: nn.GroupNorm(32, x)
        self.net = resnet50(in_channels=self.in_channels, num_classes=output_dim, norm_layer=norm_layer)
        self.output_op = _make_output_norm_op(output_normalization, shape_code_norm_scale)

    def forward(self, x):
        # concat cls token with features
        with torch.no_grad():
            patch_input = torch.cat([patch for patch, _ in x], dim=1)
            last_cls = x[-1][1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, patch_input.shape[-2], patch_input.shape[-1])
            patch_input = torch.cat([patch_input, last_cls], dim=1)
        return self.output_op(self.net(patch_input))


class ReconsMappingModule_FFN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_layers, output_normalization=None, shape_code_norm_scale=1
    ):
        super().__init__()
        self.hidden_dim, self.output_dim, self.hidden_layers = hidden_dim, output_dim, hidden_layers
        modules = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)

        self.output_op = _make_output_norm_op(output_normalization, shape_code_norm_scale)

    def forward(self, x):
        with torch.no_grad():
            cls_feature = torch.cat([x[0][1], x[1][1], x[2][1], x[3][1], x[3][0].flatten(2).mean(dim=2)], dim=1)
        return self.output_op(self.net(cls_feature))
