import torch
from torch import nn as nn
from torch.nn import functional as F


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    print(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Interpolate(nn.Module):
    def __init__(self, output_size, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.output_size = output_size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.output_size, mode=self.mode, align_corners=self.align_corners)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels=768, out_channels=None, readout_type="ignore", patch_size=16):
        super().__init__()

        if out_channels is None:
            out_channels = [96, 192, 384, 768]

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    bias=False,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                # x4
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                # x2
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                # x1
                nn.Identity(),
                # /2
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, activation, use_norm, stride=1, dilation=1, norm_type="bn"):
        super().__init__()

        self.use_norm = use_norm
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=not self.use_norm,
        )

        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            bias=not self.use_norm,
        )

        if self.use_norm:
            if norm_type == "bn":
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.bn2 = nn.BatchNorm2d(in_channels)
            elif norm_type == "gn":
                self.bn1 = nn.GroupNorm(4, in_channels)
                self.bn2 = nn.GroupNorm(4, in_channels)
            else:
                raise NotImplementedError

        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.use_norm:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.use_norm:
            out = self.bn2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, expand=False, align_corners=True, norm_type="bn"):
        super().__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)

        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels,
            activation=nn.ReLU(False),
            use_norm=True,
            norm_type=norm_type,
        )
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels,
            activation=nn.ReLU(False),
            use_norm=True,
            norm_type=norm_type,
        )

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x
