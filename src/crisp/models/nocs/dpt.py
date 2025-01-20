# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from crisp.models.dpt import Interpolate, ReassembleBlocks, FeatureFusionBlock


class NOCSBaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (List): Input channels.
        channels (int): Channels after modules, before conv_depth.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        loss_decode (dict): Config of decode loss.
            Default: dict(type='SigLoss').
        sampler (dict|None): The config of depth map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        min_nocs (int): Min depth in dataset setting.
            Default: 0.
        max_nocs (int): Max depth in dataset setting.
            Default: 1.
        classify (bool): Whether predict NOCS in a cls.-reg. manner.
            Default: False.
        n_bins (int): The number of bins used in cls. step.
            Default: 256.
        bins_strategy (str): The discrete strategy used in cls. step.
            Default: 'UD'.
        norm_strategy (str): The norm strategy on cls. probability
            distribution. Default: 'linear'
        scale_up (str): Whether predict NOCS in a scale-up manner.
            Default: False.
        separate_xyz (bool): Whether predict NOCS in a separate xyz manner.
    """

    def __init__(
        self,
        in_channels,
        channels=96,
        align_corners=False,
        min_nocs=0,
        max_nocs=1,
        classify=False,
        n_bins=256,
        bins_strategy="UD",
        norm_strategy="linear",
        final_act_type="relu",
        separate_xyz=False,
    ):
        super(NOCSBaseDecodeHead, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.min_nocs = min_nocs
        self.max_nocs = max_nocs
        self.classify = classify
        self.n_bins = n_bins
        self.separate_xyz = separate_xyz
        self.final_act_type = final_act_type

        if self.classify:
            assert self.separate_xyz, "AdaBins approach only support separate_xyz=True"
            assert bins_strategy in ["UD", "SID"], "Support bins_strategy: UD, SID"
            assert norm_strategy in ["linear", "softmax", "sigmoid"], "Support norm_strategy: linear, softmax, sigmoid"

            self.bins_strategy = bins_strategy
            self.norm_strategy = norm_strategy
            self.softmax = nn.Softmax(dim=1)
            self.conv_nocs_x = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
            self.conv_nocs_y = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
            self.conv_nocs_z = nn.Conv2d(channels, n_bins, kernel_size=3, padding=1, stride=1)
        else:
            # self.conv_nocs = nn.Conv2d(channels, 3, kernel_size=3, padding=1, stride=1)
            if self.separate_xyz:
                self.conv_nocs_x = None
                self.conv_nocs_y = None
                self.conv_nocs_z = None
            else:
                self.conv_nocs = None

        self.fp16_enabled = False

        if self.final_act_type == "sigmoid":
            self.final_act = self._sigmoid_final_act
        elif self.final_act_type == "relu":
            self.final_act = self._relu_final_act
        elif self.final_act_type == "identity":
            self.final_act = self._identity_final_act
        else:
            raise ValueError

    def extra_repr(self):
        """Extra repr."""
        s = f"align_corners={self.align_corners}"
        return s

    def _sigmoid_final_act(self, x):
        return F.sigmoid(self.conv_nocs(x)) * self.max_nocs

    def _relu_final_act(self, x):
        return F.relu(self.conv_nocs(x)) + self.min_nocs

    def _identity_final_act(self, x):
        return self.conv_nocs(x)

    def _cls_nocs_pred_helper(self, cv_net, feat):
        logit = cv_net(feat)

        if self.bins_strategy == "UD":
            bins = torch.linspace(self.min_nocs, self.max_nocs, self.n_bins, device=feat.device)
        elif self.bins_strategy == "SID":
            bins = torch.logspace(self.min_nocs, self.max_nocs, self.n_bins, device=feat.device)

        # following Adabins, default linear
        if self.norm_strategy == "linear":
            logit = torch.relu(logit)
            eps = 0.1
            logit = logit + eps
            logit = logit / logit.sum(dim=1, keepdim=True)
        elif self.norm_strategy == "softmax":
            logit = torch.softmax(logit, dim=1)
        elif self.norm_strategy == "sigmoid":
            logit = torch.sigmoid(logit)
            logit = logit / logit.sum(dim=1, keepdim=True)

        output = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(dim=1)
        return output

    def nocs_pred(self, feat):
        """Prediction each pixel."""
        if self.classify:
            # TODO: Not tested
            ox = self._cls_nocs_pred_helper(cv_net=self.conv_nocs_x, feat=feat)
            oy = self._cls_nocs_pred_helper(cv_net=self.conv_nocs_y, feat=feat)
            oz = self._cls_nocs_pred_helper(cv_net=self.conv_nocs_z, feat=feat)

            output = torch.cat((ox, oy, oz), dim=1)
        else:
            if self.separate_xyz:
                ox = self.final_act(feat)
                oy = self.final_act(feat)
                oz = self.final_act(feat)
                output = torch.cat((ox, oy, oz), dim=1)
            else:
                output = self.final_act(feat)

        return output


class NOCSHead(nn.Module):
    def __init__(self, features, output_size, output_dim=3, norm_type="none"):
        """Final head to predict NOCS
        Note that in the case with ViT of patch size 16,
        if the Interpolate block has a scale factor of 2,
        the final output will be recovered to the original input size.

        For the case of DinoV2, because it uses a patch size of 14,
        we need to instead interpolate to the exact output size to keep
        the output size the same as input size.
        Or use a scale factor of 1.75 (=14/8) (but this may cause floating point errors).
        """
        super().__init__()
        if norm_type == "none":
            self.head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(output_size=output_size, mode="bilinear", align_corners=True),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
            )
        elif norm_type == "gn":
            self.head = nn.Sequential(
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(output_size=output_size, mode="bilinear", align_corners=True),
                nn.GroupNorm(4, features // 2),
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        x = self.head(x)
        return x


class NOCSHead_XYZ(nn.Module):
    def __init__(self, features, output_size, norm_type="none"):
        super().__init__()
        self.headx = NOCSHead(features, output_size, output_dim=1, norm_type=norm_type)
        self.heady = NOCSHead(features, output_size, output_dim=1, norm_type=norm_type)
        self.headz = NOCSHead(features, output_size, output_dim=1, norm_type=norm_type)

    def forward(self, x):
        x1 = self.headx(x)
        x2 = self.heady(x)
        x3 = self.headz(x)
        x = torch.cat((x1, x2, x3), dim=1)
        return x


class DPTNOCSHead(NOCSBaseDecodeHead):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        embed_dims=768,
        post_process_channels=None,
        output_size=None,
        readout_type="ignore",
        patch_size=16,
        expand_channels=False,
        separate_xyz=False,
        fusion_norm_type="bn",
        nocs_head_norm_type="none",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels, readout_type, patch_size)

        if post_process_channels is None:
            post_process_channels = [96, 192, 384, 768]
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

        # final NOCS head
        assert not self.classify
        if separate_xyz:
            self.conv_nocs = NOCSHead_XYZ(self.channels, output_size, norm_type=nocs_head_norm_type)
        else:
            self.conv_nocs = NOCSHead(self.channels, output_size, norm_type=nocs_head_norm_type)

    def forward(self, inputs, mask, resize_op=nn.Identity()):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        # out = x8 size in H and W
        out = self.project(out)
        out = self.nocs_pred(out)
        out = resize_op(out) * mask
        return out
