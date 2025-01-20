from functools import partial

import torch.nn as nn
import torch
from torch.nn.modules.module import T

# project local imports
from crisp.models.nocs.basic import NOCSModule, NOCSModule_xyz
from crisp.models.nocs.dpt import DPTNOCSHead
from crisp.models.shape.decoder import ReconsModule_Film
from crisp.models.pspnet import PSPNet
from crisp.models.shape.encoder import (
    ReconsMappingModule_ResNet34,
    ReconsMappingModule_ResNet50,
    ReconsMappingModule_ResNet101,
    ReconsMappingModule_FFN,
    ReconsMappingModule_DPT,
)


@torch.no_grad()
def apply_lipschitz_constraint(m, lipschitz_constraint_type, lambda_):
    if type(m) == nn.Linear:
        if lipschitz_constraint_type == "linf":
            m.weight.data = inf_norm_lipschitz_constraint(m.weight.data, lambda_)
        elif lipschitz_constraint_type == "l2":
            m.weight.data = two_norm_linear_lipschitz_constraint(m.weight.data, lambda_)
        elif lipschitz_constraint_type == "l1":
            m.weight.data = one_norm_lipschitz_constraint(m.weight.data, lambda_)
        else:
            raise NotImplementedError


def inf_norm_lipschitz_constraint(w, lambda_):
    """L-infinity lipschitz constraint
    https://arxiv.org/pdf/1804.04368

    L-infinity norm of matrix: maximum absolute row sum of the matrix (sum over the columns).
    """
    axes = 1
    if len(w.shape) == 4:
        # For conv2d layer
        axes = [1, 2, 3]
    norm = torch.max(torch.sum(torch.abs(w), dim=axes, keepdim=False))
    return w * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / lambda_))


def one_norm_lipschitz_constraint(w, lambda_):
    """1-norm lipschitz constraint
    https://arxiv.org/pdf/1804.04368

    One norm of matrix: maximum absolute column sum of the matrix (sum over the rows).
    """
    axes = 0
    if len(w.shape) == 4:
        # For conv2d layer
        axes = [0, 2, 3]
    norm = torch.max(torch.sum(torch.abs(w), dim=axes, keepdim=False))
    return w * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / lambda_))


def two_norm_linear_lipschitz_constraint(w, lambda_, iterations=1):
    """2-norm lipschitz constraint
    https://arxiv.org/pdf/1804.04368
    """
    x = torch.randn((w.shape[1], 1), device=w.device)
    for i in range(iterations):
        x_p = w @ x
        x = torch.transpose(w, 0, 1) @ x_p
    norm = torch.sqrt(torch.sum(torch.square(w @ x)) / torch.sum(torch.square(x)))

    return w * (1.0 / torch.maximum(torch.tensor(1.0, device=norm.device), norm / lambda_))


def two_norm_conv2d_lipschitz_constraint(w, lambda_, iterations=1):
    """2-norm lipschitz constraint
    https://arxiv.org/pdf/1804.04368
    """
    """
    if len(w.shape) == 4:
        x = K.random_normal_variable(shape=(1,) + self.in_shape[1:3] + (self.in_shape[0],), mean=0, scale=1)

        for i in range(0, self.iterations): 
            x_p = K.conv2d(x, w, strides=self.stride, padding=self.padding)
            x = K.conv2d_transpose(x_p, w, x.shape, strides=self.stride, padding=self.padding)

        Wx = K.conv2d(x, w, strides=self.stride, padding=self.padding)
        norm = K.sqrt(K.sum(K.pow(Wx, 2.0)) / K.sum(K.pow(x, 2.0)))

    return w * (1.0 / K.maximum(1.0, norm / self.max_k))
    """
    raise NotImplementedError


class FPNModule(nn.Module):
    def __init__(self, in_channels, out_channels, scales, img_size):
        super().__init__()
        self.fpn = []
        self.img_size = img_size

        for scale in scales:
            if scale > 0:
                layer_i = [
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, device="cuda"),
                    nn.BatchNorm2d(out_channels, device="cuda"),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                ] * scale
            else:
                raise NotImplementedError
            layer_i = nn.Sequential(*layer_i)
            self.fpn.append(layer_i)

    def forward(self, feature_map: list, mask):
        fpn_map = 0

        for i, layer_feature in enumerate(feature_map):
            layer_i = self.fpn[i]
            fpn_map += layer_i(layer_feature)

        fpn_map = nn.functional.interpolate(fpn_map, self.img_size, mode="bilinear")

        return fpn_map


class JointShapePoseNetwork(nn.Module):
    def __init__(
        self,
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        recons_adaptor_hidden_dim=512,
        recons_adaptor_hidden_layers=2,
        recons_modulate_last_layer=False,
        local_backbone_model_path=None,
        lateral_layers_type="last_four",
        backbone_model="resnet34",
        backbone_input_res=(420, 420),
        freeze_pretrained_weights=False,
        nonlinearity="sine",
        normalization_type="weight",
        nocs_network_type="simple",
        nocs_channels=256,
        recons_encoder_type="ffn",
        normalize_shape_code=None,
        recons_shape_code_norm_scale=1,
    ):
        super().__init__()

        # Backbone network
        print(f"Backbone model specified as: {backbone_model}")
        self.backbone_model = backbone_model
        self.lateral_layers_type = lateral_layers_type
        self.backbone_input_res = backbone_input_res
        self.nocs_network_type = nocs_network_type
        self.recons_encoder_type = recons_encoder_type
        self.recons_modulate_last_layer = recons_modulate_last_layer

        if "dinov2" in backbone_model:
            seg_type = "ms"  # multiscale
            assert backbone_model in [
                "dinov2_vits14",
                "dinov2_vitb14",
                "dinov2_vitl14",
                "dinov2_vitg14",
                "dinov2_vits14_reg",
                "dinov2_vitb14_reg",
                "dinov2_vitl14_reg",
                "dinov2_vitg14_reg",
            ]
            assert seg_type in ["ms", "linear"]

            if local_backbone_model_path is None:
                self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_model)
            else:
                self.backbone = torch.hub.load(
                    local_backbone_model_path, backbone_model, source="local", force_reload=True
                )

            # mapping network to latent code for shape reconstruction
            if self.lateral_layers_type == "last_four":
                self.lateral_layers = 4
                print(f"Lateral layers to use: last four.")
            elif self.lateral_layers_type == "spaced":
                self.lateral_layers = self._get_lateral_layer_indices(backbone_model)
                print(f"Lateral layers to use: {self.lateral_layers}")
            else:
                raise NotImplementedError

            # note: forward_helper return two objects: nocs_features, shape_code
            # nocs_features may be a list (for dpt) or a tensor (for other)
            # shape_code is a tensor
            if self.nocs_network_type == "fpn":
                self.forward_helper = self.dino_fpn_forward_helper
            elif "dpt" in self.nocs_network_type:
                self.forward_helper = self.dino_dpt_forward_helper
            else:
                self.forward_helper = self.dino_multiscale_foward_helper

        else:
            self.backbone = PSPNet(bins=(1, 2, 3, 6), backbone=backbone_model, use_pretrained_backbone=True)
            self.recons_backbone_adaptor = nn.Linear(32, 10)
            self.forward_helper = self.resnet_forward_helper

        # NOCS module
        if self.nocs_network_type == "simple":
            self.nocs_backbone_adaptor = nn.Sequential(
                nn.Linear(self.backbone.embed_dim * self.lateral_layers, 384), nn.ReLU(), nn.Linear(384, 32)
            )
            self.nocs_net = NOCSModule()
        elif self.nocs_network_type == "xyz":
            self.nocs_backbone_adaptor = nn.Sequential(
                nn.Linear(self.backbone.embed_dim * self.lateral_layers, 384), nn.ReLU(), nn.Linear(384, 32)
            )
            self.nocs_net = NOCSModule_xyz()
        elif self.nocs_network_type == "fpn":
            self.fpn_layer_ids = self.lateral_layers
            self.fpn_scales = [1, 2, 3, 4]
            self.nocs_backbone_adaptor = nn.Identity()  # nn.Sequential(nn.Identity(), nn.ReLU(), nn.Identity())
            self.nocs_net = FPNModule(in_channels=384, out_channels=384, scales=self.fpn_scales, img_size=(224, 224))
        elif "dpt" in self.nocs_network_type:
            self.nocs_backbone_adaptor = nn.Identity()
            separate_xyz = "xyz" in self.nocs_network_type
            f_act = "identity"
            if "sigmoid" in self.nocs_network_type:
                f_act = "sigmoid"
            elif "relu" in self.nocs_network_type:
                f_act = "relu"
            fusion_norm_type = "bn"
            nocs_norm_type = "none"
            if "gnfusion" in self.nocs_network_type:
                fusion_norm_type = "gn"
            if "gnnocs" in self.nocs_network_type:
                nocs_norm_type = "gn"
            self.nocs_net = DPTNOCSHead(
                in_channels=[384, 384, 384, 384],
                channels=nocs_channels,
                embed_dims=384,
                post_process_channels=[48, 96, 192, 384],
                output_size=self.backbone_input_res,
                readout_type="project",
                separate_xyz=separate_xyz,
                final_act_type=f_act,
                fusion_norm_type=fusion_norm_type,
                nocs_head_norm_type=nocs_norm_type,
            )
        else:
            raise ValueError(f"Unknown NOCS network type: {self.nocs_network_type}")

        # Reconstruction module
        self.recons_net = ReconsModule_Film(
            input_dim=input_dim,
            num_layers=recons_num_layers,
            hidden_dim=recons_hidden_dim,
            nonlinearity=nonlinearity,
            normalization_type=normalization_type,
            modulate_last_layer=recons_modulate_last_layer,
        )

        # the input dimension = embedding dimension * number of layers used for cls tokens + embedding dimension of
        # final output
        recons_input_dim = self.backbone.embed_dim * (
            (len(self.lateral_layers) if isinstance(self.lateral_layers, list) else self.lateral_layers) + 1
        )

        if self.recons_encoder_type == "ffn":
            self.recons_backbone_adaptor = ReconsMappingModule_FFN(
                input_dim=recons_input_dim,
                output_dim=self.recons_net.latent_code_dim,
                hidden_layers=recons_adaptor_hidden_layers,
                hidden_dim=recons_adaptor_hidden_dim,
                output_normalization=normalize_shape_code,
                shape_code_norm_scale=recons_shape_code_norm_scale,
            )
        elif self.recons_encoder_type == "dpt":
            self.recons_backbone_adaptor = ReconsMappingModule_DPT(
                vit_feature_dim=self.backbone.embed_dim,
                output_dim=self.recons_net.latent_code_dim,
                output_normalization=normalize_shape_code,
                shape_code_norm_scale=recons_shape_code_norm_scale,
            )
        elif self.recons_encoder_type == "resnet50":
            self.recons_backbone_adaptor = ReconsMappingModule_ResNet50(
                vit_feature_dim=self.backbone.embed_dim,
                output_dim=self.recons_net.latent_code_dim,
                output_normalization=normalize_shape_code,
                shape_code_norm_scale=recons_shape_code_norm_scale,
            )
        elif self.recons_encoder_type == "resnet101":
            self.recons_backbone_adaptor = ReconsMappingModule_ResNet101(
                vit_feature_dim=self.backbone.embed_dim,
                output_dim=self.recons_net.latent_code_dim,
                output_normalization=normalize_shape_code,
                shape_code_norm_scale=recons_shape_code_norm_scale,
            )
        elif self.recons_encoder_type == "resnet34":
            self.recons_backbone_adaptor = ReconsMappingModule_ResNet34(
                vit_feature_dim=self.backbone.embed_dim,
                output_dim=self.recons_net.latent_code_dim,
                output_normalization=normalize_shape_code,
                shape_code_norm_scale=recons_shape_code_norm_scale,
            )
        else:
            raise ValueError(f"Unknown recons_encoder_type: {self.recons_encoder_type}")

        self.freeze_pretrained_weights = False
        if freeze_pretrained_weights:
            print(f"Freezing backbone ({self.backbone_model}) weights and set backbone to eval mode.")
            self.freeze_pretrained_weights = True
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    def dino_dpt_forward_helper(self, image):
        """Forward function for DINOv2 based backbones, using a multiscale approach"""
        if self.backbone_input_res[0] == image.shape[2] and self.backbone_input_res[1] == image.shape[3]:
            img = image
        else:
            img = nn.functional.interpolate(image, self.backbone_input_res, mode="bilinear")
        bs, _, w, h = img.shape

        with torch.no_grad():
            # https://github.com/facebookresearch/dinov2/issues/2#issuecomment-1513210404
            # https://github.com/facebookresearch/dinov2/blob/6a6261546c3357f2c243a60cfafa6607f84efcb7/hubconf.py#L120
            x = self.backbone.get_intermediate_layers(
                img, self.lateral_layers, return_class_token=True, reshape=True, norm=False
            )

        # reconstruction
        # freq_shifts are codes for layers concatenated
        # shape_code = torch.mean(self.recons_backbone_adaptor(img_feature), dim=1, keepdim=False)
        shape_code = self.recons_backbone_adaptor(x)

        return x, shape_code

    def dino_multiscale_foward_helper(self, image):
        """Forward function for DINOv2 based backbones, using a multiscale approach"""
        if self.backbone_input_res[0] == image.shape[2] and self.backbone_input_res[1] == image.shape[3]:
            img = image
        else:
            img = nn.functional.interpolate(image, self.backbone_input_res, mode="bilinear")
        bs, _, w, h = img.shape

        with torch.no_grad():
            # https://github.com/facebookresearch/dinov2/issues/2#issuecomment-1513210404
            # https://github.com/facebookresearch/dinov2/blob/6a6261546c3357f2c243a60cfafa6607f84efcb7/hubconf.py#L120
            x = self.backbone.get_intermediate_layers(img, self.lateral_layers, return_class_token=True)
            img_feature = torch.cat([x[0][0], x[1][0], x[2][0], x[3][0]], dim=2)
            cls_feature = torch.cat([x[0][1], x[1][1], x[2][1], x[3][1], x[3][0].mean(dim=1)], dim=1)

        # reconstruction
        # freq_shifts are codes for layers concatenated
        # shape_code = torch.mean(self.recons_backbone_adaptor(img_feature), dim=1, keepdim=False)
        shape_code = self.recons_backbone_adaptor(cls_feature)

        # nocs
        nocs_feature_map = (
            self.nocs_backbone_adaptor(img_feature)
            .reshape(bs, w // self.backbone.patch_size, h // self.backbone.patch_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        nocs_feature_map = nn.functional.interpolate(
            nocs_feature_map, (image.shape[-2], image.shape[-1]), mode="bilinear"
        )

        return nocs_feature_map, shape_code

    def resnet_forward_helper(self, image):
        """Forward function for TIMM ResNet backbones"""
        bs = image.shape[0]

        nocs_feature = self.backbone(image)[1]
        shape_feature = nn.functional.interpolate(nocs_feature, (16, 16)).reshape(bs, 32, 256).permute(0, 2, 1)

        return nocs_feature, shape_feature

    def forward(self, img, mask, coords):
        nocs_features, shape_code = self.forward_helper(img)

        # nocs
        nocs_map = self.nocs_net.forward(
            inputs=nocs_features,
            mask=mask,
            resize_op=partial(nn.functional.interpolate, size=mask.shape[-2:], mode="bilinear", align_corners=False),
        )

        # reconstruction
        sdf = self.recons_net.forward(shape_code=shape_code, coords=coords)

        return nocs_map, sdf, shape_code

    def forward_nocs_and_shape_code(self, img, mask):
        nocs_features, shape_code = self.forward_helper(img)

        # nocs
        nocs_map = self.nocs_net.forward(
            inputs=nocs_features,
            mask=mask,
            resize_op=partial(nn.functional.interpolate, size=mask.shape[-2:], mode="bilinear", align_corners=False),
        )
        return nocs_map, shape_code

    def forward_shape_code(self, img):
        _, shape_code = self.forward_helper(img)
        return shape_code

    def forward_recons(self, img, mask, coords):
        _, shape_code = self.forward_helper(img)
        # reconstruction
        sdf = self.recons_net.forward(shape_code=shape_code, coords=coords)
        return sdf

    def forward_nocs(self, img, mask):
        nocs_feature_map, _ = self.forward_helper(img)

        # nocs
        nocs_map = self.nocs_net.forward(
            inputs=nocs_feature_map,
            mask=mask,
            resize_op=partial(nn.functional.interpolate, size=mask.shape[-2:], mode="bilinear", align_corners=False),
        )

        return nocs_map

    def get_lr_params_list(self, nocs_lr, recons_lr):
        """Return a list of dictionaries used to set learning rates"""
        params_data = [
            # nocs
            {"params": self.nocs_backbone_adaptor.parameters(), "lr": nocs_lr},
            {"params": self.nocs_net.parameters(), "lr": nocs_lr},
            # recons
            {"params": self.recons_backbone_adaptor.parameters(), "lr": recons_lr},
            {"params": self.recons_net.parameters(), "lr": recons_lr},
        ]

        return params_data

    def get_nocs_lr_params_list(self, nocs_lr):
        params_data = [
            # nocs
            {"params": self.nocs_backbone_adaptor.parameters(), "lr": nocs_lr},
            {"params": self.nocs_net.parameters(), "lr": nocs_lr},
        ]
        return params_data

    def fpn_forward_features(self, img, layer_ids, masks=None):
        x = self.backbone.prepare_tokens_with_masks(img, masks)
        lateral_output, layers_tokens, total_block_len = [], [], len(self.backbone.blocks)
        blocks_to_take = range(total_block_len - self.lateral_layers, total_block_len)

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            x_norm = self.backbone.norm(x)

            if i in layer_ids:
                layers_tokens.append(x_norm[:, 1:])

            if i in blocks_to_take:
                lateral_output.append(x_norm)

        lateral_class_tokens = [out[:, 0] for out in lateral_output]
        lateral_output = [out[:, 1:] for out in lateral_output]

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
            "layers_tokens": layers_tokens,
            "lateral_output": tuple(zip(lateral_output, lateral_class_tokens)),
        }

    def dino_fpn_forward_helper(self, img):
        """Forward function for DINOv2 based backbones"""
        bs = img.shape[0]

        temp = self.fpn_forward_features(img, self.fpn_layer_ids)
        fpn_feature_maps = temp["layers_tokens"]

        with torch.no_grad():
            x = temp["lateral_output"]

            # x = self.backbone.get_intermediate_layers(img, self.lateral_layers, return_class_token=True)

            cls_feature = torch.cat(
                [
                    x[0][1],
                    x[1][1],
                    x[2][1],
                    x[3][1],
                    x[3][0].mean(dim=1),
                ],
                dim=1,
            )

        for i, layer_feature in enumerate(fpn_feature_maps):
            layer_feature = self.nocs_backbone_adaptor(layer_feature).reshape(bs, 16, 16, -1).permute(0, 3, 1, 2)
            scale = 2 ** (self.fpn_scales[-1 - i] - min(self.fpn_scales))
            layer_feature = nn.functional.interpolate(
                layer_feature, (layer_feature.shape[-2] * scale, layer_feature.shape[-1] * scale), mode="bilinear"
            )
            fpn_feature_maps[i] = layer_feature

        shape_code = self.recons_backbone_adaptor(cls_feature)

        return fpn_feature_maps, shape_code

    def train(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        # override mode change on backbone
        if self.freeze_pretrained_weights:
            self.backbone.eval()
        return self

    def eval(self: T) -> T:
        return self.train(False)

    def _get_lateral_layer_indices(self, model_name):
        """Config helper functions to get the indices of the layers to use"""
        if "vits14" in model_name:
            out_indices = [2, 5, 8, 11]
        elif "vitb14" in model_name:
            out_indices = [2, 5, 8, 11]
        elif "vitl14" in model_name:
            out_indices = [4, 11, 17, 23]
        elif "vitg14" in model_name:
            out_indices = [9, 19, 29, 39]
        else:
            raise NotImplementedError(f"Unknown model name: {model_name}")
        return out_indices
