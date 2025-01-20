import torch.nn as nn
import torch
from torch import nn as nn

# local libs import
from crisp.models.pspnet import PSPNet
from crisp.models.registration import umeyama


# class Interpolate(nn.Module):
#     def __init__(self, size, mode='bilinear'):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode

#     def forward(self, x):
#         x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
#         return x


class NocsNetwork(nn.Module):
    """Deprecated.
    Network for predicting NOCS.
    Uses a backbone network to obtain features.
    """

    def __init__(
        self,
        backbone_model="resnet34",
        backbone_model_pretrained=True,
        freeze_pretrained_weights=True,
    ):
        super().__init__()

        if backbone_model == "dino":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self.backbone_layer = nn.Sequential(nn.Linear(384, 384), nn.ReLU(), nn.Linear(384, 32))

            self.foward_helper = self.dino_foward_helper

        else:
            # currently using PSPNet as pixel-wise feature backbone
            self.backbone = PSPNet(
                bins=(1, 2, 3, 6),
                backbone=backbone_model,
                use_pretrained_backbone=backbone_model_pretrained,
                freeze_backbone=freeze_pretrained_weights,
            )

            self.foward_helper = self.resnet_forward_helper

        self.nocs_net = nn.Sequential(
            nn.Conv1d(32, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 3, 1),
        )

        if freeze_pretrained_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if backbone_model == "dino":
                for param in self.backbone_layer.parameters():
                    param.requires_grad = False

    def dino_foward_helper(self, image):
        bs = image.shape[0]
        img = nn.functional.interpolate(image, (224, 224))
        output = self.backbone.forward_features(img)["x_norm_patchtokens"]
        output = self.backbone_layer(output).reshape(bs, 16, 16, -1).permute(0, 3, 1, 2)
        output = nn.functional.interpolate(output, (image.shape[-2], image.shape[-1]))
        return output

    def resnet_forward_helper(self, image):
        return self.backbone(image)[1]

    def forward(self, image, mask):
        bs = image.shape[0]
        output = self.foward_helper(image)

        # mask out image RGB features
        masked_features = output * mask[:, None, ...]
        masked_features = masked_features.reshape(bs, masked_features.shape[1], -1)

        # NOCS mlp
        nocs = self.nocs_net(masked_features)
        nocs_map = nocs.reshape(bs, -1, image.shape[-2], image.shape[-1])

        return nocs_map


class NOCSModule(nn.Module):
    """NOCS module. Input are feature map of the image dimension."""

    def __init__(self):
        super().__init__()
        self.nocs_net = nn.Sequential(
            nn.Conv1d(32, 16, 1), nn.ReLU(), nn.Conv1d(16, 8, 1), nn.ReLU(), nn.Conv1d(8, 3, 1)
        )

    def forward(self, inputs, mask):
        bs = inputs.shape[0]

        # zero out non instance features
        masked_features = inputs * mask
        masked_features = masked_features.reshape(bs, masked_features.shape[1], -1)

        # NOCS mlp
        nocs = self.nocs_net(masked_features)
        nocs_map = nocs.reshape(bs, -1, inputs.shape[-2], inputs.shape[-1])

        return nocs_map


class NOCSModule_xyz(nn.Module):
    """NOCS module. Input are feature map of the image dimension."""

    def __init__(self):
        super().__init__()
        self.nocs_net_x = nn.Sequential(
            nn.Conv1d(32, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 1, 1),
        )
        self.nocs_net_y = nn.Sequential(
            nn.Conv1d(32, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 1, 1),
        )
        self.nocs_net_z = nn.Sequential(
            nn.Conv1d(32, 16, 1),
            nn.ReLU(),
            nn.Conv1d(16, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 1, 1),
        )

    def forward(self, inputs, mask):
        bs = inputs.shape[0]

        # zero out non instance features
        masked_features = inputs  # * mask[:, None, ...]
        masked_features = masked_features.reshape(bs, masked_features.shape[1], -1)

        nocs_x = self.nocs_net_x(masked_features)
        nocs_y = self.nocs_net_y(masked_features)
        nocs_z = self.nocs_net_z(masked_features)

        nocs = torch.concat([nocs_x, nocs_y, nocs_z], dim=1)
        nocs_map = nocs.reshape(bs, -1, inputs.shape[-2], inputs.shape[-1])

        return nocs_map


if __name__ == "__main__":
    print("Some simple registration tests")
    from crisp.utils.evaluation_metrics import rotation_error

    for _ in range(10):
        U, _, V = torch.linalg.svd(torch.rand((3, 3)))
        R = U @ V.T
        if torch.linalg.det(R) < 0:
            V[:, 2] = -V[:, 2]
            R = U @ V.T
        R_gt = R
        s_gt = torch.rand(1)
        t = torch.rand((3, 1))
        source = torch.rand((3, 1000))
        target = s_gt * R_gt @ source + t
        s, R, t, T = umeyama(source, target)
        rot_err = rotation_error(R, R_gt)
        print(f"rot err: {rot_err}")
        print(f"% scale err: {(s - s_gt) / s_gt * 100}")
