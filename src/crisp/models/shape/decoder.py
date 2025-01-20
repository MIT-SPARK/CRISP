# https://github.com/marcoamonteiro/pi-GAN/tree/0800af72b8a9371b2b62fec2ae69c32994bb802f

import numpy as np
import torch.nn as nn
import torch
import timm
from crisp.utils.torch_utils import get_output_shape


def film_sine_init(m):
    """Initialize non-first layer of a SIREN network
    See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30.
    """
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


def first_layer_film_sine_init(m):
    """Initialize first layer of a SIREN network
    See SIREN paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30.
    Note that for other layers, the weights are initialized by dividing 30.
    """
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-30 / num_input, 30 / num_input)


class Interpolate(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class ConvMappingNetwork(nn.Module):
    """A simple CNN mapping network"""

    def __init__(self, feature_map_size, input_channel, hidden_channel, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channel),
            nn.AvgPool2d(feature_map_size),
        )
        self.fc = nn.Linear(hidden_channel, output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x.view(x.size(0), -1))
        return x


class FilmSirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        out = self.layer(x)
        freq = freq.unsqueeze(1).expand_as(out).contiguous()
        phase_shift = phase_shift.unsqueeze(1).expand_as(out).contiguous()
        return torch.sin(freq * out + phase_shift)


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)


class FilmLinearLayer(nn.Module):
    def __init__(self, input_dim, out_dim, weight_norm=False, modulate_scale=True, modulate_shift=True):
        super().__init__()
        if weight_norm:
            self.layer = nn.utils.weight_norm(nn.Linear(input_dim, out_dim))
        else:
            self.layer = nn.Linear(input_dim, out_dim)
        self.input_dim = input_dim
        self.hidden_dim = out_dim
        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift

    def forward(self, input):
        x, freq, phase_shift = input[0], input[1], input[2]
        out_shape = torch.Size((x.shape[0], x.shape[1], self.hidden_dim))
        out = self.layer(x)
        if self.modulate_scale:
            out = freq.unsqueeze(1).expand(out_shape) * out
        if self.modulate_shift:
            out += phase_shift.unsqueeze(1).expand(out_shape)
        return out


class ShapeNetwork(nn.Module):
    """Prototype shape network. Supports ReLU & SIREN. Deprecated.

    Ref: https://github.com/marcoamonteiro/pi-GAN/blob/0800af72b8a9371b2b62fec2ae69c32994bb802f/siren/siren.py
    Ref: https://github.com/lucidrains/siren-pytorch
    Ref: https://github.com/vsitzmann/siren
    """

    def __init__(
        self,
        input_dim=3,
        num_layers=5,
        hidden_dim=256,
        backbone_model="resnet34",
        backbone_model_pretrained=False,
        freeze_pretrained_weights=False,
        mapping_network_type="conv",
        nonlinearity="sine",
        normalization_type="weight",
    ):
        super().__init__()
        self.siren_hidden_dim = hidden_dim

        # ResNet -> FiLMed SDF Network
        # mapping network
        self.mapping_output_dim = hidden_dim * num_layers * 2
        if mapping_network_type == "conv":
            self.backbone = timm.create_model(backbone_model, pretrained=backbone_model_pretrained, num_classes=0)
            mapping_input_dim = get_output_shape(self.backbone, (1, 3, 224, 224))
            self.mapping_network = ConvMappingNetwork(
                feature_map_size=mapping_input_dim[-2:],
                input_channel=mapping_input_dim[1],
                hidden_channel=mapping_input_dim[1],
                output_dim=self.mapping_output_dim,
            )
        elif mapping_network_type == "linear":
            if backbone_model == "dino":
                self.backbone = nn.Sequential(
                    Interpolate(size=(224, 224)),
                    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14"),
                    nn.Linear(384, 2560),
                )
                # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            else:
                self.backbone = timm.create_model(
                    backbone_model, pretrained=backbone_model_pretrained, num_classes=self.mapping_output_dim
                )
            self.mapping_network = torch.nn.Identity()
        else:
            raise NotImplementedError

        if freeze_pretrained_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if backbone_model != "dino":
                self.backbone.fc.weight.requires_grad = True
                self.backbone.fc.bias.requires_grad = True

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(), film_sine_init, first_layer_film_sine_init),
            "relu": (nn.ReLU(inplace=True), None, None),
        }
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        # helper function to handle weight normalization
        def fc_builder(input_dim=None, out_dim=None):
            if normalization_type == "weight":
                return FilmLinearLayer(input_dim=input_dim, out_dim=out_dim, weight_norm=True)
            elif normalization_type == "layer":
                return nn.Sequential(
                    FilmLinearLayer(input_dim=input_dim, out_dim=out_dim, weight_norm=False), nn.LayerNorm(out_dim)
                )
            else:
                return FilmLinearLayer(input_dim=input_dim, out_dim=out_dim)

        # building the main sdf network
        self.filmed_sdf = nn.ModuleList()
        self.filmed_sdf.append(nn.Sequential(fc_builder(input_dim=input_dim, out_dim=hidden_dim), nl))

        for _ in range(num_layers - 1):
            self.filmed_sdf.append(nn.Sequential(fc_builder(input_dim=hidden_dim, out_dim=hidden_dim), nl))
        self.final_layer = nn.Linear(hidden_dim, 1)

        # initialization
        if nl_weight_init is not None:
            self.filmed_sdf.apply(film_sine_init)
        if first_layer_init is not None:
            self.filmed_sdf[0].apply(first_layer_film_sine_init)

    def forward_mapping_only(self, img):
        """Get the predicted latent FiLM vector using the mapping network"""
        freq_shifts = self.mapping_network(self.backbone(img))
        return freq_shifts

    def forward_shape(self, freq_shifts, coords):
        x = coords
        for index, layer in enumerate(self.filmed_sdf):
            start_idx = index * self.siren_hidden_dim * 2
            x = layer(
                (
                    x,
                    freq_shifts[..., start_idx : start_idx + self.siren_hidden_dim],
                    freq_shifts[..., start_idx + self.siren_hidden_dim : start_idx + 2 * self.siren_hidden_dim],
                )
            )

        sdf = self.final_layer(x)
        return sdf

    def forward(self, img, coords):
        # freq_shifts are codes for layers concatenated
        freq_shifts = self.mapping_network(self.backbone(img))
        sdf = self.forward_shape(freq_shifts, coords)
        return sdf


class SirenShapeNetwork(nn.Module):
    """Prototype shape network. Deprecated.

    Ref: https://github.com/marcoamonteiro/pi-GAN/blob/0800af72b8a9371b2b62fec2ae69c32994bb802f/siren/siren.py
    Ref: https://github.com/lucidrains/siren-pytorch
    Ref: https://github.com/vsitzmann/siren
    """

    def __init__(
        self,
        input_dim=3,
        num_layers=5,
        hidden_dim=256,
        backbone_model="resnet34",
        backbone_model_pretrained=False,
        freeze_pretrained_weights=False,
        mapping_network_type="conv",
        nonlinearity="sine",
    ):
        super().__init__()
        self.siren_hidden_dim = hidden_dim

        # ResNet -> FiLMed SDF Network
        # mapping network
        self.mapping_output_dim = hidden_dim * num_layers * 2
        if mapping_network_type == "conv":
            self.backbone = timm.create_model(backbone_model, pretrained=backbone_model_pretrained, num_classes=0)
            mapping_input_dim = get_output_shape(self.backbone, (1, 3, 224, 224))
            self.mapping_network = ConvMappingNetwork(
                feature_map_size=mapping_input_dim[-2:],
                input_channel=mapping_input_dim[1],
                hidden_channel=mapping_input_dim[1],
                output_dim=self.mapping_output_dim,
            )
        elif mapping_network_type == "linear":
            self.backbone = timm.create_model(
                backbone_model, pretrained=backbone_model_pretrained, num_classes=self.mapping_output_dim
            )
            self.mapping_network = torch.nn.Identity()
        else:
            raise NotImplementedError

        if freeze_pretrained_weights:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.fc.weight.requires_grad = True
            self.backbone.fc.bias.requires_grad = True

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(), film_sine_init, first_layer_film_sine_init),
            "relu": (nn.ReLU(inplace=True), None, None),
        }
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        # building the main sdf network
        self.filmed_siren = nn.ModuleList([FilmSirenLayer(input_dim=input_dim, hidden_dim=hidden_dim)])
        for _ in range(num_layers - 1):
            self.filmed_siren.append(FilmSirenLayer(input_dim=hidden_dim, hidden_dim=hidden_dim))
        self.final_layer = nn.Linear(hidden_dim, 1)

        # initialization
        self.filmed_siren.apply(film_sine_init)
        self.filmed_siren[0].apply(first_layer_film_sine_init)

    def forward(self, img, coords):
        # freq_shifts are codes for layers concatenated
        freq_shifts = self.mapping_network(self.backbone(img))

        x = coords
        for index, layer in enumerate(self.filmed_siren):
            start_idx = index * self.siren_hidden_dim * 2
            x = layer(
                x,
                freq_shifts[..., start_idx : start_idx + self.siren_hidden_dim],
                freq_shifts[..., start_idx + self.siren_hidden_dim : start_idx + 2 * self.siren_hidden_dim],
            )

        sdf = self.final_layer(x)
        return sdf


class ReconsModule_Film(nn.Module):
    """Reconstruction module (implicit shape reconstruction). Inputs are latent shape code that comes from
    some backbone network.
    """

    def __init__(
        self,
        input_dim=3,
        num_layers=5,
        hidden_dim=256,
        nonlinearity="sine",
        normalization_type="weight",
        modulate_last_layer=False,
    ):
        """

        Parameters
        ----------
        input_dim
        num_layers
        hidden_dim
        nonlinearity
        normalization_type
        modulate_last_layer If False, last layer is an unmodulated linear layer; otherwise it's FiLM conditioned
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modulate_last_layer = modulate_last_layer

        # calculate the modulation vector size
        self.latent_code_dim = hidden_dim * num_layers * 2
        if self.modulate_last_layer:
            # last layer is hidden_dim x 1 where 1 is the sdf dimension
            output_dim = 1
            self.latent_code_dim += 2 * output_dim

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(), film_sine_init, first_layer_film_sine_init),
            "relu": (nn.ReLU(inplace=True), None, None),
        }
        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        # helper function to handle weight normalization
        def fc_builder(input_dim=None, out_dim=None):
            if normalization_type == "weight":
                return FilmLinearLayer(
                    input_dim=input_dim,
                    out_dim=out_dim,
                    weight_norm=True,
                )
            elif normalization_type == "layer":
                return nn.Sequential(
                    FilmLinearLayer(
                        input_dim=input_dim,
                        out_dim=out_dim,
                        weight_norm=False,
                    ),
                    nn.LayerNorm(out_dim),
                )
            else:
                return FilmLinearLayer(
                    input_dim=input_dim,
                    out_dim=out_dim,
                )

        # building the main sdf network
        self.filmed_sdf = nn.ModuleList()
        self.filmed_sdf.append(nn.Sequential(fc_builder(input_dim=input_dim, out_dim=hidden_dim), nl))

        for _ in range(num_layers - 1):
            self.filmed_sdf.append(nn.Sequential(fc_builder(input_dim=hidden_dim, out_dim=hidden_dim), nl))
        if not modulate_last_layer:
            self.final_layer = nn.Linear(hidden_dim, 1)
        else:
            self.final_layer = FilmLinearLayer(
                input_dim=hidden_dim,
                out_dim=1,
            )

        # initialization
        if nl_weight_init is not None:
            self.filmed_sdf.apply(film_sine_init)
        if first_layer_init is not None:
            self.filmed_sdf[0].apply(first_layer_film_sine_init)

    def forward_shape(self, freq_shifts, coords):
        """Forward given shape code"""
        x = coords
        for index, layer in enumerate(self.filmed_sdf):
            start_idx = index * self.hidden_dim * 2
            x = layer(
                (
                    x,
                    freq_shifts[..., start_idx : start_idx + self.hidden_dim],
                    freq_shifts[..., start_idx + self.hidden_dim : start_idx + 2 * self.hidden_dim],
                )
            )

        if self.modulate_last_layer:
            sdf = self.final_layer(
                (
                    x,
                    freq_shifts[..., -2].unsqueeze(-1),
                    freq_shifts[..., -1].unsqueeze(-1),
                )
            )
        else:
            sdf = self.final_layer(x)
        return sdf

    def forward(self, shape_code, coords):
        """

        Parameters
        ----------
        shape_code: (B, K)
        coords: (B, N, 3)
        """
        sdf = self.forward_shape(shape_code, coords)
        return sdf


class ReconsModule_Concat(nn.Module):
    """Reconstruction network; taking concatenated latent codes as inputs"""

    def __init__(self, input_dim=3, latent_dim=16, num_layers=5, hidden_dim=256, nonlinearity="relu"):
        super().__init__()
        self.latent_dim = latent_dim

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {
            "sine": (Sine(), film_sine_init, first_layer_film_sine_init),
            "relu": (nn.ReLU(inplace=True), None, None),
        }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        # each layer: linear + layer norm + nonlinearity
        def fc_builder(input_dim=None, out_dim=None):
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(out_dim))

        # build the main network
        self.sdf = nn.ModuleList()
        self.sdf.append(nn.Sequential(fc_builder(input_dim=input_dim + latent_dim, out_dim=hidden_dim), nl))

        # TODO: Finish this
