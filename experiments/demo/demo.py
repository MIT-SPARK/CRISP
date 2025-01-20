import pathlib
import copy
from PIL import Image
import pyvista as pv
import imageio
import torchvision.transforms.v2 as tvt
import torchvision.ops as tops

from crisp.config.config import BaseExpSettings
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.pipeline import Pipeline
from crisp.models.certifier import FrameCertifier
from crisp.models.corrector import *
from crisp.utils.visualization_utils import get_meshes_from_shape_codes
from crisp.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

script_dir = pathlib.Path(__file__).parent.resolve().absolute()
# MODIFY this to your checkpoint path
ckpt_path = "CRISP/experiments/ycbv_supervised/model_ckpts/202403192306_AKNAN/checkpoint.pth"
data_dir = script_dir / "data"
rgb_dir = data_dir / "rgb"
mask_dir = data_dir / "mask"
depth_dir = data_dir / "depth"
base_id = "000013"


def load_data():
    # load rgb
    rgb_path = rgb_dir / f"{base_id}.png"
    rgb = np.array(Image.open(rgb_path))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    rgb = rgb[..., :3]
    h, w = rgb.shape[:2]

    # load mask
    mask_files = sorted(list(mask_dir.glob(f"{base_id}*.png")))
    mask = np.zeros((h, w), dtype=np.uint8)
    separated_masks = []
    for n, mask_file in enumerate(mask_files):
        mask_n = np.array(Image.open(mask_file))
        mask[mask_n == 255] = n + 1
        separated_masks.append(mask_n)

    separated_masks = torch.tensor(np.stack(separated_masks, axis=0))

    # bboxes
    separated_masks = torch.tensor(separated_masks, dtype=torch.float32)
    # (x1, y1, x2, y2)
    bboxes = tops.masks_to_boxes(separated_masks)

    # load depth
    depth_path = depth_dir / f"{base_id}.png"
    depth = np.array(imageio.imread(depth_path).astype(np.float32)) * 0.1 / 1000

    # object info
    # a list of N lists, where N is th number of frames
    # each list contains a list of dictionaries, where each dictionary needs
    # the following keys:
    # label: str
    # bbox: list of 4 floats (x1, y1, x2, y2)
    obj_info = [[]]
    for i in range(len(separated_masks)):
        obj_info[0].append(
            {
                "label": str(i + 1),
                "bbox": bboxes[i, ...].int().tolist(),
                "id_in_segm": i + 1,
            }
        )

    # frame info
    frame_info = [{"inst_count": len(separated_masks)}]

    return rgb, depth, mask, bboxes, obj_info, frame_info


def setup_pipeline(opt):
    # joint nocs & recons model
    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        local_backbone_model_path=None,
        freeze_pretrained_weights=True,
        backbone_input_res=opt.backbone_input_res,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
        nocs_channels=opt.nocs_channels,
        lateral_layers_type=opt.nocs_lateral_layers_type,
        normalize_shape_code=opt.recons_shape_code_normalization,
        recons_shape_code_norm_scale=opt.recons_shape_code_norm_scale,
    )

    print(f"Loading model checkpoint from {opt.checkpoint_path}.")
    try:
        state = torch.load(opt.checkpoint_path)
        model.load_state_dict(state["model"])
    except RuntimeError as e:
        print(f"Error loading state from {opt.checkpoint_path}: {e}")

    model.eval()
    model.backbone.eval()
    model.cuda()

    corrector = JointCorrector(
        model=model,
        solver_algo=opt.corrector_algo,
        device="cuda",
        log_loss_traj=opt.corrector_log_loss_traj,
        max_iters=opt.corrector_max_iters,
        max_ransac_iters=opt.corrector_max_ransac_iters,
        corrector_lr=opt.corrector_lr,
        corrector_scale_lr=opt.corrector_scale_lr,
        nocs_correction_lr=opt.corrector_nocs_correction_lr,
        shape_correction_lr=opt.corrector_shape_correction_lr,
        sdf_nocs_loss_weight=opt.corrector_sdf_nocs_loss_weight,
        sdf_input_loss_weight=opt.corrector_sdf_input_loss_weight,
        trim_quantile=opt.corrector_trim_quantile,
        solver_nesterov=opt.corrector_nesterov,
        solver_momentum=opt.corrector_momentum,
        registration_inlier_thres=opt.registration_inlier_thres,
        shape_code_library=None,
    )

    frame_certifier = FrameCertifier(
        model=model,
        depths_clamp_thres=opt.cert_depths_clamp_thres,
        depths_quantile=opt.cert_depths_quantile,
        depths_eps=opt.cert_depths_eps,
        degen_min_eig_thres=opt.cert_degen_min_eig_thres,
        use_degen_cert=opt.use_degen_cert,
        shape_code_library=None,
    )

    pipeline = Pipeline(
        model=model,
        corrector=corrector,
        frame_certifier=frame_certifier,
        frame_corrector_mode=opt.frame_corrector_mode,
        pgo_solver=None,
        multi_frame_shape_code_corrector=None,
        multi_frame_geometric_shape_corrector=None,
        device="cuda",
        input_H=opt.image_size[0],
        input_W=opt.image_size[1],
        nr_downsample_before_corrector=opt.pipeline_nr_downsample_before_corrector,
        sdf_input_loss_multiplier=opt.corrector_loss_multiplier,
        registration_inlier_thres=opt.registration_inlier_thres,
        output_intermediate_vars=True,
        output_precrt_results=opt.pipeline_output_precrt_results,
        ssl_batch_size=opt.ssl_batch_size,
        ssl_nocs_clamp_quantile=opt.ssl_nocs_clamp_quantile,
        ssl_augmentation=False,
        ssl_augmentation_type=opt.ssl_augmentation_type,
        ssl_augmentation_gaussian_perturb_std=opt.ssl_augmentation_gaussian_perturb_std,
        no_grad_model_forward=True,
        normalized_recons=False,
        output_degen_condition_number=False,
        normalize_input_image=False,
        shape_code_library=None,
    )

    return pipeline


if __name__ == "__main__":
    print("Demo script for CRISP")
    device = "cuda"

    # load data
    rgb, depth, mask, bboxes, obj_info, frame_info = load_data()
    intrinsics = torch.tensor(
        [[1.0668e03, 0.0000e00, 3.1299e02], [0.0000e00, 1.0675e03, 2.4131e02], [0.0000e00, 0.0000e00, 1.0000e00]],
    )

    # create pipeline
    # note: dataset_dir not actually used
    opt = BaseExpSettings(dataset_dir="./")
    # (H, W)
    opt.image_size = (480, 640)
    opt.checkpoint_path = ckpt_path
    opt.use_corrector = False # set to True to use framewise corrector
    pipeline = setup_pipeline(opt)

    # run pipeline
    rgb_transform = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    rgb_tensor = rgb_transform(rgb)

    payload = pipeline(
        rgbs=rgb_tensor.unsqueeze(0).to(device=device),
        masks=torch.tensor(mask, device=device)[None, None, ...],
        depths=torch.tensor(depth.astype(np.float32), device=device)[None, None, ...],
        intrinsics=torch.tensor(intrinsics, device=device).unsqueeze(0).unsqueeze(0),
        objs=obj_info,
        frames_info=frame_info,
    )

    # visualize
    print("Showing predicted meshes.")
    pred_meshes = get_meshes_from_shape_codes(pipeline.model, payload["shape_code"], mesh_recons_scale=0.35)
    pl = pv.Plotter(off_screen=False)
    for mesh in pred_meshes:
        pl.add_mesh(pv.wrap(mesh), color="white", opacity=1)
    pl.show_axes()
    pl.show_bounds()
    pl.show()

    print("Showing meshes transformed to camera frame.")
    transformed_meshes = []
    for ii, m in enumerate(pred_meshes):
        mesh = copy.deepcopy(m)
        s = payload["cam_s_nocs"][ii, ...].item()
        R = payload["cam_R_nocs"][ii, :3, :3].detach().cpu().numpy()
        t = (
            payload["cam_t_nocs"][ii, ...]
            .reshape(
                3,
            )
            .detach()
            .cpu()
            .numpy()
        )
        mesh.vertices = np.transpose(s * R @ np.transpose(mesh.vertices) + t.reshape(3, 1))
        transformed_meshes.append(mesh)

    pl = pv.Plotter(off_screen=False)

    for mesh in transformed_meshes:
        pl.add_mesh(pv.wrap(mesh), color="white", opacity=1)
    pl.show_axes()
    pl.show_bounds()
    pl.show()
