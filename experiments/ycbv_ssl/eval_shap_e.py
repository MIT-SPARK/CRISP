import yaml
from jsonargparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import lightning as L
import lightning.fabric as LF
import datetime

from torch.utils import data as tchdata
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from crisp.models.mf_corrector import MultiFrameShapeCorrector
# project local imports
from train_impl import train_sep, train_mixed, train_mixed_mf_shp
from crisp.backend.slam import ObjectPGOSolver
from crisp.datasets import bop
from crisp.datasets.bop import BOPObjectPCDataset
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.file_utils import id_generator
from crisp.models.pipeline import Pipeline
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.models.certifier import FrameCertifier

from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh


@dataclass
class ExpSettings:
    # dataset
    bop_ds_dir: str = "../../data/bop/bop_datasets"
    unified_ds_dir: str = ""
    dataloader_batch_size: int = 2
    dataloader_num_workers: int = 0
    ds_name: str = "ycbv"
    split: str = "test"
    bop19_targets: bool = True
    model_ckpts_save_dir: str = "./model_ckpts"
    objects_info_path: str = "./objects_info.yaml"

    # "mixed" or "separate"
    # mixed: load all the scenes in the same dataloader
    # separate: load each scene in a separate dataloader (during each epoch)
    normalized_recons: bool = True
    scenes_mode: str = "mixed"
    scenes_to_load: tuple = None
    test_only: bool = False
    use_torch_profile: bool = False

    # ssl training
    num_epochs: int = 100
    ssl_lr: float = 3e-4
    ssl_nocs_lr: float = 4e-4
    ssl_recons_lr: float = 3e-4
    ssl_weight_decay: float = 1e-5
    ssl_nocs_weight: float = 1e2
    ssl_shape_weight: float = 0.1
    ssl_batch_size: int = 5
    ssl_nocs_clamp_quantile: float = 0.9
    amp_mode: str = "32-true"

    # backbone model
    use_torch_compile: bool = True
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    backbone_input_res: tuple = (420, 420)
    freeze_pretrained_backbone_weights: bool = True
    backbone_model_path: str = None
    log_root_dir: str = "logs"

    # implicit recons model
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"

    # nocs model
    nocs_network_type: str = "dpt_gnfusion_gnnocs"
    nocs_lateral_layers_type: str = "spaced"

    # framewise certification parameters
    use_frame_certifier: bool = True
    cert_depths_clamp_thres: float = 10
    cert_depths_quantile: float = 0.9
    cert_depths_eps: float = 0.02

    # corrector parameters
    corrector_lr: float = 0.05
    corrector_scale_lr: float = 5e-4
    corrector_max_iters: int = 50
    corrector_max_ransac_iters: int = 50
    corrector_sdf_input_loss_weight: float = 10.0
    corrector_sdf_nocs_loss_weight: float = 0
    corrector_trim_quantile: float = 0.9
    corrector_log_loss_traj: bool = False
    corrector_log_dump_dir: str = "./corrector_logs"
    corrector_nesterov: bool = True
    corrector_momentum: float = 0.9
    corrector_loss_multiplier: float = 100

    # multi-frame corrector parameters
    mf_corrector_clamping_thres: float = 0.5

    # pgo parameters
    pgo_odom_noise_variance: tuple = (1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1)
    pgo_obj_noise_variance: tuple = (1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1)
    pgo_prior_noise_variance: tuple = (1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)

    # pipeline
    pipeline_nr_downsample_before_corrector: int = 1000
    registration_inlier_thres: float = 0.1
    #frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-upper-scale"
    frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-scale-free"
    #frame_corrector_mode: str = "nocs-only-sdf-input-nocs-scale-free"

    mf_corrector_mode: str = "gnc"
    mf_corrector_rolling_window_size: int = 50
    pipeline_output_intermediate_vars: bool = True
    pipeline_output_precrt_results: bool = False

    # set to true to run PGO on the obtained poses
    # if corrector is set to True, PGO will be run
    # on the corrected poses
    use_corrector: bool = True
    use_mf_shape_corrector: bool = True
    use_pgo: bool = False

    # loading model & testing
    visualize: bool = False
    visualize_rgb: bool = False
    visualize_test: bool = False
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    checkpoint_path: str = None

    # automatically populated if missing
    exp_id: str = None
    pose_noise_scale: float = 0


def main(opt: ExpSettings):
    LF.seed_everything(42)
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    torch.set_float32_matmul_precision("high")

    # one logger to log root dir, another to the exp folder
    tb_logger_path = os.path.join(opt.log_root_dir, "tensorboard")
    safely_make_folders([tb_logger_path])
    tb_logger_root = LF.loggers.TensorBoardLogger(
        root_dir=tb_logger_path, name="joint_model_training", flush_secs=10, max_queue=5
    )
    tb_logger_exp = LF.loggers.TensorBoardLogger(root_dir=exp_dump_path, name="log", flush_secs=10, max_queue=5)
    fabric = L.Fabric(accelerator="gpu", loggers=[tb_logger_root, tb_logger_exp], devices=[0], precision=opt.amp_mode)
    fabric.launch()

    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        local_backbone_model_path=opt.backbone_model_path,
        freeze_pretrained_weights=True,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
        lateral_layers_type=opt.nocs_lateral_layers_type,
        backbone_input_res=opt.backbone_input_res,
    )

    corrector = None
    if opt.use_corrector:
        print("Pipeline will use a corrector.")
        corrector = JointCorrector(
            model=model,
            solver_algo="torch-gd-accel",
            device="cuda",
            log_loss_traj=opt.corrector_log_loss_traj,
            max_iters=opt.corrector_max_iters,
            max_ransac_iters=opt.corrector_max_ransac_iters,
            corrector_lr=opt.corrector_lr,
            corrector_scale_lr=opt.corrector_scale_lr,
            sdf_nocs_loss_weight=opt.corrector_sdf_nocs_loss_weight,
            sdf_input_loss_weight=opt.corrector_sdf_input_loss_weight,
            trim_quantile=opt.corrector_trim_quantile,
            solver_nesterov=opt.corrector_nesterov,
            solver_momentum=opt.corrector_momentum,
            registration_inlier_thres=opt.registration_inlier_thres,
        )

    frame_certifier = None
    if opt.use_frame_certifier:
        print("Pipeline will use a certifier")
        frame_certifier = FrameCertifier(
            model=model,
            depths_clamp_thres=opt.cert_depths_clamp_thres,
            depths_quantile=opt.cert_depths_quantile,
            depths_eps=opt.cert_depths_eps,
        )

    # PGO solver
    pgo_solver = ObjectPGOSolver(opt.pgo_odom_noise_variance, opt.pgo_obj_noise_variance, opt.pgo_prior_noise_variance)

    # Multi-frame shape corrector
    mf_shp_corrector = None
    if opt.use_mf_shape_corrector:
        mf_shp_corrector = MultiFrameShapeCorrector(
            mode=opt.mf_corrector_mode,
            clamping_thres=opt.mf_corrector_clamping_thres,
            rolling_window_size=opt.mf_corrector_rolling_window_size,
        )

    print(f"Loading model checkpoint from {opt.checkpoint_path}.")
    try:
        state = torch.load(opt.checkpoint_path)
        model.load_state_dict(state["model"])
    except RuntimeError as e:
        state = fabric.load(opt.checkpoint_path)
        # handle https://github.com/Lightning-AI/pytorch-lightning/issues/17177#issuecomment-1506101461
        new_state = {}
        for k, v in state["model"].items():
            if "_orig_mod" in k:
                new_k = k.replace("_orig_mod.", "")
                new_state[new_k] = v
        model.load_state_dict(new_state)

    pipeline = Pipeline(
        model=model,
        corrector=corrector,
        frame_certifier=frame_certifier,
        frame_corrector_mode=opt.frame_corrector_mode,
        pgo_solver=pgo_solver,
        multi_frame_shape_code_corrector=mf_shp_corrector,
        device=fabric.device,
        nr_downsample_before_corrector=opt.pipeline_nr_downsample_before_corrector,
        sdf_input_loss_multiplier=opt.corrector_loss_multiplier,
        registration_inlier_thres=opt.registration_inlier_thres,
        output_intermediate_vars=opt.pipeline_output_intermediate_vars,
        output_precrt_results=opt.pipeline_output_precrt_results,
        ssl_batch_size=opt.ssl_batch_size,
        ssl_nocs_clamp_quantile=opt.ssl_nocs_clamp_quantile,
        ssl_nocs_weight=opt.ssl_nocs_weight,
        ssl_shape_weight=opt.ssl_shape_weight,
        normalized_recons=opt.normalized_recons,
    )

    # prepare for SSL training
    pipeline.freeze_backbone_weights()
    pipeline.freeze_sdf_decoder_weights()
    if opt.use_torch_compile and os.getenv("CORE_USE_TORCH_COMPILE", "True").lower() in ("true", "1", "t"):
        print("Use torch.compile to speed up training.")
        pipeline.model = torch.compile(pipeline.model)
    else:
        print("Dose not use torch.compile to speed up training.")

    if opt.test_only:
        pipeline.eval()

    optim = torch.optim.SGD(
        model.get_lr_params_list(nocs_lr=opt.ssl_nocs_lr, recons_lr=opt.ssl_recons_lr),
        lr=opt.ssl_lr,
        weight_decay=opt.ssl_weight_decay,
    )
    pipeline, optim = fabric.setup(pipeline, optim)

    print(f"Noise scale: {opt.pose_noise_scale}")

    # load objects info (parameters for recovering unnormalized frames)
    objects_info = None
    with open(os.path.join(opt.objects_info_path), "r") as f:
        objects_info = yaml.safe_load(f)

    # load object dataset
    obj_pc_ds = BOPObjectPCDataset(ds_dir=Path(opt.bop_ds_dir) / opt.ds_name / "models")

    # load unified objects dataset
    unified_shape_ds = UnifiedObjects(
        folder_path=opt.unified_ds_dir,
        bop_dataset_path=opt.bop_ds_dir,
        preload_to_mem=False,
        pc_size=5000,
        sample_surface_points_count=1500,
        sample_global_nonmnfld_points_count=3000,
        sample_bounds=(-1.0, 1.0),
        debug_vis=False,
        normalized_recons=opt.normalized_recons,
        data_to_output=[
            "rgb",
            "nocs",
            "coords",
            "metadata",
            "cam_intrinsics",
            "cam_pose",
            "object_pc",
        ],
    )

    if opt.scenes_to_load is None:
        dataset = bop.BOPSceneDataset(
            ds_name="ycbv",
            split=opt.split,
            bop_ds_dir=Path(opt.bop_ds_dir),
            load_depth=True,
            scenes_to_load=opt.scenes_to_load,
            bop19_targets=opt.bop19_targets,
        )
        opt.scenes_to_load = tuple(sorted(dataset.frame_index["scene_id"].unique().tolist()))

    if not opt.test_only:
        if opt.scenes_mode == "separate":
            train_sep(pipeline, optim, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path)
        elif opt.scenes_mode == "mixed":
            train_mixed(pipeline, optim, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path)
        elif opt.scenes_mode == "mixed_mf_shp":
            train_mixed_mf_shp(pipeline, optim, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path)
    else:
        print("Testing only mode.")
        
        xm = load_model('transmitter', device="cuda")
        # model = load_model('image300M', device="cuda")
        # diffusion = diffusion_from_config(load_config('diffusion'))

    if opt.use_torch_profile:
        opt.num_epochs = 1
    profile_lengths = 1
    all_results = []
    for s_id in opt.scenes_to_load:
        # train on a specific trajectory
        dataset = bop.BOPSceneDataset(
            ds_name="ycbv",
            split=opt.split,
            bop_ds_dir=Path(opt.bop_ds_dir),
            load_depth=True,
            scenes_to_load=(s_id,),
            bop19_targets=opt.bop19_targets,
        )
        ssl_dl = fabric.setup_dataloaders(
            tchdata.DataLoader(
                dataset,
                shuffle=False,
                num_workers=0,
                batch_size=opt.dataloader_batch_size,
                collate_fn=dataset.collate_fn,
            )
        )

        metric_folder = os.path.join(exp_dump_path, f"traj_data", f"traj_{s_id}")
        safely_make_folders([metric_folder])
        traj_results = []

        prof = None
        warmup = float("inf")
        if opt.use_torch_profile:
            print("Use PyTorch profiler")
            warmup = 1
            schdule = torch.profiler.schedule(wait=0, warmup=warmup, active=profile_lengths)
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schdule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces/no_grad_ransac"),
                record_shapes=False,
                profile_memory=False,
            )
            prof.start()

        train_dataloader=ssl_dl

        for i, (rgbs, masks, depths, intrinsics, world_T_cam, gt_objs_data, frames_info) in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with record_function("pipeline.forward"):
                payload = pipeline.forward(rgbs, masks, depths, intrinsics, gt_objs_data, frames_info)

            breakpoint()
            for i, latent in enumerate(latents):
                # images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
                mesh = decode_latent_mesh(xm, latent)

            traj_results.extend(results)

            # save metrics
            np.save(metric_dump_path, all_results)

            if opt.use_torch_profile:
                prof.step()
                if i >= profile_lengths + warmup:
                    break

        if opt.use_torch_profile:
            prof.stop()

        all_results.extend(traj_results)

        # save metrics
        np.save(os.path.join(metric_folder, f"traj_results.npy"), all_results)

        del dataset, ssl_dl
        torch.cuda.empty_cache()

        if opt.use_torch_profile:
            break



    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    # generate random timestamped experiment ID
    if opt.exp_id is None:
        opt.exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"

    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    safely_make_folders([exp_dump_path])
    parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))

    opt = ExpSettings(**opt)
    main(opt)
