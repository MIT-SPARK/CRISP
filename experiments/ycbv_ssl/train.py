import pickle
import yaml
from jsonargparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import lightning as L
import lightning.fabric as LF
import datetime

# project local imports
from crisp.models.mf_corrector import MultiFrameShapeCorrector, MultiFrameGeometricShapeCorrector
from crisp.datasets import bop
from crisp.datasets.bop import BOPObjectPCDataset
from crisp.datasets.unified_objects import UnifiedObjects
from crisp.utils.file_utils import id_generator
from crisp.models.pipeline import Pipeline
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.models.certifier import FrameCertifier

# experiment local imports
from experiments.ycbv_object_slam.train_impl import train_sep, train_mixed, train_mixed_mf_shp
from experiments.ycbv_object_slam.test_impl import test_sep


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
    shape_code_library_path: str = "./shape_code_library.npy"
    artifacts_paths: str = "./exp_results/by_models/"

    # "mixed" or "separate"
    # mixed: load all the scenes in the same dataloader
    # separate: load each scene in a separate dataloader (during each epoch)
    normalized_recons: bool = True
    scenes_mode: str = "mixed"
    scenes_to_load: tuple = None
    test_only: bool = False
    use_torch_profile: bool = False
    save_every_epoch: int = 1

    # ssl training
    # v1: treat certified as GT pesudo labels
    # v2: use corrector loss directly
    ssl_type: str = "v1"
    num_epochs: int = 100
    ssl_lr: float = 3e-4
    ssl_nocs_lr: float = 4e-4
    ssl_recons_lr: float = 3e-4
    ssl_weight_decay: float = 1e-5
    ssl_batch_size: int = 5
    ssl_nocs_clamp_quantile: float = 0.9
    ssl_augmentation: bool = False
    ssl_augmentation_type: str = "random-brightness-contrast"
    ssl_augmentation_gaussian_perturb_std: float = 0.01

    # for v2 SSL
    ssl_sdf_input_trim_quantile: float = 0.7
    ssl_sdf_input_weight: float = 50
    ssl_sdf_nocs_weight: float = 1.0
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
    # degeneracy certificate parameters
    use_degen_cert: bool = False
    cert_degen_min_eig_thres: float = 1e-4
    # set the following parameters to use cert multiplier schedule
    # multiply the depth eps by cert_schedule_multiplier every cert_schedule_period epochs
    cert_schedule_period: int = None
    cert_schedule_multiplier: float = 0.5

    # corrector parameters
    corrector_algo: str = "torch-adam"
    corrector_lr: float = 0.001
    corrector_scale_lr: float = 5e-4
    corrector_nocs_correction_lr: float = 1e-3
    corrector_shape_correction_lr: float = 5e-4
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

    # mf corrector settings
    mf_corrector_mode: str = "gnc"
    mf_corrector_rolling_window_size: int = 50
    mf_corrector_min_buffer_size: int = 25
    mf_geometric_corrector_nocs_sample_size: int = 2000
    mf_geometric_corrector_type: str = "PROJ_GD"
    mf_geometric_corrector_ignore_cert_mask: bool = False
    mf_geometric_corrector_lr: float = 1e-3
    mf_geometric_corrector_max_iters: int = 25
    mf_geometric_corrector_inter_weight: float = 1e2
    mf_geometric_corrector_mnfld_weight: float = 3e3
    mf_geometric_corrector_nonmnfld_points_voxel_res: int = 64
    mf_geometric_corrector_lsq_normalize_F_matrix: bool = False

    # pipeline
    pipeline_nr_downsample_before_corrector: int = 1000
    registration_inlier_thres: float = 0.1
    # frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-upper-scale"
    frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-scale-free"
    # frame_corrector_mode: str = "nocs-only-sdf-input-nocs-scale-free"
    pipeline_output_intermediate_vars: bool = True
    pipeline_output_precrt_results: bool = False
    pipeline_no_grad_model_forward: bool = False
    pipeline_output_degen_condition_numbers: bool = False

    # set to true to run PGO on the obtained poses
    # if corrector is set to True, PGO will be run
    # on the corrected poses
    use_corrector: bool = True
    use_mf_shape_code_corrector: bool = False
    use_mf_geometric_shape_corrector: bool = False

    # loading model & testing
    visualize: bool = False
    visualize_rgb: bool = False
    visualize_test: bool = False
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    checkpoint_path: str = None

    # seed
    fixed_random_seed: bool = True

    profile_runtime: bool = False
    dump_vis: bool = False
    dump_video: bool = False

    # automatically populated if missing
    exp_id: str = None
    pose_noise_scale: float = 0


def main(opt: ExpSettings):
    if opt.fixed_random_seed:
        LF.seed_everything(42)

    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    print(f"Experiment/checkpoint/logs dump path: {exp_dump_path}")

    # one logger to log root dir, another to the exp folder
    tb_logger_path = os.path.join(opt.log_root_dir, "tensorboard")
    safely_make_folders([tb_logger_path])
    tb_logger_root = LF.loggers.TensorBoardLogger(
        root_dir=tb_logger_path, name="joint_model_training", flush_secs=10, max_queue=5
    )
    tb_logger_exp = LF.loggers.TensorBoardLogger(root_dir=exp_dump_path, name="log", flush_secs=10, max_queue=5)

    fabric = L.Fabric(
        accelerator="gpu",
        loggers=[tb_logger_root, tb_logger_exp],
        strategy="auto",
        devices=1,
        num_nodes=1,
        precision=opt.amp_mode,
    )
    fabric.launch()

    # datasets
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

    # model, corrector and pipeline
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

    print(f"Loading shape code library.")
    shape_code_library = None
    if "shape_code_library" in state.keys():
        shape_code_library = state["shape_code_library"]
    else:
        try:
            with open(opt.shape_code_library_path, "rb") as f:
                shape_code_library = pickle.load(f)
        except Exception as e:
            print(f"Error loading shape code library: {e}")

    corrector = None
    if opt.use_corrector:
        print("Pipeline will use a corrector.")
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
            shape_code_library=shape_code_library,
        )

    frame_certifier = None
    if opt.use_frame_certifier:
        print("Pipeline will use a certifier")
        frame_certifier = FrameCertifier(
            model=model,
            depths_clamp_thres=opt.cert_depths_clamp_thres,
            depths_quantile=opt.cert_depths_quantile,
            depths_eps=opt.cert_depths_eps,
            degen_min_eig_thres=opt.cert_degen_min_eig_thres,
            use_degen_cert=opt.use_degen_cert,
            shape_code_library=shape_code_library,
        )

    # Multi-frame shape corrector
    mf_shp_corrector = None
    if opt.use_mf_shape_code_corrector:
        mf_shp_corrector = MultiFrameShapeCorrector(
            mode=opt.mf_corrector_mode,
            min_buffer_size=opt.mf_corrector_min_buffer_size,
            clamping_thres=opt.mf_corrector_clamping_thres,
            rolling_window_size=opt.mf_corrector_rolling_window_size,
        )

    # Multi-frame geometric shape corrector
    mf_geometric_shp_corrector = None
    if opt.use_mf_geometric_shape_corrector:
        mf_geometric_shp_corrector = MultiFrameGeometricShapeCorrector(
            solver=opt.mf_geometric_corrector_type,
            lr=opt.mf_geometric_corrector_lr,
            shape_code_library=shape_code_library,
            nocs_sample_size=opt.mf_geometric_corrector_nocs_sample_size,
            min_buffer_size=opt.mf_corrector_min_buffer_size,
            rolling_window_size=opt.mf_corrector_rolling_window_size,
            mnfld_weight=opt.mf_geometric_corrector_mnfld_weight,
            inter_weight=opt.mf_geometric_corrector_inter_weight,
            max_iters=opt.mf_geometric_corrector_max_iters,
            ignore_cert_mask=opt.mf_geometric_corrector_ignore_cert_mask,
            global_nonmnfld_points_voxel_res=opt.mf_geometric_corrector_nonmnfld_points_voxel_res,
            lsq_normalize_F_matrix=opt.mf_geometric_corrector_lsq_normalize_F_matrix,
            device=fabric.device,
        )

    pipeline = Pipeline(
        model=model,
        corrector=corrector,
        frame_certifier=frame_certifier,
        frame_corrector_mode=opt.frame_corrector_mode,
        pgo_solver=None,
        multi_frame_shape_code_corrector=mf_shp_corrector,
        multi_frame_geometric_shape_corrector=mf_geometric_shp_corrector,
        device=fabric.device,
        nr_downsample_before_corrector=opt.pipeline_nr_downsample_before_corrector,
        sdf_input_loss_multiplier=opt.corrector_loss_multiplier,
        registration_inlier_thres=opt.registration_inlier_thres,
        output_intermediate_vars=opt.pipeline_output_intermediate_vars,
        output_precrt_results=opt.pipeline_output_precrt_results,
        output_degen_condition_number=opt.pipeline_output_degen_condition_numbers,
        shape_code_library=shape_code_library,
        ssl_batch_size=opt.ssl_batch_size,
        ssl_nocs_clamp_quantile=opt.ssl_nocs_clamp_quantile,
        ssl_augmentation=opt.ssl_augmentation,
        ssl_augmentation_type=opt.ssl_augmentation_type,
        ssl_augmentation_gaussian_perturb_std=opt.ssl_augmentation_gaussian_perturb_std,
        no_grad_model_forward=opt.pipeline_no_grad_model_forward,
        normalized_recons=opt.normalized_recons,
        profile_runtime=opt.profile_runtime,
    )
    # hard code unnormalized cube scale for vis
    if opt.normalized_recons:
        pipeline.debug_settings["vis_sdf_grid_scale"] = 2
    else:
        pipeline.debug_settings["vis_sdf_grid_scale"] = 0.251

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

    # prepare optimizers for SSL training
    optim_nocs = torch.optim.SGD(
        model.get_nocs_lr_params_list(nocs_lr=opt.ssl_nocs_lr),
        lr=opt.ssl_nocs_lr,
        weight_decay=opt.ssl_weight_decay,
    )
    optim_recons = torch.optim.SGD(
        model.recons_backbone_adaptor.parameters(),
        lr=opt.ssl_recons_lr,
        weight_decay=opt.ssl_weight_decay,
    )
    pipeline, optim_nocs, optim_recons = fabric.setup(pipeline, optim_nocs, optim_recons)
    optimizers = {"nocs": optim_nocs, "recons": optim_recons}

    if not opt.test_only:
        if opt.scenes_mode == "separate":
            train_sep(pipeline, optimizers, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path)
        elif opt.scenes_mode == "mixed":
            train_mixed(pipeline, optimizers, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path)
        elif opt.scenes_mode == "mixed_mf_shp":
            train_mixed_mf_shp(
                pipeline, optimizers, objects_info, obj_pc_ds, unified_shape_ds, fabric, opt, exp_dump_path
            )
    else:
        print("Testing only mode.")
        results_save_path = opt.artifacts_paths
        test_sep(
            pipeline,
            objects_info,
            obj_pc_ds,
            unified_shape_ds,
            fabric,
            opt,
            results_save_path,
            use_torch_profile=opt.use_torch_profile,
            dump_vis=opt.dump_vis,
            dump_video=opt.dump_video,
        )

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

    if opt.test_only:
        tested_model_name = opt.checkpoint_path.split("/")[-2]
        results_save_path = os.path.join(opt.artifacts_paths, tested_model_name)
        if opt.use_corrector:
            results_save_path = results_save_path + "_sf_corrector"
        if opt.use_mf_geometric_shape_corrector:
            results_save_path = results_save_path + f"_mf_geom_{opt.mf_geometric_corrector_type}_corrector"
        if opt.profile_runtime:
            results_save_path = results_save_path + "_profile-runtime"

        # if results_save_path exists, increment suffix
        results_save_path = uniquify(results_save_path)

        safely_make_folders([results_save_path])
        parser.save(opt, os.path.join(results_save_path, "config.yaml"), overwrite=True)

        opt.artifacts_paths = results_save_path

    opt = ExpSettings(**opt)
    main(opt)
