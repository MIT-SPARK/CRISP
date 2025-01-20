from dataclasses import dataclass


@dataclass
class BaseExpSettings:
    """Base Experimental Settings"""

    dataset_dir: str
    model_ckpts_save_dir: str = "./model_ckpts"
    shape_code_library_path: str = "./shape_code_library.npy"
    preload_to_mem: bool = False
    dataloader_workers: int = 10

    # dataset
    normalized_recons: bool = True
    # size of the images returned by dataloader
    image_size: tuple = (256, 256)
    # size of the images backbone uses (automatically scales to this size)
    backbone_input_res: tuple = (420, 420)
    global_nonmnfld_voxel_res: int = 128
    test_split: str = "validation"
    dataset_debug_vis: bool = False

    # backbone model
    use_pretrained_backbone: bool = True
    backbone_model_name: str = "dinov2_vits14"
    backbone_model_path: str = None
    freeze_pretrained_backbone_weights: bool = True
    log_root_dir: str = "logs"

    # implicit recons model
    recons_nonlinearity: str = "sine"
    recons_normalization_type: str = "none"
    recons_loss_type: str = "sdf"
    recons_df_loss_mode: str = "metric"
    recons_shape_code_normalization: str = None
    recons_shape_code_norm_scale: float = 10
    recons_lipschitz_normalization_type: str = None
    recons_lipschitz_lambda: float = 2

    # nocs model
    nocs_network_type: str = "dpt_gnfusion_gnnocs"
    nocs_channels: int = 256
    nocs_lateral_layers_type: str = "spaced"

    # training
    loss_nocs_threshold: float = 0.1
    num_epochs: int = 100
    validate_per_n_epochs: int = 5
    batch_size: int = 32
    optimizer: str = "adam"
    lr: float = 3e-4
    nocs_lr: float = 3e-4
    recons_lr: float = 3e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine-anneal"
    cosine_anneal_period: int = 4000
    cosine_anneal_T_mult: int = 2

    # ssl training
    ssl_type: str = "v1"
    ssl_nocs_lr: float = 4e-4
    ssl_recons_lr: float = 3e-4
    ssl_weight_decay: float = 1e-5
    ssl_batch_size: int = 10
    ssl_nocs_clamp_quantile: float = 0.9
    ssl_augmentation: bool = False
    ssl_augmentation_type: str = "random-brightness-contrast"
    ssl_augmentation_gaussian_perturb_std: float = 0.01

    # for v2 SSL
    ssl_sdf_input_trim_quantile: float = 0.7
    ssl_sdf_input_weight: float = 50
    ssl_sdf_nocs_weight: float = 1.0

    # distributed
    num_devices: int = 1
    num_nodes: int = 1

    # overall loss
    nocs_loss_weight: float = 50.0
    recons_loss_weight: float = 1
    nocs_min: float = float("-inf")
    nocs_max: float = float("inf")

    # loss for recons module
    recons_df_weight: float = 3e3
    recons_inter_weight: float = 2e2
    recons_normal_weight: float = 0
    recons_grad_weight: float = 5e1

    # contrastive regularization
    use_contrastive_regularization: bool = False
    num_classes_per_batch: int = 2
    contrastive_same_object_weight: float = 0.1
    contrastive_different_object_weight: float = 1.0
    contrastive_late_start: bool = False
    contrastive_delay_epochs: int = 30

    # framewise certification parameters
    use_frame_certifier: bool = True
    cert_depths_clamp_thres: float = 10
    cert_depths_quantile: float = 0.9
    cert_depths_eps: float = 0.02

    # degeneracy certificate parameters
    use_degen_cert: bool = False
    cert_degen_min_eig_thres: float = 5e-4

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
    corrector_max_ransac_iters: int = 100
    corrector_sdf_input_loss_weight: float = 1
    corrector_sdf_nocs_loss_weight: float = 10.0
    corrector_trim_quantile: float = 0.9
    corrector_log_loss_traj: bool = False
    corrector_log_dump_dir: str = "./corrector_logs"
    corrector_nesterov: bool = True
    corrector_momentum: float = 0.9
    corrector_loss_multiplier: float = 1

    # mf corrector settings
    mf_corrector_clamping_thres: float = 0.5
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
    frame_corrector_mode: str = "inv-depths-sdf-nocs-input-scale-free"
    # frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-scale-free"
    pipeline_output_intermediate_vars: bool = True
    pipeline_output_precrt_results: bool = False
    pipeline_no_grad_model_forward: bool = False
    pipeline_output_degen_condition_numbers: bool = False

    use_corrector: bool = True
    use_mf_shape_code_corrector: bool = False
    use_mf_geometric_shape_corrector: bool = False
    use_pgo: bool = False

    # loading model & testing
    model_type: str = "joint"
    train_mode: str = "sl"
    test_only: bool = False
    test_on_unseen_objects: bool = False
    test_gt_nocs_pose: bool = False
    gen_mesh_for_test: bool = False
    gen_latent_vecs_for_test: bool = False
    resume_from_ckpt: bool = False
    resume_optimizer_state: bool = False
    checkpoint_path: str = None

    # visualization
    vis_sdf_sample_points: bool = False
    vis_pred_nocs_registration: bool = False
    vis_pred_nocs_and_cad: bool = False
    vis_pred_recons: bool = False
    vis_gt_sanity_test: bool = False
    vis_rgb_images: bool = False
    vis_gt_nocs_heatmap: bool = False
    vis_pred_sdf: bool = False
    calculate_pred_recons_metrics: bool = True
    export_all_pred_recons_mesh: bool = False
    export_average_pred_recons_mesh: bool = False
    artifacts_save_dir: str = "artifacts"

    # seed
    fixed_random_seed: bool = True

    # automatically populated if missing
    exp_id: str = None
