import pickle
import dataclasses

from crisp.models.mf_corrector import MultiFrameShapeCorrector, MultiFrameGeometricShapeCorrector

# local lib imports
from crisp.models.pipeline import Pipeline
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.corrector import *
from crisp.models.certifier import FrameCertifier


def setup_model_and_optimizer(fabric, opt):
    """Helper function to set up the model and optimizer"""
    pipeline, optimizers = None, None

    # joint nocs & recons model
    model = JointShapePoseNetwork(
        input_dim=3,
        recons_num_layers=5,
        recons_hidden_dim=256,
        backbone_model=opt.backbone_model_name,
        local_backbone_model_path=opt.backbone_model_path,
        freeze_pretrained_weights=opt.freeze_pretrained_backbone_weights,
        backbone_input_res=opt.backbone_input_res,
        nonlinearity=opt.recons_nonlinearity,
        normalization_type=opt.recons_normalization_type,
        nocs_network_type=opt.nocs_network_type,
        nocs_channels=opt.nocs_channels,
        lateral_layers_type=opt.nocs_lateral_layers_type,
        normalize_shape_code=opt.recons_shape_code_normalization,
        recons_shape_code_norm_scale=opt.recons_shape_code_norm_scale,
    )

    # set backbone to eval mode (if pretrained)
    if opt.freeze_pretrained_backbone_weights:
        print("Setting backbone to eval mode.")
        model.backbone.eval()

    # create optimizer
    lr = None
    if opt.train_mode == "ssl":
        nocs_lr, recons_lr, weight_decay = opt.ssl_nocs_lr, opt.ssl_recons_lr, opt.ssl_weight_decay
        print(f"SSL: NOCS LR={nocs_lr}, Recons LR={recons_lr}, Weight Decay={weight_decay}")
    else:
        lr, nocs_lr, recons_lr, weight_decay = opt.lr, opt.nocs_lr, opt.recons_lr, opt.weight_decay
        print(f"SL: LR={lr}, NOCS LR={nocs_lr}, Recons LR={recons_lr}, Weight Decay={weight_decay}")

    assert opt.optimizer in ["sgd", "adam"]
    optim_func = torch.optim.SGD if opt.optimizer == "sgd" else torch.optim.Adam
    print(f"Use {opt.optimizer} for training.")
    optimizers = {}
    if opt.train_mode == "sl":
        optim_all = optim_func(
            model.get_lr_params_list(nocs_lr=nocs_lr, recons_lr=recons_lr),
            lr=lr,
            weight_decay=weight_decay,
        )
        optimizers["all"] = optim_all
    else:
        optimizers["nocs"] = optim_func(
            model.get_nocs_lr_params_list(nocs_lr=opt.ssl_nocs_lr),
            lr=opt.ssl_nocs_lr,
            weight_decay=opt.ssl_weight_decay,
        )
        optimizers["recons"] = optim_func(
            model.recons_backbone_adaptor.parameters(),
            lr=opt.ssl_recons_lr,
            weight_decay=opt.ssl_weight_decay,
        )

    hparams = dataclasses.asdict(opt)
    state = None
    if opt.test_only or opt.resume_from_ckpt or opt.train_mode == "ssl":
        print(f"Loading model checkpoint at {opt.checkpoint_path}." + (" For SSL." if opt.train_mode == "ssl" else ""))
        state = fabric.load(opt.checkpoint_path)
        model.load_state_dict(state["model"])

    # we resume optimizer state only if we are not testing and not in SSL mode
    if opt.resume_from_ckpt and not opt.test_only and not opt.train_mode == "ssl":
        hparams = state["hparams"]

    # load optimizer state if we want to continue training
    if opt.resume_optimizer_state:
        if opt.train_mode == "ssl":
            raise NotImplementedError
        else:
            if state is not None and "optimizer" in state.keys():
                print(f"Reloading optimizer state.")
                optimizers["all"].load_state_dict(state["optimizer"])
            else:
                print(f"No checkpoint provided/No optimizer state found in the checkpoint. Skipping.")

    print(f"Loading shape code library.")
    shape_code_library = None
    if state is not None and "shape_code_library" in state.keys():
        shape_code_library = state["shape_code_library"]
    else:
        try:
            with open(opt.shape_code_library_path, "rb") as f:
                shape_code_library = pickle.load(f)
        except Exception as e:
            print(f"Error loading shape code library: {e}")

    if opt.model_type == "joint":
        print(f"Use the JointPoseShapeNetwork model in training.")
        pipeline = model
    elif opt.model_type == "pipeline":
        print(f"Use the full pipeline in training.")
        corrector = None
        if opt.use_corrector:
            print("Use corrector for training.")
            corrector = JointCorrector(
                model=model,
                solver_algo=opt.corrector_algo,
                device=fabric.device,
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

        mf_shp_corrector = None
        if opt.use_mf_shape_code_corrector:
            mf_shp_corrector = MultiFrameShapeCorrector(
                mode=opt.mf_corrector_mode,
                min_buffer_size=opt.mf_corrector_min_buffer_size,
                clamping_thres=opt.mf_corrector_clamping_thres,
                rolling_window_size=opt.mf_corrector_rolling_window_size,
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
            input_H=opt.image_size[0],
            input_W=opt.image_size[1],
            nr_downsample_before_corrector=opt.pipeline_nr_downsample_before_corrector,
            sdf_input_loss_multiplier=opt.corrector_loss_multiplier,
            registration_inlier_thres=opt.registration_inlier_thres,
            output_intermediate_vars=opt.pipeline_output_intermediate_vars,
            output_precrt_results=opt.pipeline_output_precrt_results,
            ssl_batch_size=opt.ssl_batch_size,
            ssl_nocs_clamp_quantile=opt.ssl_nocs_clamp_quantile,
            ssl_augmentation=opt.ssl_augmentation,
            ssl_augmentation_type=opt.ssl_augmentation_type,
            ssl_augmentation_gaussian_perturb_std=opt.ssl_augmentation_gaussian_perturb_std,
            no_grad_model_forward=opt.pipeline_no_grad_model_forward,
            normalized_recons=opt.normalized_recons,
            shape_code_library=shape_code_library,
            output_degen_condition_number=opt.pipeline_output_degen_condition_numbers,
            # dataloader will output normalized images
            normalize_input_image=False,
        )

        # prepare for SSL training
        pipeline.freeze_backbone_weights()
        pipeline.freeze_sdf_decoder_weights()

    if opt.train_mode == "ssl":
        pipeline, optimizers["nocs"], optimizers["recons"] = fabric.setup(
            pipeline, optimizers["nocs"], optimizers["recons"]
        )
    else:
        pipeline, optimizers["all"] = fabric.setup(pipeline, optimizers["all"])

    if opt.train_mode == "ssl":
        try:
            pipeline.mark_forward_method("ssl_step")
        except:
            print("No mark_forward_method for Fabric module. Fabric may complain about calling model outside forward.")

    return pipeline, optimizers, hparams
