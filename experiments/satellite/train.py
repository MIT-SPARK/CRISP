import copy
import logging
import os
from dataclasses import dataclass
from jsonargparse import ArgumentParser
import lightning as L
import lightning.fabric as LF
import datetime
import torch.utils.data as tchdata
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# local lib imports
from crisp.datasets.spe3r import SatReconNOCSDataset, DistributedObjClassBatchSampler
from crisp.utils.torch_utils import mesh_collate_fn
from experiments.satellite.test_impl import test_model, test_pipeline
from experiments.satellite.train_impl import train_sl, train_ssl
from crisp.models.builder import setup_model_and_optimizer
from crisp.models.loss_functions import (
    siren_udf_loss,
    siren_sdf_fast_loss,
    implicit_loss_helper,
    explicit_loss_helper,
    two_cls_contrastive_explicit_loss_helper,
)
from crisp.config.config import BaseExpSettings
from crisp.utils.file_utils import safely_make_folders, id_generator, uniquify


@dataclass
class ExpSettings(BaseExpSettings):
    spe3r_unified_objects_dataset_dir: str = "./"

    # only used for SSL
    train_split: str = "train"
    test_split: str = "validation"

    # ssl training
    ssl_type: str = "v1"
    ssl_nocs_lr: float = 3e-4
    ssl_recons_lr: float = 3e-4

    # framewise certification parameters
    cert_depths_quantile: float = 0.95
    cert_depths_eps: float = 0.01

    # corrector parameters
    corrector_scale_lr: float = 5e-4
    corrector_nocs_correction_lr: float = 5e-4
    corrector_shape_correction_lr: float = 5e-5
    corrector_sdf_input_loss_weight: float = 1
    corrector_sdf_nocs_loss_weight: float = 1

    # pipeline
    pipeline_nr_downsample_before_corrector: int = 2000
    # frame_corrector_mode: str = "nocs-only-inv-depths-sdf-input-nocs-scale-free"

    dump_video: bool = False


def make_datasets(opt):
    """Helper function to generate the train/val/test set"""
    ssl = opt.train_mode == "ssl"
    if ssl:
        train_split, val_split, test_split = opt.train_split, "validation", opt.test_split
        if opt.train_split == "train":
            raise ValueError("SSL training: use val or test set to train.")
        train_data_to_output = [
            "rgb",
            "depth",
            "instance_segmap",
            "cam_intrinsics",
            "model_name",
        ]
    else:
        train_split, val_split, test_split = "train", "validation", opt.test_split
        train_data_to_output = [
            "rgb",
            "nocs",
            "coords",
            "normals",
            "sdf",
            "instance_segmap",
            "model_name",
        ]

    shape_train_ds = SatReconNOCSDataset(
        # SatReconDataset params
        opt.dataset_dir,
        split=train_split,
        model_name=None,
        input_H=opt.image_size[0],
        input_W=opt.image_size[1],
        # Unified objects_params
        unified_objects_dataset_path=opt.spe3r_unified_objects_dataset_dir,
        normalized_recons=opt.normalized_recons,
        preload_to_mem=False,
        data_to_output=train_data_to_output,
    )
    shape_val_ds = SatReconNOCSDataset(
        # SatReconDataset params
        opt.dataset_dir,
        split=val_split,
        model_name=None,
        input_H=opt.image_size[0],
        input_W=opt.image_size[1],
        # Unified objects_params
        unified_objects=copy.deepcopy(shape_train_ds.unified_objects),
        unified_objects_dataset_path=opt.spe3r_unified_objects_dataset_dir,
        normalized_recons=opt.normalized_recons,
        preload_to_mem=False,
        data_to_output=train_data_to_output,
    )
    test_data_to_output = [
        "rgb",
        "depth",
        "instance_segmap",
        "nocs",
        "coords",
        "normals",
        "object_pc",
        "cam_intrinsics",
        "cam_pose",
        "model_name",
    ]
    if opt.vis_gt_sanity_test or opt.vis_pred_sdf:
        test_data_to_output = [
            "rgb",
            "depth",
            "nocs",
            "coords",
            "normals",
            "object_pc",
            "sdf",
            "cam_intrinsics",
            "cam_pose",
            "instance_segmap",
            "normalized_mesh",
            "sdf_grid",
            "model_name",
        ]

    shape_test_ds = SatReconNOCSDataset(
        # SatReconDataset params
        opt.dataset_dir,
        split=test_split,
        model_name=None,
        input_H=opt.image_size[0],
        input_W=opt.image_size[1],
        # Unified objects_params
        unified_objects=copy.deepcopy(shape_train_ds.unified_objects),
        unified_objects_dataset_path=opt.spe3r_unified_objects_dataset_dir,
        normalized_recons=opt.normalized_recons,
        preload_to_mem=False,
        data_to_output=test_data_to_output,
    )
    return shape_test_ds, shape_train_ds, shape_val_ds


def main(opt: ExpSettings):
    if opt.fixed_random_seed:
        LF.seed_everything(42)
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    artifacts_dump_path = opt.artifacts_save_dir
    safely_make_folders([exp_dump_path, artifacts_dump_path])

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    logging.info(f"Experiment/checkpoint/logs dump path: {exp_dump_path}")
    logging.info(f"Artifacts dump path: {artifacts_dump_path}")
    logging.info(f"Options:")
    logging.info(opt)

    # one logger to log root dir, another to the exp folder
    tb_logger_root = LF.loggers.TensorBoardLogger(
        root_dir=os.path.join(opt.log_root_dir, "tensorboard"), name="joint_model_training", flush_secs=10, max_queue=5
    )
    tb_logger_exp = LF.loggers.TensorBoardLogger(root_dir=exp_dump_path, name="log", flush_secs=10, max_queue=5)

    fabric = L.Fabric(
        accelerator="gpu",
        loggers=[tb_logger_root, tb_logger_exp],
        strategy="auto",
        devices=opt.num_devices,
        num_nodes=opt.num_nodes,
    )
    fabric.launch()

    # build dataset
    shape_test_ds, shape_train_ds, shape_val_ds = make_datasets(opt)

    shuffle = True
    if opt.test_only:
        shuffle = False

    if not opt.use_contrastive_regularization:
        print("Does not use contrastive regularization.")
        test_dl, val_dl, train_dl = [
            tchdata.DataLoader(
                x,
                shuffle=shuffle,
                num_workers=opt.dataloader_workers,
                batch_size=opt.batch_size,
                collate_fn=mesh_collate_fn,
            )
            for x in [shape_test_ds, shape_val_ds, shape_train_ds]
        ]
        test_dl, val_dl, train_dl = fabric.setup_dataloaders(test_dl, val_dl, train_dl)
    else:
        print("Using contrastive regularization.")
        B = int(opt.batch_size / opt.num_classes_per_batch)
        print(f"Using batch size of {B} for each class.")
        splits = [shape_test_ds, shape_val_ds, shape_train_ds]

        dls = []
        for x in splits:
            bsampler = DistributedObjClassBatchSampler(
                per_obj_class_batch_size=B,
                num_classes_per_batch=opt.num_classes_per_batch,
                sat_dataset=x,
                num_replicas=fabric.world_size,
                rank=fabric.global_rank,
            )

            dl = tchdata.DataLoader(
                x,
                num_workers=opt.dataloader_workers,
                collate_fn=mesh_collate_fn,
                batch_sampler=bsampler,
            )
            dls.append(dl)
        test_dl, val_dl, train_dl = fabric.setup_dataloaders(*dls, use_distributed_sampler=False)

    model, optimizers, hparams = setup_model_and_optimizer(fabric, opt)

    scheduler = None
    if opt.scheduler == "cosine-anneal":
        if opt.train_mode == "sl":
            scheduler = CosineAnnealingWarmRestarts(
                optimizers["all"], T_0=opt.cosine_anneal_period, T_mult=opt.cosine_anneal_T_mult
            )
        # no scheduler for SSL
    elif opt.scheduler == "none":
        scheduler = None
    else:
        raise NotImplementedError

    # loss closure to modify behavior & hyperparams of loss functions
    if opt.recons_loss_type == "sdf":
        recons_loss_fn = siren_sdf_fast_loss
    elif opt.recons_loss_type == "udf":
        recons_loss_fn = siren_udf_loss

    if opt.use_contrastive_regularization:
        print("Using contrastive regularization for losses.")
        assert opt.num_classes_per_batch == 2

        def loss_fn(**kwargs):
            return two_cls_contrastive_explicit_loss_helper(**kwargs, opt=opt)

    else:
        if opt.recons_df_loss_mode == "implicit":

            def loss_fn(**kwargs):
                return implicit_loss_helper(**kwargs, recons_loss_fn=recons_loss_fn, opt=opt)

        elif opt.recons_df_loss_mode == "metric":

            def loss_fn(**kwargs):
                return explicit_loss_helper(**kwargs, opt=opt)

        else:
            raise ValueError(f"Invalid recons_df_loss_mode: {opt.recons_df_loss_mode}")

    if not opt.test_only:
        print(f"Training mode (will not run test)")
        if opt.train_mode == "sl":
            train_sl(
                fabric=fabric,
                model=model,
                optimizer=optimizers["all"],
                scheduler=scheduler,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                loss_fn=loss_fn,
                num_epochs=opt.num_epochs,
                validate_per_n_epochs=opt.validate_per_n_epochs,
                model_save_path=exp_dump_path,
                hparams=hparams,
                opt=opt,
            )
        elif opt.train_mode == "ssl":
            train_ssl(
                fabric=fabric,
                pipeline=model,
                optimizers=optimizers,
                scheduler=scheduler,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                loss_fn=loss_fn,
                num_epochs=opt.num_epochs,
                validate_per_n_epochs=opt.validate_per_n_epochs,
                model_save_path=exp_dump_path,
                visualize=False,
                hparams=hparams,
                opt=opt,
            )
    else:
        print(f"Testing mode.")
        model.eval()
        results_save_path = opt.artifacts_save_dir
        if opt.model_type == "joint":
            test_model(
                fabric,
                model,
                test_dl,
                loss_fn=None,
                normalized_recons=opt.normalized_recons,
                vis_sdf_sample_points=opt.vis_sdf_sample_points,
                vis_pred_nocs_registration=opt.vis_pred_nocs_registration,
                vis_pred_nocs_and_cad=opt.vis_pred_nocs_and_cad,
                vis_pred_recons=opt.vis_pred_recons,
                vis_pred_sdf=opt.vis_pred_sdf,
                vis_gt_sanity_test=opt.vis_gt_sanity_test,
                vis_rgb_image=opt.vis_rgb_images,
                vis_gt_nocs_heatmap=opt.vis_gt_nocs_heatmap,
                artifacts_save_dir=results_save_path,
                calculate_pred_recons_metrics=opt.calculate_pred_recons_metrics,
                export_all_pred_recons_mesh=opt.export_all_pred_recons_mesh,
                export_average_pred_recons_mesh=opt.export_average_pred_recons_mesh,
            )
        elif opt.model_type == "pipeline":
            test_pipeline(
                fabric=fabric,
                pipeline=model,
                test_dataloader=test_dl,
                # just test one epoch
                num_epochs=1,
                model_save_path=exp_dump_path,
                artifacts_save_dir=results_save_path,
                visualize=False,
                calculate_test_recons_metrics=opt.calculate_pred_recons_metrics,
                dump_video=opt.dump_video,
            )

    return


if __name__ == "__main__":
    """Joint model training for SEP3R dataset"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    # generate random timestamped experiment ID
    if opt.exp_id is None:
        opt.exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    if not opt.test_only:
        safely_make_folders([exp_dump_path])
        parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))
    if opt.test_only:
        # generate results_save_path
        tested_model_name = opt.checkpoint_path.split("/")[-2]
        results_save_path = os.path.join(opt.artifacts_save_dir, tested_model_name)
        if opt.use_corrector:
            results_save_path = results_save_path + "_sf_corrector"
        if opt.use_mf_geometric_shape_corrector:
            results_save_path = results_save_path + f"_mf_geom_{opt.mf_geometric_corrector_type}_corrector"
        if opt.model_type == "pipeline" and opt.pipeline_output_degen_condition_numbers:
            results_save_path = results_save_path + "_degen"
        results_save_path = results_save_path + "_" + opt.test_split
        # if results_save_path exists, increment suffix
        results_save_path = uniquify(results_save_path)

        safely_make_folders([results_save_path])
        parser.save(opt, os.path.join(results_save_path, "config.yaml"), overwrite=True)

        # update artifacts_save_dir
        opt.artifacts_save_dir = results_save_path

    main(ExpSettings(**opt))
