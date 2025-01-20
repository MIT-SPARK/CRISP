import pickle
import os
import logging
import dataclasses
import torch
from dataclasses import dataclass, field
from jsonargparse import ArgumentParser
import lightning as L
import lightning.fabric as LF
import datetime
import torch.utils.data as tchdata
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# local lib imports
from crisp.config.config import BaseExpSettings
from crisp.datasets import nocs, nocs_config
from crisp.datasets import nocs_utils as nutils
from crisp.models.joint import JointShapePoseNetwork
from crisp.models.loss_functions import (
    siren_udf_loss,
    siren_sdf_fast_loss,
    metric_sdf_loss,
    implicit_loss_helper,
    explicit_loss_helper,
)
from crisp.utils.torch_utils import mesh_collate_fn
from crisp.utils.file_utils import safely_make_folders, id_generator
from crisp.models.builder import setup_model_and_optimizer

from experiments.nocs.train_impl import train_sl, train_ssl
from experiments.nocs.test_impl import test_pipeline, test_model


@dataclass
class ExpSettings(BaseExpSettings):
    dataset_dir: str = "./data/NOCS"

    # subsets options: ["camera", "real"]
    subsets: list = field(default_factory=lambda: ["camera"])
    split: str = "train"

    # stub directory for unified objects (for storing generated SDF caches)
    nocs_unified_objects_dir: str = "./data/unified_renders/NOCS"

    # override
    normalized_recons = False


def our_admissible_objects(subset, obj_label):
    """Manually reject camera_val/02876657/d3b53f56b4a7b3b3c9f016d57db96408"""
    if subset == "camera_val":
        if obj_label == "02876657_d3b53f56b4a7b3b3c9f016d57db96408":
            return False
    return True


def make_cache_filename(subsets, split):
    name = ""
    for subset in subsets:
        name += f"{subset}_{split}_"
    name += "cache.pkl"
    return name


def make_dataset_and_dl(split, subsets, opt: ExpSettings):
    # prepare and load NOCS dataset
    allowed_subsets = ["camera", "real"]
    assert set(subsets) <= set(allowed_subsets)

    camera_dir = os.path.join(opt.dataset_dir, "camera")
    real_dir = os.path.join(opt.dataset_dir, "real")
    obj_dir = os.path.join(opt.dataset_dir, "obj_models")

    # update config
    ds_config = nocs_config.Config()
    ds_config.ROOT_DIR = opt.dataset_dir
    ds_config.OBJ_MODEL_DIR = os.path.join(opt.dataset_dir, "obj_models")
    class_map = {
        "bottle": "bottle",
        "bowl": "bowl",
        "cup": "mug",
        "laptop": "laptop",
    }

    # Look for serialized version of the dataset, load it if it exists, otherwise create it
    cache_fname = make_cache_filename(subsets, split)
    safely_make_folders([os.path.join(opt.dataset_dir, "cache")])
    cache_path = os.path.join(opt.dataset_dir, "cache", cache_fname)

    cache_loaded = False
    if os.path.exists(cache_path):
        try:
            logging.info(f"Loading NOCS {subsets}-{split} dataset from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                dataset_train = pickle.load(f)
                cache_loaded = True
        except Exception as e:
            logging.info(f"Error loading cache: {e}")
            cache_loaded = False
    else:
        logging.info(f"Cache for NOCS {split} dataset at {cache_path} not found.")

    if not cache_loaded:
        logging.info(f"Cache for NOCS {split} dataset not found/loaded. Creating it...")
        dataset_train = nocs.Dataset(
            subset=split, make_yaml=False, admissible_objects=our_admissible_objects, config=ds_config
        )

        if "camera" in subsets:
            logging.info("Loading camera scenes...")
            dataset_train.load_camera_scenes(camera_dir, obj_dir)
        if "real" in subsets:
            logging.info("Loading real scenes...")
            dataset_train.load_real_scenes(real_dir, obj_dir)

        logging.info("All scenes loaded. Preparing dataset...")
        dataset_train.prepare(class_map)
        logging.info("Making obj-image index...")
        dataset_train.make_obj_image_index()
        logging.info("Obj-image index made.")
        logging.info(f"Saving NOCS {split} dataset to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(dataset_train, f)

    # TODO: Move this to cache
    # load ground truth transformations
    dataset_train.load_gt_transforms(os.path.join(opt.dataset_dir, "gt_results"))

    data_to_output = [
        "rgb",
        "nocs",
        "sdf",
        "coords",
        "instance_segmap",
    ]
    # depth output for testing
    if opt.test_only:
        data_to_output.extend(["obj_pose", "depth_pc", "depth", "depth_pc_mask", "metadata"])
    ds = nocs.NOCSDataset(
        dataset_train,
        nocs_objects_dataset_path=obj_dir,
        folder_path=opt.nocs_unified_objects_dir,
        split=split,
        model_img_H=opt.backbone_input_res[0],
        model_img_W=opt.backbone_input_res[1],
        normalized_recons=opt.normalized_recons,
        force_recompute_sdf=False,
        data_to_output=data_to_output,
        debug_vis=False,
    )
    dl = tchdata.DataLoader(
        ds, shuffle=True, num_workers=opt.dataloader_workers, batch_size=opt.batch_size, collate_fn=mesh_collate_fn
    )
    return ds, dl


def main(opt: ExpSettings):
    LF.seed_everything(42)
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    artifacts_dump_path = os.path.join(opt.artifacts_save_dir, opt.exp_id)
    safely_make_folders([exp_dump_path, artifacts_dump_path])
    logging.info(f"Experiment/checkpoint/logs dump path: {exp_dump_path}")
    logging.info(f"Artifacts dump path: {artifacts_dump_path}")
    torch.set_float32_matmul_precision("high")

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

    # make datasets and dataloaders
    train_ds, train_dl = None, None
    val_ds, val_dl = None, None
    test_ds, test_dl = None, None
    if opt.test_only:
        test_ds, test_dl = make_dataset_and_dl(opt.split, opt.subsets, opt)
        test_dl.dataset.data_to_output.extend(["normalized_mesh", "object_pc"])
        test_dl = fabric.setup_dataloaders(test_dl)
    else:
        train_ds, train_dl = make_dataset_and_dl(opt.split, opt.subsets, opt)
        val_ds, val_dl = make_dataset_and_dl("val", ["camera"], opt)
        val_dl, train_dl = fabric.setup_dataloaders(val_dl, train_dl)

    # set up models
    model, optimizers, hparams = setup_model_and_optimizer(fabric, opt)

    # scheduler
    if opt.scheduler == "cosine-anneal":
        if opt.train_mode == "sl":
            scheduler = CosineAnnealingWarmRestarts(
                optimizers["all"], T_0=opt.cosine_anneal_period, T_mult=opt.cosine_anneal_T_mult
            )
    elif opt.scheduler == "none":
        scheduler = None
    else:
        raise NotImplementedError

    # loss closure to modify behavior & hyperparams of loss functions
    if opt.recons_loss_type == "sdf":
        recons_loss_fn = siren_sdf_fast_loss
    elif opt.recons_loss_type == "udf":
        recons_loss_fn = siren_udf_loss

    # fmt: off
    if opt.recons_df_loss_mode == "implicit":
        def loss_fn(**kwargs):
            return implicit_loss_helper(**kwargs, recons_loss_fn=recons_loss_fn, opt=opt)
    elif opt.recons_df_loss_mode == "metric":
        def loss_fn(**kwargs):
            return explicit_loss_helper(**kwargs, opt=opt)
    else:
        raise ValueError(f"Invalid recons_df_loss_mode: {opt.recons_df_loss_mode}")
    # fmt: on

    if not opt.test_only:
        if opt.train_mode == "sl":
            train_sl(
                fabric=fabric,
                model=model,
                optimizer=optimizers["all"],
                scheduler=scheduler,
                loss_fn=loss_fn,
                train_dataloader=train_dl,
                val_dataloader=val_dl,
                num_epochs=opt.num_epochs,
                validate_per_n_epochs=opt.validate_per_n_epochs,
                model_save_path=exp_dump_path,
                hparams=hparams,
                # visualize=["object_pc"],
                visualize=[],
            )
        elif opt.train_mode == "ssl":
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if opt.model_type == "joint":
            test_model(
                fabric=fabric,
                model=model,
                dataloader=test_dl,
                loss_fn=None,
                artifacts_save_dir=artifacts_dump_path,
                normalized_recons=opt.normalized_recons,
                vis_sdf_sample_points=opt.vis_sdf_sample_points,
                vis_pred_nocs_registration=opt.vis_pred_nocs_registration,
                vis_pred_nocs_and_cad=opt.vis_pred_nocs_and_cad,
                vis_pred_recons=opt.vis_pred_recons,
                vis_pred_sdf=opt.vis_pred_sdf,
                vis_gt_sanity_test=opt.vis_gt_sanity_test,
                vis_rgb_image=opt.vis_rgb_images,
                vis_gt_nocs_heatmap=opt.vis_gt_nocs_heatmap,
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
                artifacts_save_dir=opt.artifacts_save_dir,
                visualize=True,
                calculate_test_recons_metrics=opt.calculate_pred_recons_metrics,
            )
        else:
            raise ValueError(f"Use pipeline mode for testing.")

    return


if __name__ == "__main__":
    """NOCS dataset experiment"""
    parser = ArgumentParser()
    parser.add_class_arguments(ExpSettings)
    opt = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    # generate random timestamped experiment ID
    if opt.exp_id is None:
        opt.exp_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M')}_{id_generator(size=5)}"
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
    if not opt.test_only:
        safely_make_folders([exp_dump_path])
        parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))

    main(ExpSettings(**opt))
