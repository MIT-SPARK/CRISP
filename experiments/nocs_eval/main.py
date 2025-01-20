import numpy as np
import pickle
import os
import logging
import dataclasses
import torch
from dataclasses import dataclass, field
from jsonargparse import ArgumentParser
import datetime
import torch.utils.data as tchdata
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightning as L
import lightning.fabric as LF

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

from experiments.nocs_ssl.test_impl import test_pipeline


@dataclass
class ExpSettings(BaseExpSettings):
    dataset_dir: str = "./data/NOCS"

    subsets: list = field(default_factory=lambda: ["real"])
    segmentation_results_path: str = "/mnt/jnshi_data/datasets/NOCS/segmentation_results/REAL275"

    # overrides
    image_size: tuple = (480, 640)
    model_type: str = "pipeline"
    normalized_recons = False
    use_umeyama: bool = True


def main(opt: ExpSettings):
    exp_dump_path = os.path.join(opt.model_ckpts_save_dir, opt.exp_id)
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

    # set up models
    model, optimizers, hparams = setup_model_and_optimizer(fabric, opt)

    assert len(opt.subsets) == 1
    if opt.subsets[0] == "real":
        K = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    elif opt.subsets[0] == "camera":
        K = np.array([[577.5, 0, 319.5], [0.0, 577.5, 239.5], [0.0, 0.0, 1.0]])
    else:
        raise ValueError(f"Unknown subset: {opt.subsets[0]}")

    if opt.test_only:
        test_pipeline(
            pipeline=model,
            intrinsics=K,
            segmentation_results_path=opt.segmentation_results_path,
            test_log_dir=exp_dump_path,
            nocs_dataset_path=opt.dataset_dir,
            use_umeyama=opt.use_umeyama,
            dump_video=True,
        )


if __name__ == "__main__":
    """NOCS SSL/testing dataset experiment"""
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
    safely_make_folders([exp_dump_path])
    parser.save(opt, os.path.join(exp_dump_path, "config.yaml"))

    main(ExpSettings(**opt))
