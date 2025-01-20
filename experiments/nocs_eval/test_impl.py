import copy
import torch
import torchvision.transforms.v2 as tvt
import os
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
import math
import cmath
from crisp.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import glob
import _pickle as cPickle
import cv2

from crisp.utils.file_utils import safely_make_folders
from crisp.utils.visualization_utils import (
    get_meshes_from_shape_codes,
    generate_video_from_meshes,
    generate_orbiting_video_from_meshes,
)
from crisp.datasets.nocs_utils import load_depth, backproject
from crisp.models.registration import (
    umeyama_ransac_batched,
    umeyama_ransac_batched_inlier_thres_target,
    arun_ransac_batched,
)


def merge_masks(pred_masks):
    """Merge a list of masks into one mask, the ids of each mask is the
    index of the mask in the list.
    """
    final_mask = np.zeros((pred_masks.shape[0], pred_masks.shape[1]), dtype=np.int16)
    for i in range(pred_masks.shape[-1]):
        final_mask = final_mask + pred_masks[:, :, i] * (i + 1)
    return final_mask


def make_objs_info(pred_masks, pred_bboxes, pred_class_ids, pred_scores):
    """Helper function to generate the object info for pipeline"""
    obj_info = [
        {
            "label": pred_class_ids[i],
            "id_in_segm": i + 1,
            # the dataset's bbox format is (y1, x1, y2, x2)
            # we use (x1, y1, x2, y2) for the pipeline
            "bbox": [pred_bboxes[i, 1], pred_bboxes[i, 0], pred_bboxes[i, 3], pred_bboxes[i, 2]],
            "score": pred_scores[i],
        }
        for i in range(pred_masks.shape[-1])
    ]
    return [obj_info]


def make_frames_info(image_path, num_instances):
    """Helper function to generate the frames info for pipeline"""
    return [{"path": image_path, "inst_count": num_instances}]


def test_pipeline(
    pipeline,
    intrinsics,
    segmentation_results_path,
    test_log_dir,
    nocs_dataset_path,
    use_umeyama=True,
    device="cuda",
    dump_video=False,
):
    print(f"Loading segmentation results from {segmentation_results_path}")

    # test
    result_pkl_list = glob.glob(segmentation_results_path + "/results_*.pkl")
    result_pkl_list = sorted(result_pkl_list)
    n_image = len(result_pkl_list)
    print("no. of test images: {}\n".format(n_image))

    rgb_transform = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    for i, path in tqdm(enumerate(result_pkl_list), total=n_image):
        with open(path, "rb") as f:
            data = cPickle.load(f)

        # data keys:
        # 'image_path', 'gt_class_ids', 'gt_bboxes', 'gt_RTs', 'gt_scales',
        # 'pred_class_ids', 'pred_bboxes', 'pred_scores', 'pred_masks', 'gt_handle_visibility'
        image_path = data["image_path"]
        num_instance = len(data["pred_class_ids"])

        # replace image path saved in seg with the one we have
        image_path = image_path.replace("data", nocs_dataset_path)

        image = cv2.imread(image_path + "_color.png")[:, :, :3]
        image = image[:, :, ::-1].copy()
        depth = load_depth(image_path) / 1000
        pred_mask = data["pred_masks"]

        pred_RTs = np.zeros((num_instance, 4, 4))
        pred_scales = np.zeros((num_instance, 3))
        pred_RTs[:, 3, 3] = 1

        merged_mask = merge_masks(pred_mask)
        objs_info = make_objs_info(pred_mask, data["pred_bboxes"], data["pred_class_ids"], data["pred_scores"])
        frames_info = make_frames_info(image_path, num_instance)
        rgb_tensor = rgb_transform(image)
        payload = pipeline(
            rgbs=rgb_tensor.unsqueeze(0).to(device=device),
            masks=torch.tensor(merged_mask, device=device)[None, None, ...],
            depths=torch.tensor(depth.astype(np.float32), device=device)[None, None, ...],
            intrinsics=torch.tensor(intrinsics, device=device).reshape((1, 1, 3, 3)),
            objs=objs_info,
            frames_info=frames_info,
        )

        # umeyama alg
        if use_umeyama:
            cam_s_nocs, cam_R_nocs, cam_t_nocs, _, status = umeyama_ransac_batched(
                payload["postcrt_nocs"],
                payload["pcs"],
                masks=payload["masks"],
                inlier_thres=payload["reg_inlier_thres"],
                confidence=0.99,
                max_iters=100,
            )
        else:
            cam_s_nocs, cam_R_nocs, cam_t_nocs = payload["cam_s_nocs"], payload["cam_R_nocs"], payload["cam_t_nocs"]

        cam_s_nocs, cam_R_nocs, cam_t_nocs = (
            cam_s_nocs.cpu().detach().numpy(),
            cam_R_nocs.cpu().detach().numpy(),
            cam_t_nocs.cpu().detach().numpy(),
        )

        if dump_video:
            frame_info = frames_info[0]
            imname = "_".join(image_path.split("/")[-3:])
            # generate meshes
            pred_meshes = get_meshes_from_shape_codes(pipeline.model, payload["shape_code"], mesh_recons_scale=0.35)

            # transform meshes to camera frame
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

            def cam_traj_fn(i, pl):
                if i == 0:
                    pl.camera_position = "xy"
                    pl.camera.azimuth = 10
                    pl.camera.roll = 180
                    pl.camera.elevation = 30
                else:
                    pl.camera.azimuth = 10 + i * 2
                    pl.camera.roll = 180 - i * 0.025
                    pl.camera.elevation = 30 - i * 0.25

            folder_name = f"artifacts/video_dump_{imname}"
            safely_make_folders([folder_name])
            video_path = os.path.join(folder_name, f"mesh_video.mp4")
            generate_video_from_meshes(transformed_meshes, cam_traj_fn, 100, video_path)

            # dump rgb
            rgb_pil = Image.fromarray(image)
            rgb_path = os.path.join(folder_name, f"rgb_{imname}.jpg")
            rgb_pil.save(rgb_path)

            # dump rgb with mask
            mask_pil = Image.fromarray(merged_mask.astype(np.int16) * 255)
            # invert_mask = PIL.ImageOps.invert(mask_pil)
            rgb_masked = Image.blend(mask_pil.convert("RGB"), rgb_pil, 0.5)
            rgb_masked_path = os.path.join(folder_name, f"rgb_masked_{imname}.jpg")
            rgb_masked.save(rgb_masked_path)

        if num_instance != 0:
            for j in range(num_instance):
                inst_mask = 255 * pred_mask[:, :, j].astype("uint8")
                pts_ori, idx = backproject(depth, intrinsics, inst_mask)
                # pts_ori: (N, 3)
                pts_ori = pts_ori / 1000.0

                if j in payload["obj_index"]:
                    jj = payload["obj_index"].index(j)

                    # [x,y,z], size of the 3D bbox centered at the origin for NOCS
                    pred_size = (
                        torch.max(payload["postcrt_nocs"][jj], dim=1)[0]
                        - torch.min(payload["postcrt_nocs"][jj], dim=1)[0]
                    )
                    pred_size = pred_size.cpu().detach().numpy()

                    pred_RTs[j, :3, :3] = np.diag(np.ones(3) * cam_s_nocs[jj].item()) @ cam_R_nocs[jj, ...]
                    pred_RTs[j, :3, 3] = cam_t_nocs[jj, ...].flatten()
                    pred_scales[j] = pred_size
                else:
                    pred_RTs[j] = np.eye(4)
                    pred_scales[j] = np.ones(3)

        data.pop("pred_masks")
        data["pred_RTs"] = pred_RTs
        data["pred_scales"] = pred_scales

        with open(os.path.join(test_log_dir, path.split("/")[-1]), "wb") as f:
            cPickle.dump(data, f)
