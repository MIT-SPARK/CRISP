"""
This implements various visualization functions that are used in our code.

"""
import copy
import logging
import pathlib

import numpy as np
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import torch
import torchvision.transforms.functional as F
import trimesh.points
import trimesh.creation
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from tqdm import tqdm

import crisp.utils as utils
from crisp.utils.file_utils import safely_make_folders
from crisp.utils.general import pos_tensor_to_o3d
from crisp.utils.sdf import create_sdf_samples_generic, convert_sdf_samples_to_mesh


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Credit: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_point_cloud_and_mesh(point_cloud, mesh_model=None):
    """
    point_cloud      : torch.tensor of shape (3, m)

    """
    point_cloud = pos_tensor_to_o3d(pos=point_cloud)
    point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
    point_cloud.estimate_normals()
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coor_frame.scale(100, coor_frame.get_center())
    # breakpoint()
    if mesh_model is None:
        o3d.visualization.draw_geometries([point_cloud, coor_frame])
    else:
        mesh_model.compute_vertex_normals()
        mesh_model.paint_uniform_color([0.8, 0.1, 0.1])
        o3d.visualization.draw_geometries([point_cloud, mesh_model, coor_frame])

    return None


def visualize_bop_obj_point_clouds_in_frame(scene_pc, objs_in_frame, mesh_db):
    """Visualize object point clouds in one frame"""
    N_objs = len(objs_in_frame)
    for i in range(N_objs):
        obj_label = objs_in_frame[i]["name"]
        T = objs_in_frame[i]["TCO"]
        obj_pc = objs_in_frame[i]["point_cloud"]

        # CAD model
        mesh_path = mesh_db.infos[obj_label]["mesh_path"]
        obj = o3d.io.read_point_cloud(mesh_path)
        cam_obj = obj.scale(1.0e-3, [0, 0, 0])
        cam_obj = cam_obj.transform(np.array(T))

        # visualize
        pcs_to_vis = [scene_pc, obj_pc, cam_obj]
        colors = [[0.0, 0.0, 0.8], [0.0, 0.8, 0.0], [0.8, 0.0, 0.0]]  # scene  # masked scene  # objects
        visualize_pcs(pcs=pcs_to_vis, colors=colors)

    return


def visualize_batched_bop_point_clouds(scene_pcs, masked_scene_pcs, batched_obj_labels, mesh_db, Ts):
    """Visualize based point clouds

    Args:
        scene_pcs: (B, 3, N)
        masked_scene_pcs: in the same frame as scene_pcs
        obj_pcs: T @ obj_pc is in the same frame as scene_pcs
        Ts:
    """
    batch_size = scene_pcs.shape[0]
    assert batch_size == masked_scene_pcs.shape[0]
    assert batch_size == batched_obj_labels.shape[0]
    assert batch_size == Ts.shape[0]

    for b in range(batch_size):
        obj_label = batched_obj_labels[b]
        mesh_path = mesh_db.infos[obj_label]["mesh_path"]
        obj = o3d.io.read_point_cloud(mesh_path)
        # note: mesh db's mesh is in mm, and we assume Ts are in m
        # so we need to scale the models down here
        cam_obj = obj.scale(1.0e-3, [0, 0, 0])
        cam_obj = cam_obj.transform(np.array(Ts[b, ...].detach().to("cpu")))
        pcs_to_vis = [scene_pcs[b, ...], masked_scene_pcs[b, ...], cam_obj]
        colors = [[0.0, 0.0, 0.8], [0.0, 0.8, 0.0], [0.8, 0.0, 0.0]]  # scene  # masked scene  # objects
        visualize_pcs(pcs=pcs_to_vis, colors=colors)


def visualize_pcs_pyvista(pcs, colors, pt_sizes=None, bg_color="grey", show_axes=True):
    assert len(pcs) == len(colors)
    if pt_sizes is None:
        pt_sizes = [10.0] * len(pcs)
    pl = pv.Plotter(shape=(1, 1))
    pl.set_background(bg_color)
    for pc, color, pt_size in zip(pcs, colors, pt_sizes):
        if pc.shape[0] == 3:
            pv_pc = pv.PolyData(torch.as_tensor(pc.T).cpu().numpy())
        else:
            pv_pc = pv.PolyData(torch.as_tensor(pc).cpu().numpy())
        pl.add_mesh(pv_pc, color=color, point_size=pt_size, render_points_as_spheres=True, opacity=0.5)
    if show_axes:
        pl.show_grid()
        pl.show_axes()
    pl.show()


def visualize_pcs_pyvista_headless(pcs, colors, pt_sizes=None, bg_color="grey"):
    assert len(pcs) == len(colors)
    if pt_sizes is None:
        pt_sizes = [10.0] * len(pcs)
    pl = pv.Plotter(off_screen=True)
    pl.set_background(bg_color)
    for pc, color, pt_size in zip(pcs, colors, pt_sizes):
        if pc.shape[0] == 3:
            pv_pc = pv.PolyData(torch.as_tensor(pc.T).cpu().numpy())
        else:
            pv_pc = pv.PolyData(torch.as_tensor(pc).cpu().numpy())
        pl.add_mesh(pv_pc, color=color, point_size=pt_size, render_points_as_spheres=True, opacity=0.5)
    pl.show_grid()
    pl.show_axes()
    image = pl.screenshot(None, return_img=True)
    return image


def visualize_pcs(pcs, colors=None):
    """Visualize point clouds with objects transformed"""
    geo_list = []
    if colors is None:
        colors = [None] * len(pcs)
    for pc, color in zip(pcs, colors):
        if torch.is_tensor(pc):
            pc = pc.detach().to("cpu")
            pc_o3d = pos_tensor_to_o3d(pos=pc)
        elif isinstance(pc, np.ndarray):
            pc = torch.as_tensor(pc)
            pc_o3d = pos_tensor_to_o3d(pos=pc)
        else:
            pc_o3d = pc
        if color is not None:
            pc_o3d.paint_uniform_color(color)
        geo_list.append(pc_o3d)
    o3d.visualization.draw_geometries(geo_list)


def imgs_show(imgs, unnormalize=False):
    """Draw images on screen (using matplotlib)"""
    if torch.is_tensor(imgs):
        imgs = [imgs[i, ...] for i in range(imgs.shape[0])]
    elif not isinstance(imgs, list):
        imgs = [imgs]
    #fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if img.shape[-1] == 3 or img.shape[-1] == 1:
            img = torch.as_tensor(img).detach().permute(2, 0, 1)
        else:
            img = torch.as_tensor(img).detach()
        if unnormalize:
            img[0, ...] = img[0, ...] * 0.229 + 0.485
            img[1, ...] = img[1, ...] * 0.224 + 0.456
            img[2, ...] = img[2, ...] * 0.225 + 0.406
        img = F.to_pil_image(img)
        img.show()
    #    axs[0, i].imshow(np.asarray(img))
    #    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #plt.show()


def visualize_rgb_bboxes(rgb, bboxes, show=False):
    """Visualize bounding boxes on an RGB frame"""
    # plot the boxes and the labels
    image_with_boxes = draw_bounding_boxes(rgb, boxes=bboxes, width=4)
    if show:
        imgs_show(image_with_boxes)
    return image_with_boxes


def visualize_rgb_segmentation(rgb, masks, alpha=0.7, show=False):
    """Visualize masks on an RGB frame"""
    images_with_segmentation = draw_segmentation_masks(rgb, masks=masks, alpha=alpha)
    if show:
        imgs_show(images_with_segmentation)
    return images_with_segmentation


def create_o3d_spheres(points, color, r=0.01):
    """Turn points into Open3D sphere meshes for visualization"""
    if points.shape[0] == 3:
        tgt_pts = copy.deepcopy(points.numpy().transpose())
    elif points.shape[1] == 3:
        tgt_pts = copy.deepcopy(points.numpy())
    else:
        raise ValueError("Incorrect input dimensions.")
    spheres = []
    for xyz_idx in range(len(tgt_pts)):
        kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
        kpt_mesh.translate(tgt_pts[xyz_idx])
        kpt_mesh.paint_uniform_color(color)
        spheres.append(kpt_mesh)
    return spheres


def visualize_gt_and_pred_keypoints(input_point_cloud, kp_gt, kp_pred=None, pc_pred=None, meshes=None, radius=0.01):
    """Visualize ground truth keypoints
    Ground truth keypoints are green, predicted keypoints are blue
    """
    for b in range(input_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu")
        kp = kp_gt[b, ...].clone().detach().to("cpu")

        # gt keypoints
        o3d_gt_keypoints = create_o3d_spheres(kp, color=[0, 1, 0], r=radius)

        # input point cloud
        o3d_input_point_cloud = pos_tensor_to_o3d(pc, color=[0, 1, 0])

        pcs_to_vis = [o3d_input_point_cloud]
        pcs_to_vis.extend(o3d_gt_keypoints)

        if pc_pred is not None:
            c_pc_pred = pc_pred[b, ...].clone().detach().to("cpu")
            pcs_to_vis.append(pos_tensor_to_o3d(c_pc_pred, color=[1.0, 0, 1.0]))

        # predicted keypoints
        if kp_pred is not None:
            kp_p = kp_pred[b, ...].clone().detach().to("cpu")
            o3d_pred_keypoints = create_o3d_spheres(kp_p, color=[0, 0, 1], r=radius)
            pcs_to_vis.extend(o3d_pred_keypoints)

        if meshes is not None:
            pcs_to_vis.append(meshes[b])
            visualize_pcs(pcs_to_vis)

    return


def visualize_gt_and_pred_keypoints_w_trimesh(
    input_point_cloud=None,
    kp_gt=None,
    pc_gt=None,
    kp_pred=None,
    pc_pred=None,
    meshes=None,
    radius=0.01,
    save_render=False,
    save_render_path="./",
    render_name="render",
):
    """Visualize ground truth keypoints
    Ground truth keypoints are green, predicted keypoints are blue
    """
    for b in range(input_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu").numpy()
        scene = trimesh.scene.Scene()
        trimesh_input_point_cloud = trimesh.points.PointCloud(
            pc.T, colors=np.repeat([[0, 0, 255]], pc.shape[1], axis=0)
        )
        scene.add_geometry(trimesh_input_point_cloud)

        if kp_gt is not None:
            kp = kp_gt[b, ...].clone().detach().to("cpu").numpy()
            gt_kpts = points2trimesh_spheres(kp, r=radius, color=[255, 0, 0])
            for s in gt_kpts:
                scene.add_geometry(s)

        if pc_gt is not None:
            c_pc_gt = pc_gt[b, ...].clone().detach().to("cpu").numpy()
            trimesh_pc_gt = trimesh.points.PointCloud(
                c_pc_gt.T, colors=np.repeat([[0, 0, 255]], c_pc_gt.shape[1], axis=0)
            )
            scene.add_geometry(trimesh_pc_gt)

        if pc_pred is not None:
            c_pc_pred = pc_pred[b, ...].clone().detach().to("cpu").numpy()
            trimesh_pc_pred = trimesh.points.PointCloud(c_pc_pred.T)
            scene.add_geometry(trimesh_pc_pred)

        if kp_pred is not None:
            kp_p = kp_pred[b, ...].clone().detach().to("cpu").numpy()
            pred_kpts = points2trimesh_spheres(kp_p, r=radius, color=[0, 0, 255])
            for s in pred_kpts:
                scene.add_geometry(s)

        if meshes is not None:
            scene.add_geometry(meshes[b])

        # scene.show(viewer="gl", line_settings={"point_size": 15})
        corners = scene.bounds_corners
        t_r = scene.camera.look_at(corners["geometry_0"], distance=2)
        scene.camera_transform = t_r

        if not save_render:
            scene.show()
        else:
            png = scene.save_image(viewer="gl", line_settings={"point_size": 15})
            with open(os.path.join(save_render_path, render_name + ".png"), "wb") as f:
                f.write(png)
                f.close()

    return


def points2trimesh_spheres(x, r=0.05, color=None):
    """Convert 3 by N matrices to trimesh spheres"""
    N = x.shape[1]
    spheres = []
    for n in range(N):
        s = trimesh.creation.icosphere(radius=r, color=color)
        s.vertices += x[:, n].T
        spheres.append(s)
    return spheres


def visualize_c3po_outputs(input_point_cloud, outputs_data, model_keypoints):
    """Visualize outputs from C3PO

    Predicted point cloud: Blue
    Predicted model keypoints: Red
    Detected keypoints: Green
    """
    if len(outputs_data) == 5:
        predicted_point_cloud, corrected_keypoints, R, t, correction = outputs_data
        predicted_model_keypoints = R @ model_keypoints + t
    elif len(outputs_data) == 6:
        # explanations of the different variables
        # predicted_point_cloud: CAD models transformed by estimated R & t
        # corrected_keypoints: detected keypoints after correction
        # R: estimated rotation
        # t: estimated translation
        # correction: corrected amounts after corrector
        # predicted_model_keypoints: model keypoints transformed by estimated R & t
        predicted_point_cloud, corrected_keypoints, R, t, correction, predicted_model_keypoints = outputs_data
    else:
        raise ValueError("Unknown C3PO data.")

    # predicted point cloud: CAD model points transformed using estimted R & t
    # predicted keypoints: keypoints detected on the input point cloud (with corrector's correction)
    # R, t: estimated pose
    # correction: amount of correction
    # predicted model keypoints: model keypoints after transformation using R, t

    for b in range(predicted_point_cloud.shape[0]):
        pc = input_point_cloud[b, ...].clone().detach().to("cpu")
        kp = corrected_keypoints[b, ...].clone().detach().to("cpu")
        kp_p = predicted_model_keypoints[b, ...].clone().detach().to("cpu")
        pc_p = predicted_point_cloud[b, ...].clone().detach().to("cpu")

        # making O3D meshes
        # predicted point cloud
        o3d_predicted_point_cloud = pos_tensor_to_o3d(pc_p)
        o3d_predicted_point_cloud.paint_uniform_color(color=[0, 0, 1])

        # predicted model keypoints
        o3d_predicted_model_keypoints = create_o3d_spheres(kp_p, color=[1, 0, 0])

        # detected keypoints (after correction)
        o3d_detected_keypoints = create_o3d_spheres(kp, color=[0, 1, 0])

        # input point cloud
        o3d_input_point_cloud = pos_tensor_to_o3d(pc)

        pcs_to_vis = [o3d_predicted_point_cloud, o3d_input_point_cloud]
        pcs_to_vis.extend(o3d_detected_keypoints)
        pcs_to_vis.extend(o3d_predicted_model_keypoints)

        visualize_pcs(pcs_to_vis)


def visualize_cosypose_input_detections(rgbs, detection_inputs, show=False):
    """Visualize the detection inputs to CosyPose coarse+refine model"""
    imgs_w_bbox_drawn = []
    for det_id in range(detection_inputs.infos["batch_im_id"].shape[0]):
        c_im_id = detection_inputs.infos["batch_im_id"][det_id]
        c_bbox = detection_inputs.bboxes[det_id, :].cpu()
        imgs_w_bbox_drawn.append(visualize_rgb_bboxes(rgbs[c_im_id, ...], c_bbox.view(1, 4), show=show))
    return imgs_w_bbox_drawn


def overlay_image(rgb_input, rgb_rendered):
    """
    Overlay a rendered RGB mask on another RGB image
    Assume the color channels of the rgb_rendered
    """
    rgb_input = np.asarray(rgb_input)
    rgb_rendered = np.asarray(rgb_rendered)
    assert rgb_input.dtype == np.uint8 and rgb_rendered.dtype == np.uint8
    mask = ~(rgb_rendered.sum(axis=-1) == 0)

    overlay = np.zeros_like(rgb_input)
    overlay[~mask] = rgb_input[~mask] * 0.6 + 255 * 0.4
    overlay[mask] = rgb_rendered[mask] * 0.8 + 255 * 0.2
    # overlay[mask] = rgb_rendered[mask] * 0.3 + rgb_input[mask] * 0.7

    return overlay


def overlay_mask(rgb_input, masks):
    """
    Overlay boolean mask(s) on another RGB image.

    Args:
        rgb_input: (3, H, W)
        mask: (H, W)
    """
    rgb_input = np.asarray(rgb_input)
    overlay = np.copy(rgb_input)
    for i, mask in enumerate(masks):
        mask_input = np.asarray(mask, dtype=np.uint8)
        assert rgb_input.dtype == np.uint8 and mask_input.dtype == np.uint8

        shift = 0.3 * (i + 1)
        overlay[..., ~mask] = overlay[..., ~mask] * 0.5
        overlay[..., mask] = overlay[..., mask] * (1 - shift) + 255 * shift
        # overlay[..., mask] = 255 * shift

    return overlay


def render_cosypose_prediction_wrt_camera(renderer, pred, camera=None, resolution=(640, 480)):
    pred = pred.cpu()
    camera.update(TWC=np.eye(4))

    list_objects = []
    for n in range(len(pred)):
        row = pred.infos.iloc[n]
        obj = dict(name=row.label, color=(1, 1, 1, 1), TWO=pred.poses[n].detach().numpy())
        list_objects.append(obj)
    rgb_rendered = renderer.render_scene(list_objects, [camera])
    return rgb_rendered


def visualize_cosypose_output(rgbs, preds, K, renderer, show=False):
    """Render Cosypose results"""
    imgs_w_det_drawn = []
    unique_im_ids = np.unique(preds.infos["batch_im_id"])
    assert len(unique_im_ids) == K.shape[0]
    assert rgbs.shape[0] == K.shape[0]
    for c_im_id in range(K.shape[0]):
        c_K = K[c_im_id, ...].detach().numpy()
        camera = dict(K=c_K, resolution=rgbs[c_im_id, ...].shape[-2:], TWC=np.eye(4))
        # select predictions on current frame
        mask = preds.infos["batch_im_id"] == c_im_id
        keep_ids = np.where(mask)[0]
        c_preds = preds[keep_ids]
        rgb_rendered = render_cosypose_prediction_wrt_camera(renderer=renderer, pred=c_preds, camera=camera)[0]["rgb"]
        if show:
            imgs_show(rgb_rendered.transpose((2, 0, 1)))

        # overlay picture
        overlaid_image = overlay_image(rgbs[c_im_id, ...].detach().cpu().numpy().transpose((1, 2, 0)), rgb_rendered)
        if show:
            imgs_show(overlaid_image.transpose((2, 0, 1)))

        imgs_w_det_drawn.append(overlaid_image)

    return imgs_w_det_drawn


def visualize_det_and_pred_masks(rgbs, batch_im_id, det_masks, pred_masks, show=False, cert_scores=None):
    """Visualize the detected and rendered/predicted masks"""
    # 2x2 grid
    # 1. det mask only
    # 2. pred mask only
    # 3. det mask & pred mask
    imgs_w_det_drawn = []
    unique_im_ids = np.unique(batch_im_id)
    assert len(unique_im_ids) == rgbs.shape[0]
    for c_im_id in range(rgbs.shape[0]):
        keep_ids_mask = batch_im_id == c_im_id
        keep_ids = np.where(keep_ids_mask)[0]
        for kk in keep_ids:
            det_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [det_masks[kk, ...].detach().cpu().numpy()],
            )
            pred_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [pred_masks[kk, ...].detach().cpu().numpy()],
            )
            both_overlay = overlay_mask(
                (rgbs[c_im_id, ...].detach().cpu().numpy() * 255).astype("uint8"),
                [det_masks[kk, ...].detach().cpu().numpy(), pred_masks[kk, ...].detach().cpu().numpy()],
            )

            if cert_scores is not None:
                logging.info(f"Mask scores (for certification): {cert_scores[kk]}")

            if show:
                imgs_show([det_overlay, pred_overlay, both_overlay])

    return


def visualize_bop_rgb_obj_masks(rgb, bboxes, masks=None, alpha=0.7, show=False):
    """Visualize object RGB masks on one image

    Args:
        rgb:
        bboxes:
        masks:
        alpha:
        show:
    """
    if rgb.dtype is torch.float:
        image = (rgb.clone().detach() * 255.0).to(dtype=torch.uint8)
    else:
        image = rgb.clone().to(dtype=torch.uint8)

    # plot bboxes on images
    image = visualize_rgb_bboxes(image, bboxes, show=False)

    # plot segmentation mask on images
    if masks is not None:
        image = visualize_rgb_segmentation(image, masks=masks, alpha=alpha, show=False)
    if show:
        imgs_show(image)
    return image


def visualize_batched_bop_masks(rgbs, bboxes, masks=None, alpha=0.7, show=False):
    """Helper function to visualize batched data returned from a PoseData dataloader
    for the BOP dataset.

    Args:
        rgbs:
        bboxes: (B, 4)
        masks: batched tensors with uint, each id indicate a new object type
        alpha:
        show:
    """
    assert rgbs.shape[0] == masks.shape[0]
    batch_size = rgbs.shape[0]
    for b in range(batch_size):
        logging.info(f"Visualizing image-{b}")
        # plot bboxes on images
        if rgbs.dtype is torch.float:
            c_image = (rgbs[b, ...].clone().detach() * 255.0).to(dtype=torch.uint8)
        else:
            c_image = rgbs[b, ...].clone().to(dtype=torch.uint8)
        c_image = visualize_rgb_bboxes(c_image, bboxes[b, :].view(1, 4), show=False)
        # plot segmentation mask on images
        if masks is not None:
            c_image = visualize_rgb_segmentation(c_image, masks=masks[b, ...], alpha=alpha, show=False)
        if show:
            imgs_show(c_image)


def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):
    """
    Displays one or more models and keypoints.
    :param model_list: list of o3d Geometry objects to display
    :param keypoints_xyz: list of 3d coordinates of keypoints to visualize
    :param camera_locations: optional camera location to display
    :return: list of o3d.geometry.TriangleMesh mesh objects as keypoint markers
    """
    d = 0
    for model in model_list:
        max_bound = model.get_max_bound()
        min_bound = model.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

    keypoint_radius = 0.03 * d

    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    camera_locations.paint_uniform_color([0.1, 0.5, 0.1])
    o3d.visualization.draw_geometries(keypoint_markers + model_list + [camera_locations])

    return keypoint_markers


def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]

    for b in range(batch_size):
        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()

        visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

    return 0


def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.detach()[0, ...].to("cpu")
    pc2 = pc2.detach()[0, ...].to("cpu")
    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None


def scatter_bar_plot(plt, x, y, label, color="orangered"):
    """
    x   : torch.tensor of shape (n)
    y   : torch.tensor of shape (n, k)

    """
    n, k = y.shape
    width = 0.2 * torch.abs(x[1] - x[0])

    x_points = x.unsqueeze(-1).repeat(1, k)
    x_points += width * (torch.rand(size=x_points.shape) - 1)
    y_points = y

    plt.scatter(x_points, y_points, s=20.0, c=color, alpha=0.5, label=label)

    return plt


def update_pos_tensor_to_keypoint_markers(vis, keypoints, keypoint_markers):
    keypoints = keypoints[0, ...].to("cpu")
    keypoints = keypoints.numpy().transpose()

    for i in range(len(keypoint_markers)):
        keypoint_markers[i].translate(keypoints[i], relative=False)
        vis.update_geometry(keypoint_markers[i])
        vis.poll_events()
        vis.update_renderer()
    print("FINISHED UPDATING KEYPOINT MARKERS IN CORRECTOR")
    return keypoint_markers


def display_results(
    input_point_cloud, detected_keypoints, target_point_cloud, target_keypoints=None, render_text=False
):
    """
    inputs:
    input_point_cloud   :   torch.tensor of shape (B, 3, m)
    detected_keypoints  :   torch.tensor of shape (B, 3, N)
    target_point_cloud  :   torch.tensor of shape (B, 3, n)
    target_keypoints    :   torch.tensor of shape (B, 3, N)

    where
    B = batch size
    N = number of keypoints
    m = number of points in the input point cloud
    n = number of points in the target point cloud
    """

    if render_text:
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window("Mesh-Viewer", 1024, 750)
        scene = gui.SceneWidget()
        scene.scene = rendering.Open3DScene(window.renderer)
        window.add_child(scene)
        # displaying only the first item in the batch
    if input_point_cloud is not None:
        input_point_cloud = input_point_cloud[0, ...].to("cpu")
    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints[0, ...].to("cpu")
    if target_point_cloud is not None:
        target_point_cloud = target_point_cloud[0, ...].to("cpu")

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz in detected_keypoints:
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            kpt_mesh.translate(xyz)
            kpt_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(kpt_mesh)
        detected_keypoints = keypoint_markers

    if target_keypoints is not None:
        target_keypoints = target_keypoints[0, ...].to("cpu")
        target_keypoints = target_keypoints.numpy().transpose()
        keypoint_markers = []
        for xyz_idx in range(len(target_keypoints)):
            kpt_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            kpt_mesh.translate(target_keypoints[xyz_idx])
            kpt_mesh.paint_uniform_color([1, 0.0, 0.0])
            xyz_label = target_keypoints[xyz_idx] + np.array([0.0, 0.0, 0.0])
            if render_text:
                scene_label = scene.add_3d_label(xyz_label, str(xyz_idx))
                # scene_label.scale = 2.0
            keypoint_markers.append(kpt_mesh)
        target_keypoints = keypoint_markers

    if target_point_cloud is not None:
        target_point_cloud = pos_tensor_to_o3d(target_point_cloud)
        target_point_cloud.paint_uniform_color([0.0, 0.0, 0.7])

    if input_point_cloud is not None:
        input_point_cloud = pos_tensor_to_o3d(input_point_cloud)
        input_point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
    elements_to_viz = []
    if target_point_cloud is not None:
        elements_to_viz = elements_to_viz + [target_point_cloud]
        if render_text:
            bounds = target_point_cloud.get_axis_aligned_bounding_box()
            scene.setup_camera(60, bounds, bounds.get_center())

    if input_point_cloud is not None:
        elements_to_viz = elements_to_viz + [input_point_cloud]
    if detected_keypoints is not None:
        elements_to_viz = elements_to_viz + detected_keypoints
    if target_keypoints is not None:
        elements_to_viz = elements_to_viz + target_keypoints

    if render_text:
        for idx, element_to_viz in enumerate(elements_to_viz):
            scene.scene.add_geometry(str(idx), element_to_viz, rendering.MaterialRecord())
        gui.Application.instance.run()  # Run until user closes window
    else:
        # draw_geometries_with_rotation(elements_to_viz)
        o3d.visualization.draw_geometries(elements_to_viz)

    return None


def temp_expt_1_viz(cad_models, model_keypoints, gt_keypoints=None, colors=None):
    batch_size = model_keypoints.shape[0]
    if gt_keypoints is None:
        gt_keypoints = model_keypoints
    # print("model_keypoints.shape", model_keypoints.shape)
    # print("gt_keypoints.shape", gt_keypoints.shape)
    # print("cad_models.shape", cad_models.shape)

    for b in range(batch_size):
        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()
        gt_keypoints = gt_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        if colors is not None:
            point_cloud.colors = colors
        else:
            point_cloud = point_cloud.paint_uniform_color([1.0, 1.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()
        gt_keypoints = gt_keypoints.transpose(0, 1).numpy()

        # visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

        d = 0
        max_bound = point_cloud.get_max_bound()
        min_bound = point_cloud.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

        keypoint_radius = 0.01 * d

        keypoint_markers = []
        for xyz in keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0, 0.8, 0.0])
            keypoint_markers.append(new_mesh)
        for xyz in gt_keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0, 0.0])
            keypoint_markers.append(new_mesh)

        custom_draw_geometry_with_key_callback(keypoint_markers + [point_cloud])
        # o3d.visualization.draw_geometries(keypoint_markers + [point_cloud])

        return keypoint_markers

    return 0


def viz_rgb_pcd(
    target_object,
    viewpoint_camera,
    referenceCamera,
    viewpoint_angle,
    viz=False,
    dataset_path="../../data/ycb/models/ycb/",
):
    pcd = o3d.io.read_point_cloud(
        dataset_path
        + target_object
        + "/clouds/rgb/pc_"
        + viewpoint_camera
        + "_"
        + referenceCamera
        + "_"
        + viewpoint_angle
        + "_masked_rgb.ply"
    )
    xyzrgb = np.load(
        dataset_path
        + target_object
        + "/clouds/rgb/pc_"
        + viewpoint_camera
        + "_"
        + referenceCamera
        + "_"
        + viewpoint_angle
        + "_masked_rgb.npy"
    )
    print(xyzrgb.shape)
    rgb = xyzrgb[0, :, 3:]
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float) / 255.0)
    print(np.asarray(pcd.points).shape)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    return pcd


def draw_geometries_with_rotation(elements, toggle=True):
    def rotate_view(vis, toggle=toggle):
        ctr = vis.get_view_control()
        if toggle:
            ctr.rotate(0.05, 0)
        else:
            ctr.rotate(-0.05, 0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback(elements, rotate_view)


def custom_draw_geometry_with_key_callback(elements):
    def rotate_view_cw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(5, 0)
        return False

    def rotate_view_ccw(vis):
        ctr = vis.get_view_control()
        ctr.rotate(-5, 0)
        return False

    key_to_callback = {}
    key_to_callback[ord("A")] = rotate_view_cw
    key_to_callback[ord("D")] = rotate_view_ccw

    o3d.visualization.draw_geometries_with_key_callbacks(elements, key_to_callback)


def plt_save_figures(basename, save_folder="./", formats=None, dpi="figure"):
    """Helper function to save figures"""
    if formats is None:
        formats = ["pdf", "png"]

    for format in formats:
        fname = f"{basename}.{format}"
        plt.savefig(os.path.join(save_folder, fname), bbox_inches="tight", dpi=dpi)


def plot_obj_pose_graph(_graph, object_index, shape_models, color="black", plotter=None):
    from scipy.spatial.transform import Rotation

    _nodes = []
    for idx, _node in enumerate(_graph[0]):
        _nodes.append(_graph[0][idx][1].reshape(1, -1))
    _nodes = np.concatenate(_nodes, axis=0)

    pv.set_plot_theme("document")
    if plotter is None:
        pl = pv.Plotter()
    else:
        pl = plotter

    # adding points
    pl.add_points(_nodes[:, :3], render_points_as_spheres=True, point_size=4.0, color=color)

    # # adding lines
    # lines = []
    # for _e in self._edges:
    #     lines.append(self._nodes[_e[0], :3])
    #     lines.append(self._nodes[_e[1], :3])
    #
    # lines = np.asarray(lines)
    # pl.add_lines(lines, color='brown', width=1.0)

    for label, idx in object_index.items():
        # get object pose
        g2o_vec = _nodes[idx, :]
        t = g2o_vec[:3].reshape(3, 1)
        q = g2o_vec[3:]
        _R = Rotation.from_quat(q)
        R = _R.as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        # get object cad model
        reader_ = pv.get_reader(shape_models[label])
        mesh = reader_.read()
        mesh = mesh.scale([0.001, 0.001, 0.001])  # cad model scale is mm, pose scale is in m
        mesh = mesh.transform(T)

        # add posed cad model to pl
        pl.add_mesh(mesh, color=color)
        del mesh

    # pl.add_floor()
    return pl


def plot_obj_pose_graphs(graphs, object_indices, shape_models, colors=None):
    pl = pv.Plotter()
    if colors is None:
        colors = ["black"] * len(graphs)
    assert len(graphs) == len(object_indices) == len(shape_models)
    for g, o, s, c in zip(graphs, object_indices, shape_models, colors):
        plot_obj_pose_graph(g, o, s, color=c, plotter=pl)
    return pl


def plot_histogram_torch(data, bins=50, min=None, max=None, density=False, xlabel="Bin Centers"):
    if min is None:
        min = data.min().detach().cpu().item()
    if max is None:
        max = data.max().detach().cpu().item()

    if torch.is_tensor(data):
        data_to_plot = data.detach().cpu().numpy()
    else:
        data_to_plot = np.array(data)

    hist, bins = np.histogram(data_to_plot, density=density, bins=bins)
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    plt.bar(center, hist, align="center", width=width)
    plt.xlabel(xlabel)
    plt.show()


def gen_pyvista_voxel_slices(data_grid: np.ndarray, origin, voxel_size, slice_type="orthogonal"):
    """Visualize slices across a 3D voxel grid"""
    assert len(voxel_size) == 3

    pv_grid = pv.ImageData()
    pv_grid.dimensions = np.array(data_grid.shape) + 1
    pv_grid.origin = tuple(origin)
    pv_grid.spacing = voxel_size
    pv_grid.cell_data["values"] = data_grid.flatten(order="F")
    if slice_type == "orthogonal":
        slices = pv_grid.slice_orthogonal()
    elif slice_type == "z":
        slices = pv_grid.slice(normal=[0, 0, 1])
    elif slice_type == "x":
        slices = pv_grid.slice(normal=[1, 0, 0])
    elif slice_type == "y":
        slices = pv_grid.slice(normal=[0, 1, 0])
    return slices


def visualize_meshes_pyvista(
    meshes,
    plotter_shape=(1, 1),
    mesh_args=None,
    subplots=None,
    show_axes=True,
    show_bounds=True,
    off_screen=False,
    subplot_titles=None,
):
    """Visualize PyVista meshes/datasets

    Parameters
    ----------
    meshes
    plotter_shape
    mesh_args: additional arguments for visualization of each mesh
    subplots
    show_axes
    show_bounds
    """

    pl = pv.Plotter(shape=plotter_shape, off_screen=off_screen)
    if mesh_args is None:
        mesh_args = [{} for _ in range(len(meshes))]
    else:
        if len(mesh_args) != len(meshes):
            raise ValueError(
                f"Number of meshes and mesh_args must match. Meshes: {len(meshes)}, mesh_args: {len(mesh_args)}."
            )

    if subplots is None:
        subplots = [None for _ in range(len(meshes))]
    else:
        if len(subplots) != len(meshes):
            raise ValueError(
                f"Number of meshes and subplots must match. Meshes: {len(meshes)}, subplots: {len(subplots)}."
            )
    if subplot_titles is None:
        subplot_titles = [None for _ in range(len(meshes))]
    else:
        if len(subplot_titles) != len(meshes):
            raise ValueError(
                f"Number of meshes and subplot titles must match. Meshes: {len(meshes)}, subplot titles: {len(subplot_titles)}."
            )

    for m, args, subplot, subplot_title in zip(meshes, mesh_args, subplots, subplot_titles):
        if args is None:
            args = {}
        if type(args) is not dict:
            raise ValueError("mesh_args must be a list of dictionaries.")
        if subplot is not None:
            pl.subplot(*subplot)

        # preprocess tensors
        if torch.is_tensor(m):
            if m.shape[0] == 3:
                m = pv.PolyData(torch.as_tensor(m.T).cpu().numpy())
            elif m.shape[1] == 3:
                m = pv.PolyData(torch.as_tensor(m).cpu().numpy())

        # preprocess numpy arrays
        if isinstance(m, np.ndarray):
            if m.shape[0] == 3:
                m = pv.PolyData(m.T)
            elif m.shape[1] == 3:
                m = pv.PolyData(m)

        pl.add_mesh(m, **args)
        if subplot_title is not None:
            pl.add_text(subplot_title, font_size=12)

    if show_axes:
        pl.show_axes()
    if show_bounds:
        pl.show_bounds()
    if not off_screen:
        pl.show()
    return pl


def visualize_sdf_slices_pyvista(
    shp_code,
    shape_model,
    cube_scale=2.5,
    off_screen=False,
    additional_meshes=None,
    additional_meshes_args=None,
    slice_type="orthogonal",
):
    """Helper function to visualize SDF slices using PyVista"""

    def model_fn(coords):
        return shape_model.forward(shape_code=shp_code, coords=coords)

    with torch.no_grad():
        sdf_grid, voxel_size, voxel_grid_origin = create_sdf_samples_generic(
            model_fn=model_fn,
            N=128,
            max_batch=64**3,
            cube_center=np.array([0, 0, 0]),
            cube_scale=cube_scale,
        )

        pv_grid = pv.ImageData()
        pv_grid.dimensions = np.array(sdf_grid.shape) + 1
        pv_grid.origin = tuple(voxel_grid_origin)
        pv_grid.spacing = (voxel_size, voxel_size, voxel_size)
        pv_grid.cell_data["values"] = sdf_grid.detach().cpu().numpy().flatten(order="F")
        if slice_type == "orthogonal":
            slices = pv_grid.slice_orthogonal()
        elif slice_type == "z":
            slices = pv_grid.slice(normal=[0, 0, 1])

        # marching cubes to mesh
        pred_mesh = utils.sdf.convert_sdf_samples_to_mesh(
            sdf_grid=sdf_grid,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=None,
            scale=None,
        )

        pred_mesh = pred_mesh.simplify_quadric_decimation(5000)

    # SDF by querying on the mesh obtained from marching cubes
    def mesh_sdf_model_fn(coords):
        a = utils.geometry.query_sdf_from_mesh(coords.squeeze(0).numpy(force=True), pred_mesh)
        return torch.tensor(a)

    mesh_sdf_grid, mesh_voxel_size, mesh_voxel_grid_origin = create_sdf_samples_generic(
        model_fn=mesh_sdf_model_fn,
        N=128,
        max_batch=64**3,
        cube_center=np.array([0, 0, 0]),
        cube_scale=cube_scale,
    )

    mesh_sdf_slices = gen_pyvista_voxel_slices(
        mesh_sdf_grid.numpy(force=True), mesh_voxel_grid_origin, (mesh_voxel_size,) * 3, slice_type=slice_type
    )

    print(
        "Visualizing slices through the SDF network and the SDF calculated from mesh (marching cubes on the SDF grid)."
    )

    mesh_to_vis = [pred_mesh, slices, pred_mesh, mesh_sdf_slices]
    mesh_args = [
        {"opacity": 0.8, "color": "white"},
        {"opacity": 0.8},
        {"opacity": 0.8, "color": "white"},
        {"opacity": 0.8},
    ]
    subplots_locs = [(0, 0), (0, 0), (0, 1), (0, 1)]
    subplots_titles = ["SDF", None, "SDF from Marching Cube Mesh", None]

    additional_colors = ["blue", "red", "yellow", "green"]
    if additional_meshes is not None:
        mesh_to_vis.extend(additional_meshes)
        if additional_meshes_args is None:
            mesh_args.extend(
                [
                    {"opacity": 0.5, "color": additional_colors[i % len(additional_colors)]}
                    for i, _ in enumerate(additional_meshes)
                ]
            )
        else:
            mesh_args.extend(additional_meshes_args)
        subplots_locs.extend([(0, 0) for _ in additional_meshes])
        subplots_titles.extend([None for _ in additional_meshes])

    pl = visualize_meshes_pyvista(
        mesh_to_vis,
        mesh_args=mesh_args,
        plotter_shape=(1, 2),
        subplots=subplots_locs,
        subplot_titles=subplots_titles,
        off_screen=off_screen,
    )
    return pl


def project_mesh_to_image(mesh, rgb, cam_s_nocs, cam_R_nocs, cam_t_nocs, camera_intrinsics, camera_pose):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Load the mesh
    mesh.apply_scale(cam_s_nocs)
    objpose = np.eye(4)
    objpose[:3, :3] = cam_R_nocs
    objpose[:3, -1] = cam_t_nocs
    mesh.apply_transform(objpose)

    # pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    # scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[255, 255, 255])
    # camera = pyrender.IntrinsicsCamera(camera_intrinsics[0, 0], camera_intrinsics[1, 1],
    #                                   camera_intrinsics[0, 2], camera_intrinsics[1, 2])
    ##light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)
    # scene.add(pyrender_mesh)
    ##scene.add(light, pose=np.eye(4))
    # scene.add(camera, pose=np.eye(4))
    ## render scene
    # r = pyrender.OffscreenRenderer(rgb.shape[1], rgb.shape[0])
    # color, _ = r.render(scene)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(color)
    # plt.show()

    # Load the image
    image_ = rgb.copy()
    image = np.zeros((rgb.shape[0], rgb.shape[1], 3)).astype(np.uint8)
    image[: image_.shape[0], : image_.shape[1], :] = image_

    # Extract vertices and faces from the mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # Apply camera pose transformation
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    vertices_transformed = (camera_pose @ vertices_homogeneous.T).T

    # Project vertices to 2D
    vertices_2d = camera_intrinsics @ vertices_transformed[:, :3].T
    vertices_2d /= vertices_2d[2, :]
    vertices_2d = vertices_2d[:2, :].T

    # Draw the projected mesh
    vertice_z = vertices_transformed[:, 2].flatten()
    max_z = np.max(vertice_z)
    normalized_z = vertice_z / max_z
    overlay = image.copy()
    for face in faces:
        pts = np.array([vertices_2d[face[0]], vertices_2d[face[1]], vertices_2d[face[2]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        face_z = normalized_z[face[0]] + normalized_z[face[1]] + normalized_z[face[2]] / 3 * 255
        cv2.polylines(overlay, [pts], True, (0, face_z, 0))
    alpha = 0.7
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    image_pil = Image.fromarray(image)
    return image_pil




def save_joint_visualizations(meshes, payload, rgbs, masks, gt_objs_data, frames_info, intrinsics, export_folder):
    """Save NOCS heatmap
    Visualize:
    1. cropped RGB
    2. cropped RGB (masked)
    3. cropped RGB + NOCS
    4. cropped RGB w/ mesh transformed
    """
    bs = payload["cam_s_nocs"].shape[0]
    for i in tqdm(range(bs)):
        c_frame_idx = payload["frame_index"][i]
        c_obj_idx = payload["obj_index"][i]
        c_obj_info = gt_objs_data[c_frame_idx][c_obj_idx]
        label = c_obj_info["label"]
        frame_info = frames_info[c_frame_idx]

        # original rgb
        rgb_img_cpu = rgbs[c_frame_idx, ...].cpu()
        rgb_pil = F.to_pil_image(rgb_img_cpu)

        # processed rgb
        crop_rgb_img_cpu = payload["processed_imgs"][i, ...].cpu()
        crop_rgb_img_cpu[0, ...] = crop_rgb_img_cpu[0, ...] * 0.229 + 0.485
        crop_rgb_img_cpu[1, ...] = crop_rgb_img_cpu[1, ...] * 0.224 + 0.456
        crop_rgb_img_cpu[2, ...] = crop_rgb_img_cpu[2, ...] * 0.225 + 0.406
        crop_rgb_pil = F.to_pil_image(crop_rgb_img_cpu)

        # makes nocs map
        crop_nocs_map = payload["nocs_map"][i, :3, ...].float().cpu()
        crop_nocs_map -= crop_nocs_map.min()
        crop_nocs_map /= crop_nocs_map.max()  # from 0 to 1
        crop_nocs_pil = F.to_pil_image(crop_nocs_map.cpu())

        # crop mask
        crop_mask = payload["processed_mask"][i, ...].cpu()
        mask_pil = F.to_pil_image(crop_mask.int() * 255).convert("1")
        invert_mask = F.to_pil_image(255 - crop_mask.int() * 255).convert("1")

        # RGB masked
        rgb_masked = Image.composite(invert_mask, crop_rgb_pil, invert_mask)
        rgb_masked_array = np.array(rgb_masked)
        red, green, blue = rgb_masked_array.T
        white_areas = (red == 255) & (blue == 255) & (green == 255)
        rgb_masked_array[white_areas.T] = (0, 0, 0)
        crop_rgb_masked = Image.fromarray(rgb_masked_array)

        # RGB + NOCS
        crop_rgb_w_nocs = Image.composite(crop_nocs_pil, crop_rgb_pil, mask_pil)

        # render the mesh
        mesh = meshes[i]
        rgb_w_mesh = project_mesh_to_image(
            mesh,
            np.array(rgb_pil),
            payload["cam_s_nocs"].detach().cpu().numpy()[i, ...].item(),
            payload["cam_R_nocs"].detach().cpu().numpy()[i, :3, :3],
            payload["cam_t_nocs"]
            .detach()
            .cpu()
            .numpy()[i, ...]
            .reshape(
                3,
            ),
            intrinsics.detach().cpu().numpy()[c_frame_idx, ...].squeeze(),
            np.eye(4),
        )
        # crop it
        x1, y1, x2, y2 = c_obj_info["bbox"]
        crop_rgb_w_mesh = rgb_w_mesh.crop((x1, y1, x2, y2))
        crop_rgb_w_mesh = crop_rgb_w_mesh.resize((crop_rgb_w_nocs.size[1], crop_rgb_w_nocs.size[0]))

        # save
        c_export_folder = os.path.join(export_folder, str(frame_info["scene_id"]), "crop_vis")
        safely_make_folders([c_export_folder])
        base_name = f"{frame_info['frame_id']}_{frame_info['view_id']}_{label}.jpg"
        crop_rgb_name = "crop_rgb_" + base_name
        crop_rgb_pil.save(os.path.join(c_export_folder, crop_rgb_name))

        crop_rgb_mask_name = "crop_rgb_mask_" + base_name
        crop_rgb_masked.save(os.path.join(c_export_folder, crop_rgb_mask_name))

        crop_rgb_w_nocs_name = "crop_rgb_w_nocs_" + base_name
        crop_rgb_w_nocs.save(os.path.join(c_export_folder, crop_rgb_w_nocs_name))

        crop_rgb_w_mesh_name = "crop_rgb_w_mesh_" + base_name
        crop_rgb_w_mesh.save(os.path.join(c_export_folder, crop_rgb_w_mesh_name))

    return


def get_meshes_from_shape_codes(model, interp_shp_codes, mesh_recons_scale=0.251, cube_res=64):
    meshes = []
    # generate meshes for the shape codes
    for j in tqdm(range(interp_shp_codes.shape[0])):
        shp_code = torch.tensor(interp_shp_codes[j, ...]).unsqueeze(0).float().cuda()

        def model_fn(coords):
            return model.recons_net.forward(shape_code=shp_code, coords=coords)

        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
            model_fn=model_fn,
            N=cube_res,
            max_batch=64**3,
            cube_center=np.array([0, 0, 0]),
            cube_scale=mesh_recons_scale,
        )

        pred_mesh = convert_sdf_samples_to_mesh(
            sdf_grid=sdf_grid,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=None,
            scale=None,
        )
        meshes.append(pred_mesh)
    return meshes


def generate_video_from_meshes(meshes, cam_traj_fn, length, path):
    """Make an orbiting video around the meshes"""
    pl = pv.Plotter(off_screen=True)
    pl.open_movie(path)

    for mesh in meshes:
        pl.add_mesh(pv.wrap(mesh), color="white", opacity=1)

    pl.show(auto_close=False)
    cam_traj_fn(0, pl)
    pl.write_frame()
    for i in tqdm(range(1, length)):
        cam_traj_fn(i, pl)
        pl.update()
        pl.write_frame()  # Write this frame
    pl.close()


def generate_orbiting_video_from_meshes(meshes, path, viewup=None, shift=0.5, factor=2.0, n_points=20, show_axes=False):
    """Make an orbiting video around the meshes"""
    pl = pv.Plotter(off_screen=True)
    pl.open_movie(path)

    for mesh in meshes:
        pl.add_mesh(pv.wrap(mesh), color="white", opacity=1)

    if show_axes:
        pl.show_axes()
    pl.show(auto_close=False)
    path = pl.generate_orbital_path(factor=factor, n_points=n_points, viewup=viewup, shift=shift)
    pl.orbit_on_path(path, write_frames=True)
    pl.close()


def save_screenshots_from_shape_codes(model, interp_shp_codes, export_names, mesh_export_names=None):
    meshes = []
    # generate meshes for the shape codes
    for j in tqdm(range(interp_shp_codes.shape[0])):
        shp_code = torch.tensor(interp_shp_codes[j, ...]).unsqueeze(0).float().cuda()

        def model_fn(coords):
            return model.recons_net.forward(shape_code=shp_code, coords=coords)

        (sdf_grid, voxel_size, voxel_grid_origin) = create_sdf_samples_generic(
            model_fn=model_fn,
            N=128,
            max_batch=64**3,
            cube_center=np.array([0, 0, 0]),
            cube_scale=0.251,
        )

        pred_mesh = convert_sdf_samples_to_mesh(
            sdf_grid=sdf_grid,
            voxel_grid_origin=voxel_grid_origin,
            voxel_size=voxel_size,
            offset=None,
            scale=None,
        )

        # mesh_name = f"{o1}_{o2}_step_{j}.png"
        # safely_make_folders([f"./exports/{prefix}_{model_name}/{o1}_{o2}"])
        # file_name = f"./exports/{prefix}_{model_name}/{o1}_{o2}/{mesh_name}"
        containing_folder = pathlib.Path(export_names[j]).parent
        safely_make_folders([containing_folder])
        pv_mesh = pv.wrap(pred_mesh)
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(pv_mesh)
        pl.show(screenshot=export_names[j])
        meshes.append(pred_mesh)
        if mesh_export_names is not None:
            pred_mesh.export(mesh_export_names[j])
    return meshes
