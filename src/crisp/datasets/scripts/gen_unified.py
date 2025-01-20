import blenderproc as bproc
from typing import Tuple
from matplotlib import pyplot as plt
import csv
import yaml
import os
import numpy as np
import sys
import json
from jsonargparse import ArgumentParser
from dataclasses import dataclass
import bpy

from blenderproc.python.types.MeshObjectUtility import convert_to_meshes

# Notes for ShapeNetCore objects:
# note: Blender applies a rotation of
# array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
#        [ 0.00000000e+00, -4.37113883e-08, -1.00000000e+00],
#        [ 0.00000000e+00,  1.00000000e+00, -4.37113883e-08],
# to the loaded object. This will be reflected in the bpy object rotation attribute.
# You can use obj.get_rotation_mat() where obj is the loaded blenderproc object to see the current
# rotation matrix applied to the object.
#
# The shapenet loader will apply this rotation to the object coordinates
# world_p = world_R_cad * cad_p
# cad_p = cad_R_world * world_p

# Notes for YCBV objects:
# Objects have z-axis up

# fmt: off
blender_R_cad = {
    "shapenet": np.array([[1.0, 0, 0],
                          [0, 0, -1.0],
                          [0, 1.0, 0]]),
    "ycbv": np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]]),
    "uhumans_office": np.array([[1.0, 0, 0],
                                [0, 0, -1.0],
                                [0, 1.0, 0]]),
    "replicacad": np.array([[1.0, 0, 0],
                            [0, 0, -1.0],
                            [0, 1.0, 0]]),
    "spe3r": np.array([[1.0, 0, 0],
                       [0, 0, -1.0],
                       [0, 1.0, 0]]),
}
# fmt: on
cad_R_blender = {k: v.T for k, v in blender_R_cad.items()}


@dataclass
class RenderSettings:
    bop_path: str = None
    replicacad_path: str = None
    spe3r_path: str = None
    shapenet_path: str = None
    uhumans_path: str = None
    objects_config_to_load: str = None
    output_dir: str = "./output"
    resolution: Tuple[int, int] = (256, 256)
    num_frames: int = 1
    render_normals: bool = True
    render_depth: bool = True
    render_distance: bool = True
    render_segmap: bool = True
    render_nocs: bool = True
    vis_nocs: bool = False
    use_opencv_image_frame: bool = True

    # render settings
    camera_sample_min_radius: float = 2
    camera_sample_max_radius: float = 4
    camera_sample_elevation_min: float = 5
    camera_sample_elevation_max: float = 89


def clean_up():
    """Helper function to clean up the blender scene.
    See this issue for why use_nodes=True is set. https://github.com/DLR-RM/BlenderProc/pull/859
    """
    bproc.clean_up(clean_up_camera=True)
    bpy.context.scene.world.use_nodes = True
    return


def compute_scale(tmesh):
    """Compute scale necessary to normalize the coordinates into a unit cube centered at origin"""
    coord_max = np.amax(tmesh.vertices, axis=0, keepdims=True)
    coord_min = np.amin(tmesh.vertices, axis=0, keepdims=True)
    max_dist = np.abs(coord_max - coord_min)
    scale = np.amax(max_dist)
    return scale


def compute_bbox_centroid(tmesh):
    """Compute the centroid of the axis-aligned bounding box"""
    coord_max = np.amax(tmesh.vertices, axis=0, keepdims=True)
    coord_min = np.amin(tmesh.vertices, axis=0, keepdims=True)
    box_size = coord_max - coord_min
    assert box_size[0] > 0 and box_size[1] > 0 and box_size[2] > 0
    centroid = coord_min + box_size / 2
    return centroid


def load_shapenet_object(object_name: str, path_to_shapenet: str):
    synset_id, source_id = object_name.split("_")
    obj = bproc.loader.load_shapenet(
        path_to_shapenet, used_synset_id=synset_id, used_source_id=source_id, move_object_origin=False
    )
    return obj


def load_uhumans_object(object_name: str, path_to_uhumans: str):
    split = object_name.split("_")[0]
    actual_name = "_".join(object_name.split("_")[1:])
    model_path = os.path.join(path_to_uhumans, split, actual_name + ".obj")
    obj = bproc.loader.load_obj(model_path)[0]
    return obj


def load_spe3r_object(object_name: str, path_to_spe3r: str):
    model_path = os.path.join(path_to_spe3r, object_name, "models", "model_normalized.obj")
    obj = bproc.loader.load_obj(model_path)[0]
    return obj


def load_replicacad_object(object_name: str, path_to_replicacad: str):
    config_path = os.path.join(path_to_replicacad, "configs", "objects", f"{object_name}.object_config.json")
    with open(config_path, "r") as config_f:
        config_data = json.load(config_f)
    asset_path = os.path.abspath(
        os.path.join(path_to_replicacad, "objects", os.path.basename(config_data["render_asset"]))
    )
    bpy.ops.import_scene.gltf(filepath=asset_path)
    loaded_objects = convert_to_meshes([obj for obj in bpy.context.selected_objects])
    obj = loaded_objects[0]
    obj.set_cp("model_path", asset_path)
    obj.set_cp("object_name", object_name)
    return obj


def get_objs_to_load_per_dataset(opt, dataset_name):
    # determine which objects to load
    objs_config_path = opt.objects_config_to_load
    if objs_config_path is None:
        # load all objects
        print("No object config file provided. Loading YCBV objects.")
        objs_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs", "ycbv_only.yaml")

    with open(objs_config_path, "r") as f:
        all_objs = yaml.safe_load(f)
        if dataset_name in all_objs.keys():
            objs = all_objs[dataset_name]
            return objs
        else:
            return None


def render_uhumans_objs(objs, opt: RenderSettings):
    objs_metadata = {}
    for obj_name in objs:
        print(f"Rendering frames for uHumans - {obj_name}")
        obj = load_uhumans_object(obj_name, path_to_uhumans=opt.uhumans_path)
        blender_scale = obj.get_scale()
        uhumans_subset = obj_name.split("_")[0]
        obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)
        tmesh_after_persist = obj.mesh_as_trimesh()
        after_persist_obj_scale = compute_scale(tmesh_after_persist)

        metadata = {
            "obj_name": obj_name,
            "dataset_name": "uhumans",
            "obj_centroid": np.asarray(tmesh_after_persist.centroid).tolist(),
            "obj_scale": after_persist_obj_scale.item(),
        }

        render_single_obj(obj=obj, cad_R_blender=cad_R_blender[f"uhumans_{uhumans_subset}"], metadata=metadata, opt=opt)
        clean_up()

        # note: these are in the scale of the original CAD model without applying the scaling modifier
        objs_metadata[obj_name] = {
            # transformation from CAD to blender frame
            "blender_R_cad": np.asarray(blender_R_cad[f"uhumans_{uhumans_subset}"]).tolist(),
            "blender_s_cad": np.asarray(blender_scale).tolist(),
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_t_blender": np.array(-2 / after_persist_obj_scale * tmesh_after_persist.centroid).tolist(),
            "recons_s_blender": 2 / after_persist_obj_scale.item(),
        }

    return objs_metadata


def render_shapenet_objs(objs, opt: RenderSettings):
    objs_metadata = {}
    for obj_name in objs:
        print(f"Rendering frames for ShapeNet - {obj_name}")
        obj = load_shapenet_object(obj_name, path_to_shapenet=opt.shapenet_path)
        blender_scale = obj.get_scale()
        obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)
        tmesh_after_persist = obj.mesh_as_trimesh()
        after_persist_obj_scale = compute_scale(tmesh_after_persist)

        metadata = {
            "obj_name": obj_name,
            "dataset_name": "shapenet",
            "obj_centroid": np.asarray(tmesh_after_persist.centroid).tolist(),
            "obj_scale": after_persist_obj_scale.item(),
        }

        render_single_obj(obj=obj, cad_R_blender=cad_R_blender["shapenet"], metadata=metadata, opt=opt)
        clean_up()

        # note: these are in the scale of the original CAD model without applying the scaling modifier
        objs_metadata[obj_name] = {
            # transformation from CAD to blender frame
            "blender_R_cad": np.asarray(blender_R_cad["shapenet"]).tolist(),
            "blender_s_cad": np.asarray(blender_scale).tolist(),
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_t_blender": np.array(-2 / after_persist_obj_scale * tmesh_after_persist.centroid).tolist(),
            "recons_s_blender": 2 / after_persist_obj_scale.item(),
        }
    return objs_metadata


def render_spe3r_objs(objs, opt: RenderSettings):
    objs_metadata = {}
    for obj_name in objs:
        print(f"Rendering frames for SPE3R - {obj_name}")
        obj = load_spe3r_object(obj_name, path_to_spe3r=opt.spe3r_path)
        blender_scale = obj.get_scale()
        obj.persist_transformation_into_mesh(location=False, rotation=True, scale=False)
        tmesh_after_persist = obj.mesh_as_trimesh()
        after_persist_obj_scale = compute_scale(tmesh_after_persist)

        metadata = {
            "obj_name": obj_name,
            "dataset_name": "spe3r",
            "obj_centroid": np.asarray(tmesh_after_persist.centroid).tolist(),
            "obj_scale": after_persist_obj_scale.item(),
        }

        render_single_obj(obj=obj, cad_R_blender=cad_R_blender["spe3r"], metadata=metadata, opt=opt)
        clean_up()

        # note: these are in the scale of the original CAD model without applying the scaling modifier
        objs_metadata[obj_name] = {
            # transformation from CAD to blender frame
            "blender_R_cad": np.asarray(blender_R_cad["spe3r"]).tolist(),
            "blender_s_cad": np.asarray(blender_scale).tolist(),
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_t_blender": np.array(-2 / after_persist_obj_scale * tmesh_after_persist.centroid).tolist(),
            "recons_s_blender": 2 / after_persist_obj_scale.item(),
        }
    return objs_metadata


def render_replicacad_objs(objs, opt: RenderSettings):
    objs_metadata = {}
    for obj_name in objs:
        print(f"Rendering frames for replicacad - {obj_name}")
        obj = load_replicacad_object(obj_name, path_to_replicacad=opt.replicacad_path)
        blender_scale = obj.get_scale()
        obj.persist_transformation_into_mesh(location=False, rotation=True, scale=True)
        tmesh_after_persist = obj.mesh_as_trimesh()
        after_persist_obj_scale = compute_scale(tmesh_after_persist)

        metadata = {
            "obj_name": obj_name,
            "dataset_name": "replicacad",
            "obj_centroid": np.asarray(tmesh_after_persist.centroid).tolist(),
            "obj_scale": after_persist_obj_scale.item(),
        }
        render_single_obj(obj=obj, cad_R_blender=cad_R_blender["replicacad"], metadata=metadata, opt=opt)
        clean_up()

        # note: these are in the scale of the original CAD model without applying the scaling modifier
        objs_metadata[obj_name] = {
            # transformation from CAD to blender frame
            "blender_R_cad": np.asarray(blender_R_cad["replicacad"]).tolist(),
            "blender_s_cad": np.asarray(blender_scale).tolist(),
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_t_blender": np.array(-2 / after_persist_obj_scale * tmesh_after_persist.centroid).tolist(),
            "recons_s_blender": 2 / after_persist_obj_scale.item(),
        }

    return objs_metadata


def render_ycbv_objs(objs, opt: RenderSettings):
    # to ids
    obj_ids = [int(x[4:]) for x in objs]

    objs_metadata = {}
    for obj_id, obj_id_str in zip(obj_ids, objs):
        # load specified bop objects into the scene
        # internally, units are converted to m by just apply a scale transformation on the
        # loaded blender object, which are not reflected in the actual mesh coordinates
        print(f"Rendering frames for ycbv - {obj_id_str}")
        ycbv_obj = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(opt.bop_path, "ycbv"), mm2m=True, obj_ids=[obj_id]
        )[0]
        # this will apply the scaling operations to the coordinates of the object
        ycbv_obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)
        tmesh = ycbv_obj.mesh_as_trimesh()
        og_scale = compute_scale(tmesh)
        metadata = {
            "obj_name": obj_id_str,
            "dataset_name": "ycbv",
            "obj_centroid": np.asarray(tmesh.centroid).tolist(),
            "obj_scale": og_scale.item(),
        }
        render_single_obj(obj=ycbv_obj, cad_R_blender=cad_R_blender["ycbv"], metadata=metadata, opt=opt)
        clean_up()
        # 1000 is needed due to mm2m scaling
        # objs metadata stores the info needed to transform from the CAD frame to the blender world frame
        objs_metadata[obj_id_str] = {
            # transformation from CAD to blender frame
            "blender_R_cad": np.asarray(blender_R_cad["ycbv"]).tolist(),
            "blender_s_cad": [0.001, 0.001, 0.001],  # mm2m
            # transformation from blender to centered and normalized ([-1, 1]) frame (recons frame)
            "recons_t_blender": np.asarray(2 / og_scale * -tmesh.centroid).tolist(),
            "recons_s_blender": 2 / og_scale.item(),
        }
    return objs_metadata


def render_single_obj(obj, cad_R_blender, metadata, opt: RenderSettings):
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    # center and normalize to [-1, 1]
    # make it persistent into the blender coordinates
    obj.set_location(obj.get_location() - np.array(metadata["obj_centroid"]))
    obj.persist_transformation_into_mesh(location=True, rotation=False, scale=False)

    obj.set_scale([2 / metadata["obj_scale"], 2 / metadata["obj_scale"], 2 / metadata["obj_scale"]])
    obj.persist_transformation_into_mesh(location=False, rotation=False, scale=True)

    for i in range(opt.num_frames):
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=opt.camera_sample_min_radius,
            radius_max=opt.camera_sample_max_radius,
            elevation_min=opt.camera_sample_elevation_min,
            elevation_max=opt.camera_sample_elevation_max,
            uniform_volume=False,
        )
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            obj.get_location() - location, inplane_rot=np.random.uniform(-3.14159, 3.14159)
        )
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        bproc.camera.set_resolution(opt.resolution[0], opt.resolution[1])

    data = bproc.renderer.render()
    data["metadata"] = [metadata] * opt.num_frames

    # prepare camera poses/states data
    cam_states = []
    for frame in range(bproc.utility.num_frames()):
        cam2world = bproc.camera.get_camera_pose(frame)
        if opt.use_opencv_image_frame:
            # OpenGL: +z towards viewer, +y up
            # OpenCV: +z away from viewer, +y down
            # The function below is equivalent to post multiply by [1, 0, 0; 0, -1, 0; 0, 0, -1]
            cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
        cam_states.append({"cam2world": cam2world, "cam_K": bproc.camera.get_intrinsics_as_K_matrix()})
    data["cam_states"] = cam_states

    # segmentation masks
    if opt.render_segmap:
        segmap_data = bproc.renderer.render_segmap(
            map_by=["instance"],
            file_prefix=f"segmap_{metadata['obj_name']}",
            output_key=f"segmap_{metadata['obj_name']}",
            segcolormap_output_file_prefix=f"instance_attribute_map_{metadata['obj_name']}",
            segcolormap_output_key=f"segcolormap_{metadata['obj_name']}",
        )
        data.update(segmap_data)

    # postprocess to distance
    if opt.render_distance:
        data["distance"] = bproc.postprocessing.depth2dist(data["depth"])

    # render nocs
    if opt.render_nocs:
        nocs_data = bproc.renderer.render_nocs()
        data.update(nocs_data)

    # visualize the NOCS
    if opt.vis_nocs:
        nocs_pts = np.transpose(data["nocs"][0], (2, 0, 1))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(nocs_pts[0, ...], nocs_pts[1, ...], nocs_pts[2, ...], marker="o")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        fig.savefig("nocs_vis.pdf")

    # data.update(metadata)
    bproc.writer.write_hdf5(opt.output_dir, data, append_to_existing_output=True)


if __name__ == "__main__":
    """Main script for generating training dataset.
    Notes on frames:
    - Blender frame: the frame of the imported object models in Blender. Potentially rotated and scaled.
    - Recons frame: centered and scaled object from the Blender frame so the object lies in [-1, 1]
    - NOCS: recons frame / 2 + 0.5
    """
    parser = ArgumentParser()
    parser.add_class_arguments(RenderSettings)
    opt = parser.parse_args()
    opt = RenderSettings(**opt)

    # check whether the input/output dir exists
    assert os.path.exists(opt.replicacad_path)
    assert os.path.exists(opt.bop_path)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    bproc.init()

    # enable different options for output
    if opt.render_normals:
        bproc.renderer.enable_normals_output()

    if opt.render_depth or opt.render_distance:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

    objects_yaml_data = {}
    # spe3r
    spe3r_objs = get_objs_to_load_per_dataset(opt, "spe3r")
    if spe3r_objs is not None:
        spe3r_objs_metadata = render_spe3r_objs(objs=spe3r_objs, opt=opt)
        objects_yaml_data["spe3r"] = spe3r_objs_metadata

    # replicacad
    replicacad_objs = get_objs_to_load_per_dataset(opt, "replicacad")
    if replicacad_objs is not None:
        replicacad_objs_metadata = render_replicacad_objs(objs=replicacad_objs, opt=opt)
        objects_yaml_data["replicacad"] = replicacad_objs_metadata

    # ycbv
    ycbv_objs = get_objs_to_load_per_dataset(opt, "ycbv")
    if ycbv_objs is not None:
        ycbv_objs_metadata = render_ycbv_objs(objs=ycbv_objs, opt=opt)
        objects_yaml_data["ycbv"] = ycbv_objs_metadata

    # shapenet
    shapenet_objs = get_objs_to_load_per_dataset(opt, "shapenet")
    if shapenet_objs is not None:
        shapenet_objs_metadata = render_shapenet_objs(objs=shapenet_objs, opt=opt)
        objects_yaml_data["shapenet"] = shapenet_objs_metadata

    # uhumans
    uhumans_objs = get_objs_to_load_per_dataset(opt, "uhumans")
    if uhumans_objs is not None:
        uhumans_objs_metadata = render_uhumans_objs(objs=uhumans_objs, opt=opt)
        objects_yaml_data["uhumans"] = uhumans_objs_metadata

    # dump to yaml file
    with open(os.path.join(opt.output_dir, "objects_info.yaml"), "w") as objects_yaml:
        yaml.dump(objects_yaml_data, objects_yaml)
