import blenderproc as bproc
import csv
import os
import argparse
import json
import bpy

from blenderproc.python.types.MeshObjectUtility import convert_to_meshes

SKIP_OBJECTS = [
    "frl_apartment_rug_02",
    "frl_apartment_rug_01",
    "frl_apartment_picture_01",
    "frl_apartment_clothes_hanger_02",  # incorrect orientation in some scenes
    "frl_apartment_mat",
    "frl_apartment_table_03",  # incorrect orientation in some scenes
    "frl_apartment_cloth_03",  # does load in pyvista: glb
    "frl_apartment_vase_01",  # does load in pyvista: glb
]


def append_csv(filename, row):
    """
    Appends a row to an existing CSV file or creates a new one if the
    file doesn't exist.

    Parameters:
    filename (str): The name of the CSV file
    row (list): The row to be written to the CSV file

    Returns:
    None
    """
    mode = "w"
    if os.path.isfile(filename):
        mode = "a"
    with open(filename, mode, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def replicacad_dict(scene_config_file: str, args):
    with open(scene_config_file) as f:
        scene_config = json.load(f)

    object_dict = dict()
    for obj_ in scene_config["object_instances"]:
        obj = dict()
        if args.scene_name[:3] == "apt":
            obj_config_file = config_dir + "/" + str(obj_["template_name"] + ".object_config.json")
        else:
            obj_config_file = config_dir + "/" + "objects" + "/" + str(obj_["template_name"] + ".object_config.json")

        with open(obj_config_file) as f:
            obj_config = json.load(f)

        obj["name"] = obj_["template_name"].split("/")[-1]
        if obj["name"] in SKIP_OBJECTS:
            continue

        obj["cad_file"] = args.replicacad_path + "/" + "objects" + "/" + str(obj_config["render_asset"].split("/")[-1])
        obj["t"] = obj_["translation"]
        obj["q"] = obj_["rotation"]

        object_dict[obj["name"]] = obj

    return object_dict


def replicacad_loader(object_name: str, object_dict: dict):
    selected_obj = object_dict[object_name]
    filepath = selected_obj["cad_file"]
    bpy.ops.import_scene.gltf(filepath=filepath)

    loaded_objects = convert_to_meshes([obj for obj in bpy.context.selected_objects])

    obj = loaded_objects[0]
    obj.set_cp("model_path", filepath)
    obj.set_cp("object_name", object_name)

    return obj


if __name__ == "__main__":
    """Script to generate dataset that contains rendered ShapeNetCore objects in different poses.
    For mapping between categories to IDs, see: https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e
    This requires the installation of BlenderProc: https://github.com/DLR-RM/BlenderProc

    To visualize, run:
    ```
    blenderproc vis hdf5 output/*.hdf5
    ```
    where output is where you set the output_dir to be.
    """
    print("DEPRECATED. Please use the gen_unified.py script.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicacad_path", help="Path to the replicacad", default="data/replica_cad")
    parser.add_argument("--output_dir", default="./output", help="Path to where the final files, will be saved")
    parser.add_argument("--scene_name", help="replica cad scene", default="apt_0")
    parser.add_argument("--object_name", help="replica cad object", default="frl_apartment_chair_01")
    parser.add_argument("--append_to_existing", default=False, type=bool)
    parser.add_argument("--resolution", nargs="*", default=[256, 256], type=int)
    parser.add_argument("--num_frames", default=200, type=int)
    parser.add_argument("--render_normals", default=True)
    parser.add_argument("--render_depth", default=True)
    parser.add_argument("--render_distance", default=True)
    parser.add_argument("--render_segmap", default=True)
    parser.add_argument("--render_nocs", default=True)
    args = parser.parse_args()

    config_dir = args.replicacad_path + "/" + "configs"
    scene_config_file = config_dir + "/" + "scenes" + "/" + str(args.scene_name + ".scene_instance.json")

    # check whether the shapenet dir exists
    assert os.path.exists(args.replicacad_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    bproc.init()

    obj_dict = replicacad_dict(scene_config_file, args)
    obj = replicacad_loader(args.object_name, obj_dict)

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    for i in range(args.num_frames):
        location = bproc.sampler.sphere([0, 0, 0], radius=2, mode="SURFACE")
        rotation_matrix = bproc.camera.rotation_from_forward_vec(obj.get_location() - location)
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
        bproc.camera.set_resolution(args.resolution[0], args.resolution[1])

    # enable different options for output
    if args.render_normals:
        bproc.renderer.enable_normals_output()

    if args.render_depth or args.render_distance:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

    data = bproc.renderer.render()

    # prepare metadata
    # shapenet_state = {
    #     "used_synset_id": shapenet_obj.get_cp("used_synset_id")
    # }
    # data["shapenet_state"] = [shapenet_state] * bproc.utility.num_frames()

    # prepare camera poses/states data
    cam_states = []
    for frame in range(bproc.utility.num_frames()):
        cam_states.append(
            {"cam2world": bproc.camera.get_camera_pose(frame), "cam_K": bproc.camera.get_intrinsics_as_K_matrix()}
        )
    data["cam_states"] = cam_states

    # segmentation masks
    if args.render_segmap:
        segmap_data = bproc.renderer.render_segmap(map_by=["instance"])
        data.update(segmap_data)

    # postprocess to distance
    if args.render_distance:
        data["distance"] = bproc.postprocessing.depth2dist(data["depth"])

    # render nocs
    if args.render_nocs:
        data.update(bproc.renderer.render_nocs())

    bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=args.append_to_existing)
