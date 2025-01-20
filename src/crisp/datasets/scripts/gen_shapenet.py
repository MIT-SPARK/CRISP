import blenderproc as bproc
from matplotlib import pyplot as plt
import csv
import os
import numpy as np
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("shapenet_path", help="Path to the ShapeNetCore.V2")
    parser.add_argument("--output_dir", default="./output", help="Path to where the final files, will be saved")
    parser.add_argument("--synset_id", help="ShapeNet ID of the category", default="03001627")
    parser.add_argument(
        "--source_id", help="ID of the specific object in the category", default="1a6f615e8b1b5ae4dbbc9440457e303e"
    )
    parser.add_argument("--append_to_existing", default=False, type=bool)
    parser.add_argument("--resolution", nargs="*", default=[256, 256], type=int)
    parser.add_argument("--num_frames", default=200, type=int)
    parser.add_argument("--render_normals", default=True)
    parser.add_argument("--render_depth", default=True)
    parser.add_argument("--render_distance", default=True)
    parser.add_argument("--render_segmap", default=True)
    parser.add_argument("--render_nocs", default=True)
    parser.add_argument("--vis_nocs", default=False)
    parser.add_argument(
        "--use_opencv_image_frame",
        default=True,
        help="By default the cam2world matrix assume OpenGL/Blender image frame. Set to true to use OpenCV image "
             "frame instead.",
    )
    args = parser.parse_args()

    # check whether the shapenet dir exists
    assert os.path.exists(args.shapenet_path)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # write down objects models category id and source id
    append_csv(os.path.join(args.output_dir, "objects_info.csv"), [args.synset_id, args.source_id])

    bproc.init()

    # render
    shapenet_obj = bproc.loader.load_shapenet(
        args.shapenet_path, used_synset_id=args.synset_id, used_source_id=args.source_id, move_object_origin=False
    )
    # note: Blender applies a rotation of
    # array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    #        [ 0.00000000e+00, -4.37113883e-08, -1.00000000e+00],
    #        [ 0.00000000e+00,  1.00000000e+00, -4.37113883e-08],
    # to the loaded object. This will be reflected in the bpy object rotation attribute.
    #
    # The shapenet loader will apply this rotation to the object coordinates
    # world_p = world_R_cad * cad_p
    world_R_cad = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    # cad_p = cad_R_world * world_p
    cad_R_world = world_R_cad.T

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([5, -5, 5])
    light.set_energy(1000)

    for i in range(args.num_frames):
        location = bproc.sampler.sphere([0, 0, 0], radius=2, mode="SURFACE")
        rotation_matrix = bproc.camera.rotation_from_forward_vec(shapenet_obj.get_location() - location)
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
    shapenet_state = {
        "used_synset_id": shapenet_obj.get_cp("used_synset_id"),
        "used_source_id": shapenet_obj.get_cp("used_source_id"),
    }
    data["shapenet_state"] = [shapenet_state] * bproc.utility.num_frames()

    # prepare camera poses/states data
    cam_states = []
    for frame in range(bproc.utility.num_frames()):
        cam2world = bproc.camera.get_camera_pose(frame)
        if args.use_opencv_image_frame:
            # OpenGL: +z towards viewer, +y up
            # OpenCV: +z away from viewer, +y down
            # The function below is equivalent to post multiply by [1, 0, 0; 0, -1, 0; 0, 0, -1]
            cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
        cam_states.append(
            {"cam2world": cam2world, "cam_K": bproc.camera.get_intrinsics_as_K_matrix()}
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
        nocs_data = bproc.renderer.render_nocs()
        # nocs_data: dictionary, key=nocs, value=list containing nocs of all frames
        # the rendered NOCS coordinates are in the blender world frame
        # shape: (H, W, C)
        nf = len(nocs_data["nocs"])
        h, w, channels = nocs_data["nocs"][0].shape
        assert channels == 4
        for i in range(nf):
            # C, N
            world_nocs = np.transpose(nocs_data["nocs"][i].reshape((-1, channels)))
            world_nocs_p, nocs_mask = world_nocs[:3, ...], world_nocs[3, ...]
            # now we convert world_nocs_p to world_p (recenter and descale to get points in blender world frame)
            world_p = (world_nocs_p - 0.5) * 2
            # convert to the original CAD frame
            cad_p = cad_R_world @ world_p
            # to NOCS again (from [-1, 1] to [0, 1])
            cad_nocs_p = cad_p / 2 + 0.5
            # H, W, C
            cad_nocs = np.transpose(np.vstack([cad_nocs_p, nocs_mask])).reshape((h, w, channels))
            nocs_data["nocs"][i] = cad_nocs
        data.update(nocs_data)

    # visualize the NOCS
    if args.vis_nocs:
        nocs_pts = np.transpose(data["nocs"][0], (2, 0, 1))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(nocs_pts[0, ...], nocs_pts[1, ...], nocs_pts[2, ...], marker="o")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        fig.savefig("nocs_vis.pdf")

    bproc.writer.write_hdf5(args.output_dir, data, append_to_existing_output=args.append_to_existing)
