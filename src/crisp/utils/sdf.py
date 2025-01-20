import skimage.measure
import time
import numpy as np
import torch
import trimesh


def convert_sdf_samples_to_mesh(
    sdf_grid,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    Credit: From the DeepSDF repository https://github.com/facebookresearch/DeepSDF

    :param sdf_grid: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = sdf_grid.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except ValueError:
        print("0 not in SDF grid. Using mean of max and min.")
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    #num_verts = verts.shape[0]
    #num_faces = faces.shape[0]
    #verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    #for i in range(0, num_verts):
    #    verts_tuple[i] = tuple(mesh_points[i, :])
    #faces_building = []
    #for i in range(0, num_faces):
    #    faces_building.append(((faces[i, :].tolist(),)))
    #faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    mesh = trimesh.Trimesh(vertices=mesh_points,
                           faces=faces, validate=True)

    #el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    #el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    #ply_data = plyfile.PlyData([el_verts, el_faces])
    #logging.debug("saving mesh to %s" % (ply_filename_out))
    #ply_data.write(ply_filename_out)

    #logging.debug("converting to ply format and writing to file took {} s".format(time.time() - start_time))
    return mesh


def create_sdf_samples_generic(model_fn, N=256, max_batch=64**3, cube_center=None, cube_scale=None):
    """Sample SDF values for creating a mesh that takes in a closure

    Parameters
    ----------
    model_fn : closure function that generates sdf value
    N
    max_batch
    cube_center
    cube_scale
    """
    if cube_center is None:
        cube_center = np.array([0, 0, 0])
    if cube_scale is None:
        cube_scale = 1.0

    start = time.time()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner of the grid, not the middle
    voxel_origin = cube_center - cube_scale / 2
    voxel_size = cube_scale / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    # first 3 columns: coordinates; last column: SDF values
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].unsqueeze(0).cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            model_fn(coords=sample_subset).squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    #print("SDF sampling takes: %f" % (end - start))
    return sdf_values, voxel_size, voxel_origin


def create_sdf_samples_with_latent_vec(latent_vec, model, N=256, max_batch=64**3, cube_center=None, cube_scale=None):
    """Sample SDF values for creating a mesh

    Parameters
    ----------
    image : (1, 3, H, W)
    model : Shape mode (will be modulated by the image)
    N
    max_batch
    cube_center
    cube_scale
    """
    if cube_center is None:
        cube_center = np.array([0, 0, 0])
    if cube_scale is None:
        cube_scale = 1.0

    start = time.time()
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner of the grid, not the middle
    voxel_origin = cube_center - cube_scale / 2
    voxel_size = cube_scale / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    # first 3 columns: coordinates; last column: SDF values
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].unsqueeze(0).cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            model.forward_shape(latent_vec, coords=sample_subset).squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("SDF sampling takes: %f" % (end - start))
    return sdf_values, voxel_size, voxel_origin


def create_sdf_samples(image, model, N=256, max_batch=64**3, cube_center=None, cube_scale=None, model_kwargs=None):
    """Sample SDF values for creating a mesh

    Parameters
    ----------
    image : (1, 3, H, W)
    model : Shape mode (will be modulated by the image)
    N
    max_batch
    cube_center
    cube_scale
    model_kwargs: additional keyword parameters to pass to the model forward function
    """
    if cube_center is None:
        cube_center = np.array([0, 0, 0])
    if cube_scale is None:
        cube_scale = 1.0
    if model_kwargs is None:
        model_kwargs = {}

    start = time.time()
    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner of the grid, not the middle
    voxel_origin = cube_center - cube_scale / 2
    voxel_size = cube_scale / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    # first 3 columns: coordinates; last column: SDF values
    samples = torch.zeros(N**3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].unsqueeze(0).cuda()
        samples[head : min(head + max_batch, num_samples), 3] = (
            model(img=image, coords=sample_subset, **model_kwargs).squeeze().detach().cpu()  # .squeeze(1)
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("SDF sampling takes: %f" % (end - start))
    return sdf_values, voxel_size, voxel_origin
