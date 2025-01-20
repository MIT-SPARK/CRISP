"""
This implements various generic classes and functions that are used in our code.

"""

import copy
import csv
import numpy as np
import open3d as o3d
# import open3d.visualization.gui as gui
# import open3d.visualization.rendering as rendering
import random
import string
import time
import torch
# import torch.nn.functional as F
# from pytorch3d import ops
# from pytorch3d import transforms
from scipy.spatial.transform import rotation
from scipy.spatial.transform.rotation import Rotation as Rot


class TrackingMeter():
    def __init__(self):
        self.val = []

    def add_item(self, item):
        self.val.append(item)


class Timer:
    def __init__(self, tag, print=False):
        self.tag = tag
        self.ts = None
        self.print = print

    def tic(self):
        self.ts = time()

    def toc(self):
        if self.print:
            print("{}: {}s".format(self.tag, time() - self.ts))
        return time()


def generate_random_keypoints(batch_size, model_keypoints):
    """
    input:
    batch_size      : int
    model_keypoints : torch.tensor of shape (K, 3, N)

    where
    K = number of cad models in a shape category
    N = number of keypoints

    output:
    keypoints   : torch.tensor of shape (batch_size, 3, N)
    rotation    : torch.tensor of shape (batch_size, 3, 3)
    translation : torch.tensor of shape (batch_size, 3, 1)
    shape       : torch.tensor of shape (batch_size, K, 1)
    """

    K = model_keypoints.shape[0]
    N = model_keypoints.shape[-1]

    rotation = torch.from_numpy(Rot.random(batch_size).as_matrix())
    # rotation = transforms.random_rotations(n=batch_size)
    translation = torch.rand(batch_size, 3, 1)

    shape = torch.rand(batch_size, K, 1)
    shape = shape/shape.sum(1).unsqueeze(1) # (batch_size, K, 1)
    shape = shape.unsqueeze(-1) # (batch_size, K, 1, 1)
    keypoints = torch.einsum('bkij,ukdn->bdn', shape, model_keypoints.unsqueeze(0))
    keypoints = rotation @ keypoints + translation

    return keypoints, rotation, translation, shape.squeeze(-1)


def check_rot_mat(R, tol=0.001):
    """
    This checks if the matrix R is a 3x3 rotation matrix
    R: rotation matrix: numpy.ndarray of size (3, 3)

    Output:
    True/False: determining if R is a rotation matrix/ or not
    """
    tol = tol
    if torch.norm(torch.matmul(R.T, R) - np.eye(3), ord='fro') < tol:
        return True
    else:
        return False


def pos_tensor_to_o3d(pos, estimate_normals=True):
    """
    inputs:
    pos: torch.tensor of shape (3, N)

    output:
    open3d PointCloud
    """
    pos_o3d = o3d.utility.Vector3dVector(pos.transpose(0, 1).to('cpu').numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    if estimate_normals:
        object.estimate_normals()

    return object


def generate_filename(chars=string.ascii_uppercase + string.digits, N=10):
    """function generates random strings of length N"""
    return ''.join(random.choice(chars) for _ in range(N))


def get_pose(R, t):
    """outputs 4x4 pose matrix given 3x3 rotation R and translation t"""
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3:] = t.reshape(3, 1)
    return P


def get_extrinsic(view_dir, up, location):
    """
    Generates the extrinsic matrix (for renderer in Open3D
    [open3d.visualization.rendering]) given:
    view_dir (3x1 numpy array)  = direction of the view of the camera
    up (3x1 numpy array)        = up direction
    location (3x1 numpy array)  = location of the camera

    All these must be specified in the global coordinate frame
    """
    x = np.cross(-up, view_dir)
    y = np.cross(view_dir, x)
    z = view_dir

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)

    R = np.array([x, y, z])
    t = np.zeros((3, 1))
    t[:, 0] = -np.dot(R, location)

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3:] = t

    return extrinsic


def get_camera_locations(camera_distance_vector, color=np.array([1.0, 0.0, 0.0]), number_of_locations=200):
    """ generating sphere of points around the object """
    camera_locations = o3d.geometry.PointCloud()
    for i in range(len(camera_distance_vector)):
        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=camera_distance_vector[i], resolution=20)
        sphere_points_new = sphere_mesh.sample_points_uniformly(number_of_points=number_of_locations)

        tempPoints = np.concatenate((sphere_points_new.points, camera_locations.points), axis=0)
        camera_locations.points = o3d.utility.Vector3dVector(tempPoints)

    camera_locations.paint_uniform_color(color)

    return camera_locations


def get_radius(object_diameter, cam_location):
    """ returns radius, which is the maximum distance from cam_location within which all points in the object lie"""
    return 100*np.sqrt(object_diameter**2 + np.linalg.norm(cam_location)**2)


def get_depth_pcd(centered_pcd, camera, radius, method='1'):
    """ This produces a depth point cloud. Input:
    centered_pcd (o3d.geometry.PointCloud object) = pcd that is centered at (0, 0, 0)
    camera (numpy.ndarray[float64[3, 1]])         = location of camera in the 3d space
    radius (float)                                = radius from camera location, beyond which points are not taken
    """
    pcd = copy.deepcopy(centered_pcd)

    """Method 1"""
    if method == '1':
        _, pt_map = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd = pcd.select_by_index(pt_map)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd

    """Method 2"""
    # Do not use Method 2. It constructs an artificial mesh from sampled/visible points.
    # This leads it to connect points that belong to distinct objects, thereby changing things.
    if method == '2':
        visible_mesh, _ = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd_visible = visible_mesh.sample_points_uniformly(number_of_points=10000)
        pcd_visible.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd_visible


def create_test_object(visualize=False):
    """ This function outputs a test 3D object that is placed at origin """

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    mesh = cyl + sphere + box + solid
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    if visualize == True:
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    return mesh, pcd, coordinate_frame


def save_metadata(file_name_with_location, dict_data, dict_data_columns):
    """ This functions stores dict_data as a csv file.
    Columns are given by dict_data_columns.
    """
    csv_file = file_name_with_location
    try:
        with open(csv_file, 'w') as csv_fileTemp:
            writer = csv.DictWriter(csv_fileTemp, fieldnames=dict_data_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def sample_depth_pcd(centered_pcd, camera_locations, radius, folder_name):
    """ This function computes depth pcd for each given camera location.
    The output pcd are stored as obj files in the folder specified by folder_name.
    The function also stores the centered_pcd and camera locations in the same folder.
    The function also creates a csv file listing file name and camera locations.
     """

    # saving object point cloud data
    centered_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(folder_name + 'object.pcd', centered_pcd)

    # saving camera locations point cloud
    camera_locations.paint_uniform_color([1.0, 0.0, 0.0])
    o3d.io.write_point_cloud(folder_name + 'camera_locations.pcd', camera_locations)

    # generating and saving depth point clouds
    dict_data_columns = ['file_name', 'camera_x', 'camera_y', 'camera_z']
    dict_data = []

    for camera in camera_locations.points:
        # generate depth point cloud
        depth_pcd = get_depth_pcd(centered_pcd, camera, radius)
        depth_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        # save depth_pcd as a .pcd file in folder folder_name
        file_name = generate_filename() + '.pcd'
        o3d.io.write_point_cloud(folder_name + file_name, depth_pcd)

        # add file_name and camera coordinates to dict_data
        dict_row = dict()
        dict_row['file_name'] = file_name
        dict_row['camera_x'] = camera[0]
        dict_row['camera_y'] = camera[1]
        dict_row['camera_z'] = camera[2]
        dict_data.append(dict_row)

    save_metadata(folder_name + 'metadata.csv', dict_data, dict_data_columns)



""" The following are test codes. Not to be used in the final project files. """


def test_get_depth_pcd():
    """ THis code tests depth point cloud from 3D objects
    input: 3d object, 3d object location, and camera extrinsic
    output: depth point cloud of the object (open3d.geometry.PointCloud)
    """

    cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    mesh = cyl + sphere + box + solid
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    centered_pcd = mesh.sample_points_uniformly(number_of_points=10000)
    diameter = np.linalg.norm(np.asarray(centered_pcd.get_max_bound()) - np.asarray(centered_pcd.get_min_bound()))
    o3d.visualization.draw_geometries([centered_pcd, coordinate_frame])

    camera = [0, 50, 1.5]
    radius = 20000.0
    pcd = get_depth_pcd(centered_pcd, camera, radius)
    o3d.visualization.draw_geometries([pcd, coordinate_frame])


# def test_rendering_depth_images(save_to_folder='../data/tmp/'):
#     """ This code tests rendering depth images from 3D Objects """
#     render = rendering.OffscreenRenderer(640, 480)
#
#     yellow = rendering.Material()
#     yellow.base_color = [1.0, 0.75, 0.0, 1.0]
#     yellow.shader = "defaultLit"
#
#     green = rendering.Material()
#     green.base_color = [0.0, 0.5, 0.0, 1.0]
#     green.shader = "defaultLit"
#
#     grey = rendering.Material()
#     grey.base_color = [0.7, 0.7, 0.7, 1.0]
#     grey.shader = "defaultLit"
#
#     white = rendering.Material()
#     white.base_color = [1.0, 1.0, 1.0, 1.0]
#     white.shader = "defaultLit"
#
#     cyl = o3d.geometry.TriangleMesh.create_cylinder(.05, 3)
#     cyl.compute_vertex_normals()
#     cyl.translate([-2, 0, 1.5])
#     sphere = o3d.geometry.TriangleMesh.create_sphere(.2)
#     sphere.compute_vertex_normals()
#     sphere.translate([-2, 0, 3])
#
#     box = o3d.geometry.TriangleMesh.create_box(2, 2, 1)
#     box.compute_vertex_normals()
#     box.translate([-1, -1, 0])
#     solid = o3d.geometry.TriangleMesh.create_icosahedron(0.5)
#     solid.compute_triangle_normals()
#     solid.compute_vertex_normals()
#     solid.translate([0, 0, 1.75])
#
#     # Adding to renderer
#     render.scene.add_geometry("cyl", cyl, green)
#     render.scene.add_geometry("sphere", sphere, yellow)
#     render.scene.add_geometry("box", box, grey)
#     render.scene.add_geometry("solid", solid, white)
#     render.setup_camera(60.0, [0, 0, 0], [0, 10, 0], [0, 0, 1])
#     render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
#                                      75000)
#     render.scene.scene.enable_sun_light(True)
#     render.scene.show_axes(True)
#
#     img = render.render_to_image()
#     o3d.io.write_image(save_to_folder+"test.png", img, 9)
#
#     # camera setup using (intrinsic, extrinsic)
#     # intrinsic = ()
#     # extrinsic = 4x4 matrix pose
#     view_dir = np.array([1, 0, 0])
#     cam_location = np.array([-5, 0, 0])
#     up = np.array([0, 0, 1])
#     extrinsic = get_extrinsic(view_dir=view_dir, up=up, location=cam_location)
#     # print(extrinsic)
#     intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#     # intrinsic = o3d.camera.PinholeCameraIntrinsic()
#     # intrinsic.set_intrinsics(width=2500, height=2000, fx=1000, fy=1000, cx=500, cy=500)
#     render.setup_camera(intrinsic, extrinsic)
#
#     img = render.render_to_image()
#     depth = render.render_to_depth_image()
#     o3d.io.write_image(save_to_folder+"test2.png", img)
#     # plt.imshow(depth)
#     # plt.show()
#     # o3d.io.write_image("tmp/test2_depth.png", depth, 9)
#     plt.imsave(save_to_folder+"test2_depth.png", depth)
#
#     # generate and save point cloud from the depth image
#     pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsic=intrinsic, depth_scale=0.01, depth_trunc=0.9,
#                                                           stride=10)
#     pcd.paint_uniform_color([0.5, 0.5, 0.5])
#     o3d.io.write_point_cloud(save_to_folder+"test2_depth.pcd", pcd)
#     o3d.visualization.draw_geometries([pcd])
#
#     # removing hidden points
#     # render.scene.hidden_point_removal()



if __name__ == '__main__':

    print("Testing generate_random_keypoints:")
    B = 10
    K = 5
    N = 7
    model_keypoints = torch.rand(K, 3, N)
    y, rot, trans, shape = generate_random_keypoints(batch_size=B, model_keypoints=model_keypoints)
    print(y.shape)
    print(rot.shape)
    print(trans.shape)
    print(shape.shape)

