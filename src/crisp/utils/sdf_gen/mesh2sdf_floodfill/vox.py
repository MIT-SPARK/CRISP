import numpy as np
import os
import pickle
import struct


class VoxSDF:
    def __init__(self, dims=None, res=None, grid2world=None, sdf=None):
        """Class representing voxelized grid representing SDF function"""
        self.dims = dims
        self.res = res
        self.grid2world = grid2world
        self.sdf = sdf

    def __eq__(self, other):
        if isinstance(other, VoxSDF):
            if other.dims is None or other.res is None or other.grid2world is None or other.sdf is None:
                return False
            flag1 = self.dims[0] == other.dims[0] and self.dims[1] == other.dims[1] and self.dims[2] == other.dims[2]
            if not flag1:
                return False
            flag2 = self.res == other.res
            if not flag2:
                return False
            flag3 = np.array_equal(self.grid2world, other.grid2world)
            if not flag3:
                return False
            flag4 = np.array_equal(self.sdf, other.sdf)
            return flag4

        return False

    def __repr__(self):
        text = f"Dims: {self.dims} \nResolution: {self.res}\nGrid2World: \n{self.grid2world}\nSDF: shape={self.sdf.shape}\n"
        return text


class VoxSDFGrad:
    def __init__(self, dims, res, grid2world, sdf_grad):
        """Class representing voxelized grid representing SDF function's gradients

        Args:
            dims:
            res:
            grid2world:
            sdf_grad: a size-3 list of (x_dim, y_dim, z_dim) partial w.r.t. to x,y,z coordinates
        """
        self.dims = dims
        self.res = res
        self.grid2world = grid2world
        self.sdf_grad = sdf_grad

    def __eq__(self, other):
        if isinstance(other, VoxSDFGrad):
            if other.dims is None or other.res is None or other.grid2world is None or other.sdf is None:
                return False
            flag1 = self.dims[0] == other.dims[0] and self.dims[1] == other.dims[1] and self.dims[2] == other.dims[2]
            if not flag1:
                return False
            flag2 = self.res == other.res
            if not flag2:
                return False
            flag3 = np.array_equal(self.grid2world, other.grid2world)
            if not flag3:
                return False
            flag4 = np.array_equal(self.sdf_grad, other.sdf_grad)
            return flag4

        return False

    def __repr__(self):
        text = f"Dims: {self.dims} \nResolution: {self.res}\nGrid2World: \n{self.grid2world}\n" + \
               f"SDF grad: shape={self.sdf_grad[0].shape}, {self.sdf_grad[1].shape}, {self.sdf_grad[2].shape}\n"
        return text


def load_vox_sdf_grad(filename) -> VoxSDFGrad:
    """Load vox SDF gradients"""
    assert os.path.isfile(filename), "file not found: %s" % filename
    with open(filename, "rb") as f:
        vox_sdf_grad = pickle.load(f)
    return vox_sdf_grad


def write_vox_sdf_grad(filename, s):
    """Write vox SDF gradients"""
    with open(filename, "wb") as f:
        pickle.dump(s, f)
    assert os.path.isfile(filename), "file write failed: %s" % filename


def load_vox_sdf(filename) -> VoxSDF:
    assert os.path.isfile(filename), "file not found: %s" % filename

    fin = open(filename, "rb")

    s = VoxSDF()
    s.dims = [0, 0, 0]
    s.dims[0] = struct.unpack("I", fin.read(4))[0]
    s.dims[1] = struct.unpack("I", fin.read(4))[0]
    s.dims[2] = struct.unpack("I", fin.read(4))[0]
    s.res = struct.unpack("f", fin.read(4))[0]
    n_elems = s.dims[0] * s.dims[1] * s.dims[2]

    s.grid2world = struct.unpack("f" * 16, fin.read(16 * 4))
    s.grid2world = np.asarray(s.grid2world, dtype=np.float32).reshape([4, 4], order="F")
    fin.close()

    # -> sdf 1-channel
    offset = 4 * (3 + 1 + 16)
    # original
    # s.sdf = np.fromfile(filename, count=n_elems, dtype=np.float32, offset=offset).reshape(
    #    [s.dims[2], s.dims[1], s.dims[0]])
    # new (3D array dimensions (x, y, z) )
    s.sdf = np.fromfile(filename, count=n_elems, dtype=np.float32, offset=offset).reshape(
        [s.dims[0], s.dims[1], s.dims[2]], order="F"
    )
    # <-

    return s


def write_vox_sdf(filename, s):
    fout = open(filename, "wb")
    fout.write(struct.pack("I", s.dims[0]))
    fout.write(struct.pack("I", s.dims[1]))
    fout.write(struct.pack("I", s.dims[2]))
    fout.write(struct.pack("f", s.res))
    n_elems = np.prod(s.dims)
    fout.write(struct.pack("f" * 16, *s.grid2world.flatten("F")))
    # original
    # fout.write(struct.pack("f" * n_elems, *s.sdf.flatten("C")))
    # new (due to changes in loading into sdf)
    fout.write(struct.pack("f" * n_elems, *s.sdf.flatten("F")))
    fout.close()
