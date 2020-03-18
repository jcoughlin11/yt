import numpy as np

from yt.frontends.stream.api import load_amr_grids
from yt.testing import _amr_grid_index
from yt.testing import _geom_transforms
from yt.visualization.accumulators import get_row_major_index


g30 = "IsolatedGalaxy/galaxy0030/galaxy0030"


def _generate_fake_data(dims):
    """
    Creates fake field data to populate a grid with the number
    of cells given by dims.

    Gives each cell v(i, j, k) = i\hat{i} + j\hat{j} + k\hat{k} for the vector
    data and
    s(i, j, k) = i**2 + j**2 + k**2 for the scalar data.
    """
    # dims gives the number of cells in each dimension. We want the shape
    # of our test vector data to be: Ncells x N_spatial_dimensions so we can save
    # each component separately. The total number cells is obtained by
    # multiplying the number cells along each dimension together. The number
    # of spatial dimensions is just the length of the dims tuple
    v = np.zeros((np.cumprod(dims)[-1], len(dims)))
    s = np.zeros((np.cumprod(dims)[-1], 1))
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                I = get_row_major_index(dims, (i,j,k))
                v[I][0] = i
                v[I][1] = j
                v[I][2] = k
                s[I] = i**2 + j**2 + k**2
    return v, s


def _generate_fake_ds():
    """
    Creates a fake dataset to test the accumulator machinery. Based on
    fake_amr_dataset.
    """
    LE, RE = _geom_transforms["cartesian"]
    LE = np.array(LE)
    RE = np.array(RE)
    data = []
    for gspec in _amr_grid_index:
        level, left_edge, right_edge, dims = gspec
        left_edge = left_edge * (RE - LE) + LE
        right_edge = right_edge * (RE - LE) + LE
        gdata = dict(level = level,
                     left_edge = left_edge,
                     right_edge = right_edge,
                     dimensions = dims)
        gdata["test_vel"], gdata["test_dens"] = _generate_fake_data(dims)
        data.append(gdata)
    bbox = np.array([LE, RE]).T
    return load_amr_grids(data, [32, 32, 32], geometry="cartesian", bbox=bbox,
        length_unit=length_unit
    )


def _generate_fake_path():
    """
    Samples N (x,y,z) points along y^2=x and z^3=x from (0.1, 0.1, 0.1) ->
    (0.9,0.9^(0.5), 0.9, (1./3.)).
    """
    N = 10
    x = np.linspace(0.1, 0.9, N)
    y = np.linspace(0.1, np.sqrt(0.9), N)
    z = np.linspace(0.1, np.cbrt(0.9), N)
    path = np.stack([x,y,z], axis=1)
    return path


def test_custom_ds():
    """
    Calculates the accumulation of a test vector field and a test scalar
    field along the path y^2=x and z^3=x from (0.1, 0.1, 0.1) -> (0.9,
    0.9^(0.5), 0.9, (1./3.)) through a fake dataset volume. The fields
    are defined in _generate_fake_data and the grid structure of the ds
    is given by _amr_grid_index.
    """
    pass

def test_two_pts_same_cell():
    ds = yt.load(g30)
    path = np.array([[.16, .14, .55], [.155, .13, .56]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([-0.00043153, -0.00093425, -0.00011986])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

def test_two_pts_diff_nodes():
    ds = yt.load(g30)
    path = np.array([[.16, .14, .55], [.3, .6, .75]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([0.0120828, 0.05505828, 0.05266109])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)


