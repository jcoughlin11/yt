import numpy as np

from yt.frontends.stream.api import load_amr_grids
from yt.testing import _amr_grid_index
from yt.testing import _geom_transforms
from yt.utilities.answer_testing.framework import data_dir_load, requires_ds
from yt.visualization.accumulators import Accumulators
from yt.visualization.accumulators import get_row_major_index


g30 = "IsolatedGalaxy/galaxy0030/galaxy0030"


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

@requires_ds(g30, big_data=True)
def test_two_pts_same_cell_scalar():
    r"""
    Integrates the density field between two points that are in the same
    cell in the same node.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .55], [.155, .13, .56]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'Density')]
    accumulator.accumulate(field, is_vector=False)
    answer = np.array([0.00270012])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_two_pts_diff_cell_same_node_scalar():
    r"""
    Integrates the density field between two points that are in the same
    node but in different cells.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .44], [.22, .4, .8]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'Density')]
    accumulator.accumulate(field, is_vector=False)
    answer = np.array([.06582006])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_two_pts_diff_nodes_scalar():
    r"""
    Integrates the density field between two points that are in different
    nodes.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .55], [.3, .6, .75]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'Density')]
    accumulator.accumulate(field, is_vector=False)
    answer = np.array([0.09374269])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_npts_scalar():
    r"""
    Calculates the accumulation of the density field along the path 
    y^2=x and z^3=x from (0.1, 0.1, 0.1) -> (0.9, 0.9^(0.5), 0.9, (1./3.))
    through the IsolatedGalaxy ds.

    NOTE: The answer will depend on the number of points used to define
    the path, so if that changes, update the answer accordingly!
    The current answer is for N = 10
    """
    ds = data_dir_load(g30)
    path = _generate_path()
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'Density')]
    accumulator.accumulate(field, is_vector=False)
    answer = np.array([0.02904837,
                0.05809688,
                0.08693819,
                0.11599762,
                10.42846079,
                11.73346686,
                11.76252416,
                11.79157306,
                11.82062148])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_two_pts_same_cell_vector():
    r"""
    Integrates the velocity field between two points that are in the same
    cell in the same node.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .55], [.155, .13, .56]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([-0.00148564])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_two_pts_diff_cell_same_node_vector():
    r"""
    Integrates the velocity field between two points that are in the same
    node but in different cells.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .44], [.22, .4, .8]])
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator = Accumulators([path], ds)
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([0.0264723435])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_two_pts_diff_nodes_vector():
    r"""
    Integrates the velocity field between two points that are in different
    nodes.
    """
    ds = data_dir_load(g30)
    path = np.array([[.16, .14, .55], [.3, .6, .75]])
    accumulator = Accumulators([path], ds)
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([0.05266109])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)

@requires_ds(g30, big_data=True)
def test_npts_vector():
    r"""
    Calculates the accumulation of a test vector field and a test scalar
    field along the path y^2=x and z^3=x from (0.1, 0.1, 0.1) -> (0.9,
    0.9^(0.5), 0.9, (1./3.)) through the IsolatedGalaxy ds.

    NOTE: The answer will depend on the number of points used to define
    the path, so if that changes, update the answer accordingly!
    The current answer is for N = 10
    """
    ds = data_dir_load(g30)
    path = _generate_path()
    field = [('enzo', 'x-velocity'), ('enzo', 'y-velocity'), ('enzo', 'z-velocity')]
    accumulator = Accumulators([path], ds)
    accumulator.accumulate(field, is_vector=True)
    answer = np.array([0.01137986,
                0.03029503,
                0.06936842,
                0.19213309,
                0.96494272,
                0.63585663,
                0.56452914,
                0.5376544,
                0.52296114])
    np.testing.assert_array_almost_equal(accumulator.accum, answer)
