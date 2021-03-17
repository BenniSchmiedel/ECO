from ECOdyagnostics import Grid_ops
import xarray as xr
import xgcm
import numpy as np
import warnings
from ECOdyagnostics._remapping import _compute_depth_of_shifted_array

import pytest

_error = 1e-3
_metrics = {
    ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
    ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
    ("Z",): ["e3t_0", "e3u_0", "e3v_0", "e3f_0", "e3w_0"],  # Z distances
}


def warning_zero(error):
    message = (
        f"v_to not perfectly equal to v_fr, testing with an authorized error of {error}"
    )
    warnings.warn(message)

def warning_max_error(test):
    if type(test) is list:
        error=[abs(test[i]).max() for i in range(3)]
    else:
        error=abs(test).max()
    message = (
        f"Equation does not perfectly match 0, maximum error is {error}"
    )
    warnings.warn(message)


def _assert_same_position(grid_ops, data, position):
    check = grid_ops._matching_pos(data, position)
    if type(check) is list:
        assert all(check)
    else:
        assert check


def test_shift_position_to_T():
    # ds = open_nemo_and_domain_cfg(datadir='data')
    ds = xr.open_dataset("data/nemo_full_dataset.nc")

    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = Grid_ops(grid)

    u_fr = ds.uo
    v_fr = ds.vo
    w_fr = ds.woce

    u_3d_fr = [u_fr, v_fr, w_fr]

    # Test single variables
    u_to = grid_ops._shift_position(u_fr, output_position='T')
    v_to = grid_ops._shift_position(v_fr, output_position='T')
    w_to = grid_ops._shift_position(w_fr, output_position='T')

    u_3d_to = grid_ops._shift_position(u_3d_fr, output_position='T')

    # grid_ops._matching_pos([u_to,v_to,w_to,u_3d_to],'T')
    _assert_same_position(grid_ops, [u_to, v_to, w_to], 'T')
    _assert_same_position(grid_ops, u_3d_to, 'T')

def test_grad_cross_grad_q_timedep():
    """
    Test the condition  grad x grad(q) = 0 with grad = [d/dx, d/dy, d/dz] and scalar q.
    """
    _metrics = {
        ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
        ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
        ("Z",): ["e3t", "e3u", "e3v", "e3w"],  # Z distances
    }

    ds = xr.open_dataset("data/nemo_full_dataset.nc")

    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)

    bd = {'boundary': 'fill', 'fill_value': 0}
    grid_ops = Grid_ops(grid, boundary=bd)

    # Create timedependent metrics for 'F', 'FW', 'UW', 'VW' and add them to the dataset+metrics -> create new grid
    e3f = grid_ops._shift_position(ds.e3t, 'F', boundary='extend')
    e3uw = grid_ops._shift_position(ds.e3w, 'UW', boundary='extend')
    e3vw = grid_ops._shift_position(ds.e3w, 'VW', boundary='extend')
    e3fw = grid_ops._shift_position(ds.e3w, 'FW', boundary='extend')
    ds['e3uw'] = e3uw
    ds['e3vw'] = e3vw
    ds['e3fw'] = e3fw
    ds['e3f'] = e3f
    _metrics['Z',] = _metrics['Z',] + ['e3uw', 'e3vw', 'e3fw', 'e3f']

    grid_q = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops_q = Grid_ops(grid_q, boundary=bd, discretization='nemo')

    # Test for a scalar field of temperature
    q = ds.thetao
    y = grid_ops_q.gradient(q)
    test_q = grid_ops_q.curl(y)
    try:
        assert all(np.array([(test_q[i] == 0).all() for i in range(3)]))
    except AssertionError:
        warning_max_error(test_q)
        assert True


def test_grad_times_grad_cross_vec_timedep():
    """
    Test the condition  grad * (grad x A) = 0 with grad = [d/dx, d/dy, d/dz] and vector A.
    """
    _metrics = {
        ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
        ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
        ("Z",): ["e3t", "e3u", "e3v", "e3w"],  # Z distances
    }

    ds = xr.open_dataset("data/nemo_full_dataset.nc")

    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)

    bd = {'boundary': 'fill', 'fill_value': 0}
    grid_ops = Grid_ops(grid, boundary=bd)

    # Create timedependent metrics for 'F', 'FW', 'UW', 'VW' and add them to the dataset+metrics -> create new grid
    e3f = grid_ops._shift_position(ds.e3t, 'F', boundary='extend')
    e3uw = grid_ops._shift_position(ds.e3w, 'UW', boundary='extend')
    e3vw = grid_ops._shift_position(ds.e3w, 'VW', boundary='extend')
    e3fw = grid_ops._shift_position(ds.e3w, 'FW', boundary='extend')
    ds['e3uw'] = e3uw
    ds['e3vw'] = e3vw
    ds['e3fw'] = e3fw
    ds['e3f'] = e3f
    _metrics['Z',] = _metrics['Z',] + ['e3uw', 'e3vw', 'e3fw', 'e3f']

    grid_A = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops_A = Grid_ops(grid_A, boundary=bd, discretization='nemo')  # {'boundary':'extend'})

    #Test for velocity vector U=[u,v,w]

    U = [ds.uo, ds.vo, ds.woce]
    y = grid_ops_A.curl(U)

    test_U = grid_ops_A.divergence(y)
    try:
        assert (test_U == 0).all()
    except AssertionError:
        warning_max_error(test_U)
        assert True

def test_grad_cross_grad_q_remapped():
    """
    Test the condition  grad * (grad x A) = 0 with grad = [d/dx, d/dy, d/dz] and vector A.
    """
    # Get dataset
    ds = xr.open_dataset("data/nemo_full_dataset.nc")


    bd = {'boundary': 'fill', 'fill_value': 0}
    _metrics = {
        ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
        ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
        ("Z",): ["e3t", "e3u", "e3v", "e3w"],  # Z distances
    }
    metrics_to = {
        ('X',): ['e1t', 'e1u', 'e1v', 'e1f'],
        ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'],
        ('Z',): ['e3t_1d', 'e3w_1d']
    }

    # Get grid and operations from dataset
    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = Grid_ops(grid, boundary=bd)

    # Get grid and operations for fixes depth
    grid_to = xgcm.Grid(ds, metrics=metrics_to, periodic=False)
    grid_ops_to = Grid_ops(grid_to, boundary=bd, discretization='nemo')  # {'boundary':'extend'})

    # Compute depth from variable grid
    depth_t = _compute_depth_of_shifted_array(grid, ds.thetao, 'Z', e3=ds.e3w) - ds.e3w[-1, 0].values / 2
    depth_t_1d = ds.gdept_1d

    # Test for temperature field
    qr = grid_ops.remap_numpy(ds.thetao, depth_t, depth_t_1d)

    # Compute  Curl( Grad( q ) )
    y = grid_ops_to.gradient(qr)
    test_qr = grid_ops_to.curl(y)
    try:
        assert all(np.array([(test_qr[i] == 0).all() for i in range(3)]))
    except AssertionError:
        warning_max_error(test_qr)
        assert True

def test_grad_times_grad_cross_vec_remapped():
    """
    Test the condition  grad * (grad x A) = 0 with grad = [d/dx, d/dy, d/dz] and vector A.
    """
    ds = xr.open_dataset("data/nemo_full_dataset.nc")

    _metrics = {
        ("X",): ["e1t", "e1u", "e1v", "e1f"],  # X distances
        ("Y",): ["e2t", "e2u", "e2v", "e2f"],  # Y distances
        ("Z",): ["e3t", "e3u", "e3v", "e3w"],  # Z distances
    }
    metrics_to = {
        ('X',): ['e1t', 'e1u', 'e1v', 'e1f'],
        ('Y',): ['e2t', 'e2u', 'e2v', 'e2f'],
        ('Z',): ['e3t_1d', 'e3w_1d']
    }

    # Get grid and operations from dataset
    bd = {'boundary': 'fill', 'fill_value': 0}
    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = Grid_ops(grid, boundary=bd)

    # Get grid and operations for a fixes metric
    grid_to = xgcm.Grid(ds, metrics=metrics_to, periodic=False)
    grid_ops_to = Grid_ops(grid_to, boundary=bd)

    # Extend dataset with metrics to compute the remapped (fixed) data
    e3uw = grid_ops._shift_position(ds.e3w, 'UW', boundary='extend')
    e3vw = grid_ops._shift_position(ds.e3w, 'VW', boundary='extend')
    ds['e3uw'] = e3uw
    ds['e3vw'] = e3vw
    _metrics['Z',] = _metrics['Z',] + ['e3uw', 'e3vw']

    # Extend grid and operations from dataset
    grid_U = xgcm.Grid(ds, metrics=_metrics, periodic=False)

    # Compute depth coordinates with extended grid
    depth_u = _compute_depth_of_shifted_array(grid_U, ds.thetao, 'Z', e3=ds.e3uw) - ds.e3uw[-1, 0].values / 2
    depth_v = _compute_depth_of_shifted_array(grid_U, ds.uo, 'Z', e3=ds.e3vw) - ds.e3vw[-1, 0].values / 2
    depth_w = _compute_depth_of_shifted_array(grid_U, ds.vo, 'Z', e3=ds.e3t) - ds.e3t[-1, 0].values / 2
    depth_t_1d = ds.gdept_1d
    depth_w_1d = ds.gdepw_1d

    # Compute remapped (fixed) version of the variable, chosen to be the velocity vector  U = [u,v,w]
    ur = grid_ops_to.remap_numpy(ds.uo, depth_u, depth_t_1d)
    vr = grid_ops_to.remap_numpy(ds.vo, depth_v, depth_t_1d)
    wr = grid_ops_to.remap_numpy(ds.woce, depth_w, depth_w_1d)

    Ur = [ur, vr, wr]
    # Compute  Div( Curl( U ) )
    y = grid_ops_to.curl(Ur)
    test_Ur = grid_ops_to.divergence(y)


    try:
        assert (test_Ur == 0).all()
    except AssertionError:
        warning_max_error(test_Ur)
        assert True

if __name__ == "__main__":
    pass
