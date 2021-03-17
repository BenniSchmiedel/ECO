import xarray as xr
import xgcm
from ECOdyagnostics import remap_vertical
import numpy as np
import warnings


import os
from pathlib import Path

TESTPATH = Path(os.path.dirname(os.path.abspath(__file__)))


_error = 1e-10

metrics_fr = {
    ("X",): ["e1t", "e1u", "e1v", "e1f"],
    ("Y",): ["e2t", "e2u", "e2v", "e2f"],
    ("Z",): ["e3t", "e3u", "e3v", "e3w"],
}
metrics_to = {
    ("X",): ["e1t", "e1u", "e1v", "e1f"],
    ("Y",): ["e2t", "e2u", "e2v", "e2f"],
    ("Z",): ["e3t_1d", "e3w_1d"],
}


def warning0(fr, to, error):
    N0 = (to != fr).sum().data
    N1 = (to == fr).sum().data
    m = np.abs(to - fr).max().data
    w0 = f"Number of points that are not equal: {N0}, maximum value of the absolute difference: {m}"
    w1 = f"Number of points that are     equal: {N1}"
    w2 = (
        f"v_to not perfectly equal to v_fr, testing with an authorized error of {error}"
    )
    warnings.warn(w2 + "\n" + w0 + "\n" + w1)


def _assert_same_domcfg(v_fr, v_to, error=_error):
    try:
        assert (v_to == v_fr).all()
    except AssertionError:
        warning0(v_fr, v_to, error=error)
        assert (v_to - v_fr < error).all()


def _assert_same_integrated_value(v_fr, v_to, e3_fr, e3_to, error=_error):
    try:
        int_fr = (v_fr * e3_fr).sum(dim="z_c")
        int_to = (v_to * e3_to).sum(dim="z_c")
    except ValueError:
        int_fr = (v_fr * e3_fr).sum(dim="z_f")
        int_to = (v_to * e3_to).sum(dim="z_f")
    print(int_to)
    print(int_fr)
    # import matplotlib.pyplot as plt
    # (int_fr-int_to).plot()
    # plt.show()
    try:
        assert (int_to == int_fr).all()
    except AssertionError:
        warning0(int_fr, int_to, error=error)
        assert (int_to - int_fr < error).all()


def open_ds():
    ds = xr.open_dataset(TESTPATH / "data/nemo_full_dataset.nc")
    ds.load()
    # correct error in domcfg
    ds["gdept_0"] = ds["gdept_0"].transpose("x_c", "y_c", "z_c")
    ds["gdept_0"][0] = ds["gdept_0"][1]
    ds["gdept_0"][-1] = ds["gdept_0"][-2]
    ds["gdept_0"][:, 0] = ds["gdept_0"][:, 1]
    ds["gdept_0"][:, -1] = ds["gdept_0"][:, -2]
    ds["gdepw_0"] = ds["gdepw_0"].transpose("x_c", "y_c", "z_f")
    ds["gdepw_0"][0] = ds["gdepw_0"][1]
    ds["gdepw_0"][-1] = ds["gdepw_0"][-2]
    ds["gdepw_0"][:, 0] = ds["gdepw_0"][:, 1]
    ds["gdepw_0"][:, -1] = ds["gdepw_0"][:, -2]
    ds["e3w_0"] = ds["e3w_0"].transpose("x_c", "y_c", "z_f")
    ds["e3w_0"][0] = ds["e3w_0"][1]
    ds["e3w_0"][-1] = ds["e3w_0"][-2]
    ds["e3w_0"][:, 0] = ds["e3w_0"][:, 1]
    ds["e3w_0"][:, -1] = ds["e3w_0"][:, -2]
    return ds


def test_reshaping():
    from ECOdyagnostics._remapping import _shape_to_shape_of_len_4 as _s

    assert _s((2, 2, 2, 2)) == (2, 2, 2, 2)
    assert _s((2, 2, 2)) == (2, 2, 2, 1)
    assert _s((2,)) == (2, 1, 1, 1)
    assert _s((2, 2, 2, 2, 3, 4)) == (2, 2, 2, 2 * 3 * 4)


def test_T_0_same_fr_and_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["thetao"] * 0 * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t,
        scale_factor_to=ds.e3t_1d,
    )
    _assert_same_domcfg(v_fr, v_to)


def test_W_0_same_fr_and_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["woce"] * 0
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3w,
        scale_factor_to=ds.e3w_1d,
    )
    _assert_same_domcfg(v_fr, v_to)


def test_W_1_same_fr_and_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False, metrics=metrics_fr)
    grid_to = xgcm.Grid(ds, periodic=False, metrics=metrics_to)

    v_fr = ds["woce"] * 0 + 1
    v_to = remap_vertical(v_fr, grid_fr, grid_to, axis="Z",)
    _assert_same_domcfg(v_fr, v_to)


def test_T_0():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["thetao"] * 0 * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t,
        scale_factor_to=ds.e3t_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3t, e3_to=ds.e3t_1d)


def test_U_0():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["uo"] * 0
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3u_0,
        scale_factor_to=ds.e3t_1d,
        z_fr=grid_fr.interp(ds.gdepw_0, "X", boundary="extend"),
        z_to=ds.gdepw_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3u_0, e3_to=ds.e3t_1d)


def test_U_1():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = (ds["uo"] * 0 + 1) * ds.umask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3u_0,
        scale_factor_to=ds.e3t_1d,
        z_fr=grid_fr.interp(ds.gdepw_0, "X", boundary="extend"),
        z_to=ds.gdepw_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3u_0, e3_to=ds.e3t_1d)


def test_U():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["uo"]
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3u_0,
        scale_factor_to=ds.e3t_1d,
        z_fr=grid_fr.interp(ds.gdepw_0, "X", boundary="extend"),
        z_to=ds.gdepw_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3u_0, e3_to=ds.e3t_1d)


def test_W():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False, metrics=metrics_fr)
    grid_to = xgcm.Grid(ds, periodic=False, metrics=metrics_to)

    v_fr = ds["woce"]
    v_to = remap_vertical(v_fr, grid_fr, grid_to, axis="Z",)
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3w, e3_to=ds.e3w_1d)


def test_W_1():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False, metrics=metrics_fr)
    grid_to = xgcm.Grid(ds, periodic=False, metrics=metrics_to)

    v_fr = ds["woce"] * 0 + 1
    v_to = remap_vertical(v_fr, grid_fr, grid_to, axis="Z",)
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3w, e3_to=ds.e3w_1d)


def test_T_1_auto_get_scale_factor_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False, metrics=metrics_to)

    v_fr = (ds["thetao"] * 0 + 1) * ds.tmask
    v_to = remap_vertical(v_fr, grid_fr, grid_to, axis="Z", scale_factor_fr=ds.e3t)
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3t, e3_to=ds.e3t_1d)


def test_T_1():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = (ds["thetao"] * 0 + 1) * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t,
        scale_factor_to=ds.e3t_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3t, e3_to=ds.e3t_1d)


def test_T_theta():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["thetao"] * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t,
        scale_factor_to=ds.e3t_1d,
    )
    _assert_same_integrated_value(v_fr, v_to, e3_fr=ds.e3t, e3_to=ds.e3t_1d)


def test_T_theta_full_automatic():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False, metrics=metrics_fr)
    grid_to = xgcm.Grid(ds, periodic=False, metrics=metrics_to)

    v_fr = ds["thetao"] * ds.tmask
    v_to = remap_vertical(v_fr, grid_fr, grid_to, axis="Z",)
    v_to_auto = remap_vertical(v_fr, grid_fr, grid_to, axis="Z")
    _assert_same_integrated_value(v_fr, v_to_auto, e3_fr=ds.e3t, e3_to=ds.e3t_1d)
    _assert_same_domcfg(v_to, v_to_auto)


def test_T_1_same_fr_and_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = (ds["thetao"] * 0 + 1) * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t_0,
        scale_factor_to=ds.e3t_0,
    )
    _assert_same_domcfg(v_fr, v_to)


def test_T_theta_same_fr_and_to():
    ds = open_ds()

    grid_fr = xgcm.Grid(ds, periodic=False)
    grid_to = xgcm.Grid(ds, periodic=False)

    v_fr = ds["thetao"] * ds.tmask
    v_to = remap_vertical(
        v_fr,
        grid_fr,
        grid_to,
        axis="Z",
        scale_factor_fr=ds.e3t_0,
        scale_factor_to=ds.e3t_0,
    )

    _assert_same_domcfg(v_fr, v_to)


if __name__ == "__main__":
    pass
