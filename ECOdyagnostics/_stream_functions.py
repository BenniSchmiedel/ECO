def compute_moc(v, grid, X="X", Z="Z"):
    psi = _compute_moc_with_v_at_cst_depth(v, grid, X="X", Z="Z")
    return psi


def _compute_moc_with_v_at_cst_depth(v, grid, X="X", Z="Z"):
    """
    Compute the meridional overturning streamfunction.
    """
    v_x_dx = grid.integrate(v, axis=X)  # (vo_to * domcfg_to['y_f_dif']).sum(dim='x_c')
    # integrate from top to bot
    psi = grid.cumint(v_x_dx, axis=Z, boundary="fill", fill_value=0) * 1e-6
    # convert -> from bot to top
    psi = psi - psi.isel({grid.axes[Z]._get_axis_coord(psi)[1]: -1})
    return psi
