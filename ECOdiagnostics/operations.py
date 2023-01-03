import numpy as np
import xarray as xr
import xgcm


class Grid_ops:
    """
    An object that includes operations for variables defined on a xgcm compatible grid.
    Those operations are defined to simplify dealing with mathematical operations that shift grid point positions
    when applied.

    Note: The operations include approximations of variables onto neighbouring grid positions by interpolation.
    For low grid resolutions the approximation through interpolation might introduce large deviations from the
    original data, especially where strong value shifts are present.
    """

    def __init__(self,
                 grid,
                 discretization='standard',
                 boundary={'boundary': 'fill', 'fill_value': 0},
                 maskargs={'mask':None}):
        """
        Creates the standard configuration for the operations used.
        The configuration of an operation can be customized by will if required.
        """
        self.grid = grid
        self.discretization = discretization
        self.boundary = boundary
        # Position types, hard coded
        self.points = {'T': (('z_c', 'y_c', 'x_c'), ('z_c', 'x_c'), ('z_c', 'y_c'), ('y_c', 'x_c'), ('z_c',)),
                       'U': (('z_c', 'y_c', 'x_f'), ('y_c', 'x_f'), ('x_f',)),
                       'V': (('z_c', 'y_f', 'x_c'), ('y_f', 'x_c'), ('y_f',)),
                       'F': (('z_c', 'y_f', 'x_f'), ('y_f', 'x_f')),
                       'W': (('z_f', 'y_c', 'x_c'), ('y_c', 'x_c'), ('z_f',)),
                       'UW': (('z_f', 'y_c', 'x_f'),),
                       'VW': (('z_f', 'y_f', 'x_c'),),
                       'FW': (('z_f', 'y_f', 'x_f'),)
                       }
        self.shift_mask = {}
        self.maskargs = maskargs

    def _update(self,
                metric):
        
        for m, v in metric.items():
            if v.name not in [var.name for var in self.grid._metrics[frozenset({m})]]:
                self.grid._metrics[frozenset({m})].append(v)
        
        return self.grid
                
    def _data_skip(self, da, skip):
        """
        Returns a cut version of data vector da by skipping physical dimensions


        """
        if type(skip) is int:
            da = da[:skip] + da[skip + 1:]
        elif type(skip) is list:
            k = 0
            for ind in skip:
                da = da[:ind - k] + da[ind + 1 - k:]
                k += 1
        else:
            raise Exception("""Input for "skip" not understood. Give integer or list.""")
        return da

    def _get_dims(self, da):
        """
        Returns the spatial position of the data variable of data vector.
        Note: Output does not include 't'-dimension

        :param da:  data variable or vector as xarray.Dataarray or List(xarray.Dataarray, ...)

        :return:    position of data variable or vector
        """

        # If da is a list (vector), return a list with the dimension tuple for every direction.
        if type(da) == list:
            pos = []
            for i in range(len(da)):
                dims = da[i].dims
                # Remove temporal dimension if present
                if 't' in dims:
                    ind = dims.index('t')
                    dims = dims[:ind] + dims[ind + 1:]
                pos.append(dims)
        else:
            dims = da.dims
            if 't' in dims:
                ind = dims.index('t')
                dims = dims[:ind] + dims[ind + 1:]
            pos = dims

        return pos

    def _get_dims_order(self, da):

        dims = da.dims
        pos = self._get_position(da)
        if 't' in dims:
            ind = dims.index('t')
            dims = dims[:ind] + dims[ind + 1:]

            if dims in self.points[pos]:
                dims = ('t',) + dims
            else:
                for p in self.points[pos]:
                    if len(dims) == len(p) and np.all([d in p for d in dims]):
                        dims = ('t',) + p
        else:
            for p in self.points[pos]:
                if len(dims) == len(p) and np.all([d in p for d in dims]):
                    dims = p
        return dims

    def _product_transpose(self, da):
        
        p = np.prod(da)
        if 't' in p.dims:
            if 't' in dims:
                p= p.transpose(*dims)
            else:
                p= p.transpose('t',*dims)
        else:
            if 't' in dims:
                p= p.transpose(*dims[1:])
            else:
                p= p.transpose(*dims)
                
        return dV
    def _combine_metric(self, da, axes, skip=None):
        """
        Returns the product of metrics corresponding to the respective datavariable.

        :param da:  data variable or vector as xarray.Dataarray or List(xarray.Dataarray, ...)

        :return:    position of data varaible or vector
        """
        if type(skip) is list:
            pass
        elif skip is None:
            skip = list('_')
        else:
            skip = list(skip)

        metric = 1
        for i in range(len(axes)):
            if any(axes[i] == np.array(skip)):
                pass
            else:
                metric = metric * self.grid.get_metric(da, axes[i])

        return metric

    def _get_metric_by_pos(self, axes, pos, combine = False, case = 'largest'):
        """
        Returns the metrics for given axes corresponding to a grid point position.

        :param da:  data variable or vector as xarray.Dataarray or List(xarray.Dataarray, ...)

        :return:    position of data varaible or vector
        """
        if not type(axes) is list: axes = [axes]
        metric_out = {}
        metric_out_dims = {}
        for ax in axes:
            check = False
            for i in self.points[pos]:
                metric_dims = self._get_dims(self.grid._metrics[frozenset(ax)])

                if any([i == a for a in metric_dims]):
                    check = True
                    ind = np.where([i == a for a in metric_dims])[0]
                    if ax in metric_out:
                        if case == 'largest' and len(i) < len(metric_out[ax].dims):
                            continue
                        elif case == 'smallest' and len(i) > len(metric_out[ax].dims):
                            continue
                            
                    if len(ind) == 1:
                        metric_out[ax] = self.grid._metrics[frozenset(ax)][ind[0]]
                        metric_out_dims[ax] = metric_dims[ind[0]]
                    else:
                        raise Exception(
                            "Found multiple matches for the metric, ensure that metrics are not doubled.")
                        
            if not check:
                raise Exception("No matching metric was found on axis %s for position %s" % (ax, pos))
        metric_out = list(metric_out.values())
        metric_out_dims = list(metric_out_dims.values())

        if combine and len(metric_out)>1:
            metric = 1
            try:
                for m in [metric_out[i] for i in np.argsort(metric_out_dims,axis=0)[::-1]]:
                    metric *= m
            except:
                for m in [metric_out[i[0]]for i in np.argsort(metric_out_dims,axis=0)[::-1]]:
                    metric *= m
            
            name = '_'.join(['{}']*len(metric_out)).format(*tuple(m.name for m in metric_out))
            metric_out = [metric.rename(name)]  
            
        return metric_out

    def _get_shift_mask(self, da, fill_value=None, scaling=1):
        """
        :param da: dataset from which the mask is generated from

        Return the 3D boundary mask which follows the terrain needed when a shift is performed.
        - scaling*boundary_value for the boundary value at the shifted position
        - 0 inside the boundary
        - nan outside the boundary

        If a specific fill_value is used it replaces scaling*boundary_value

        Boundary values of da have to be nan!

        Applicable for positions 'U', 'V', 'W' and 'F' with a shift to 'T'.
        When first calculated it is added to self.shift_mask.

        :return: da_mask
        """
        t = False
        if da.dims[0]=='t':
            t=True
            mask = np.zeros(da.shape[1:])
            da_ref = da[0]
        else:
            mask = np.zeros(da.shape)
            da_ref = da
        x_len=da_ref.shape[2]
        y_len=da_ref.shape[1]
        z_len=da_ref.shape[0]
        pos = self._get_position(da)

        ### Distinguish mask by position
        if pos=='U':
            for z in range(z_len):
                for y in range(y_len):
                    for x in range(x_len-1):
                        v = da_ref[z,y,x]
                        v1 = da_ref[z,y,x+1]
                        if x==0:
                            skip=False
                        elif x==x_len-2:
                            if np.isnan(da_ref[z,y,x+1]):
                                mask[z,y,x+1] = np.nan

                        if skip:
                            skip = False
                            continue
                        else:
                            if not np.isnan(v) and not np.isnan(v1):
                                continue

                            if np.isnan(v) and np.isnan(v1):
                                mask[z,y,x] = np.nan
                            elif np.isnan(v) and not np.isnan(v1):
                                mask[z,y,x] = np.nan
                                if fill_value is None:
                                    mask[z, y, x + 1] = scaling*v1
                                else:
                                    mask[z, y, x + 1] = fill_value
                                skip = True
                            elif not np.isnan(v) and np.isnan(v1):
                                #mask[z, y, x] = 1
                                if fill_value is None:
                                    mask[z, y, x + 1] = scaling * v
                                else:
                                    mask[z, y, x + 1] = fill_value
                                skip = True
            if t:
                return da[0].copy(data=mask).rename('shift_mask' + pos)
            else:
                return da.copy(data=mask).rename('shift_mask' + pos)

        if pos == 'V':
            for z in range(z_len):
                for x in range(x_len):
                    for y in range(y_len-1):
                        v = da_ref[z, y, x]
                        v1 = da_ref[z, y+1, x]

                        if y == 0:
                            skip = False
                        elif y == y_len - 2:
                            if np.isnan(da_ref[z, y+1, x]):
                                mask[z, y+1, x] = np.nan
                        if skip:
                            skip = False
                            continue
                        else:
                            if not np.isnan(v) and not np.isnan(v1):
                                continue
                            if np.isnan(v) and np.isnan(v1):
                                mask[z, y, x] = np.nan
                            elif np.isnan(v) and not np.isnan(v1):
                                mask[z, y, x] = np.nan
                                if fill_value is None:
                                    mask[z, y+1, x] = scaling*v1
                                else:
                                    mask[z, y+1, x] = fill_value
                                skip = True
                            elif not np.isnan(v) and np.isnan(v1):
                                if fill_value is None:
                                    mask[z, y + 1, x] = scaling * v
                                else:
                                    mask[z, y + 1, x] = fill_value
                                skip = True
            if t:
                return da[0].copy(data=mask).rename('shift_mask' + pos)
            else:
                return da.copy(data=mask).rename('shift_mask' + pos)

        if pos == 'W':
            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len - 1):
                        v = da_ref[z, y, x]
                        v1 = da_ref[z+1, y, x]

                        if z == 0:
                            skip = False
                        elif z == z_len - 2:
                            if np.isnan(da_ref[z+1, y, x]):
                                mask[z+1, y, x] = np.nan
                        if skip:
                            skip = False
                            continue
                        else:
                            if not np.isnan(v) and not np.isnan(v1):
                                continue
                            if np.isnan(v) and np.isnan(v1):
                                mask[z, y, x] = np.nan
                            elif np.isnan(v) and not np.isnan(v1):
                                mask[z, y, x] = np.nan
                                if fill_value is None:
                                    mask[z+1, y, x] = scaling*v1
                                else:
                                    mask[z+1, y, x] = fill_value
                                skip = True
                            elif not np.isnan(v) and np.isnan(v1):
                                if fill_value is None:
                                    mask[z+1, y, x] = scaling * v
                                else:
                                    mask[z+1, y, x] = fill_value
                                skip = True
            if t:
                return da[0].copy(data=mask).rename('shift_mask' + pos)
            else:
                return da.copy(data=mask).rename('shift_mask' + pos)

        """if pos == 'F':
            if 'F' in self.shift_mask:
                mask = self.shift_mask[pos]
            elif 'U' not in self.shift_mask:
                raise Exception('Please run shift mask for U first')
            elif 'V' not in self.shift_mask:
                raise Exception('Please run shift mask for V first')
            else:
                if t:
                    self.shift_mask[pos] = da[0].copy(data=self.shift_mask['U'].values*self.shift_mask['V'].values).rename('shift_mask'+pos)
                else:
                    self.shift_mask[pos] = da.copy(data=self.shift_mask['U'].values*self.shift_mask['V'].values).rename('shift_mask'+pos)
        if t:
            return da[0].copy(data=mask).rename('shift_mask'+pos)
        else:
            return da.copy(data=mask).rename('shift_mask'+pos)"""

    def _get_position(self, da, skip=None):
        """
        Returns the spatial position of the data variable of data vector.
        Note: Output does not include 't'-dimension

        :param da:  data variable or vector as xarray.DataArray or List(xarray.DataArray, ...)

        :return:    position of data variable or vector
        """

        # If da is a list (vector), return a list with the dimension tuple for every direction.
        if skip != None:
            da = self._data_skip(da, skip)
        dims = self._get_dims(da)
        if type(dims) is list:
            positions = []
            for dim in dims:
                for pos in self.points:
                    if dim in self.points[pos]:
                        positions.append(pos)
                        break
            if len(positions) != len(dims):
                raise Exception("""Could not get an appropriate grip point position for the given dimensions:
                                    %s """ % (dims,))
            return positions
        else:
            position = None
            for pos in self.points:
                if dims in self.points[pos]:
                    position = pos
                    break
            if position is None:
                raise Exception("""Dimension does not match any know grip point position: %s """ % (dims,))
            return position

    def _get_missmatch(self, da, pos):
        """
        Returns the axes-name where a missmatch between the variable and a position is found.
        The axes-names are hard coded to 'X', 'Y', or 'Z'. Has to match the name given for the xgcm metric!

        :param da:  data variable
        :param pos: position where it is compared to ('T','U','V','F','W','UW','VW','FW')

        :return:    List of axes-names of missmatches
        """
        # Get dimensions, if temporal dimension cut it from dims to get spatial dims only
        dims = da.dims
        if 't' in dims:
            ind = dims.index('t')
            dims = dims[:ind] + dims[ind + 1:]

        # Get the expected dimension with corresponding number of axes
        pos_opt = [len(l) for l in self.points[pos]].count(len(dims))
        if pos_opt==1:
            ax_num = [len(l) for l in self.points[pos]].index(len(dims))
            expect = self.points[pos][ax_num]
        else:
            for opt in range(len(self.points[pos])):
                find = [a[0] for a in dims]
                current = [a[0] for a in self.points[pos][opt]]
                if find==current:
                    expect = self.points[pos][opt]
                    break

        # Add physical dimensions where a missmatch is found
        missmatch = []
        for ax in range(len(expect)):
            
            if dims[ax] == expect[ax]:
                pass
            else:
                ax_miss = 'X' if dims[ax][0] == 'x' else \
                    'Y' if dims[ax][0] == 'y' else \
                        'Z' if dims[ax][0] == 'z' else None
                if ax_miss == None:
                    raise Exception("""Axis %s does not match to any known dimensions""" % (da.dims[ax]))
                missmatch.append(ax_miss)

        return missmatch

    def _shift_position(self, da, output_position, elements=None, bd='interp', bd_value=None, bd_scaling=1, maskargs={}, **kwargs):
        """
        Returns the variable interpolated to a prescribed output position.

        :param da:                  data variable or vector to shift
        :param output_position:     desired output position
        :param elements:            elements to include in the shift if a data vector is given, List(Boolean, ..)

        :return:                    interpolated data variable or vector onto the output_position
        """

        # If da is a list (vector), interpolate every element in da along the axes to match the output_position
        # If elements is given only indices where True is given are included. Default is True for every element
        if type(da) == list or type(da) == np.ndarray:
            if elements is None:
                elements = [True] * len(da)
            if type(output_position) is str:
                output_position = [output_position] * len(da)
            da_out = []
            for i in range(len(da)):
                if elements[i]:
                    element = da[i]
                    pos = output_position[i]
                    if self._matching_pos(element, pos):
                        da_out.append(element)
                    else:
                        missmatch = self._get_missmatch(element, pos)
                        if bd == 'interp':
                            da_out_i = self.interp(da[i].fillna(0), axis=missmatch, **kwargs)
                            da_out_i *= self._get_mask(da_out_i, maskargs=maskargs, **kwargs)
                        elif bd == 'fill':
                            da_out_i = self.interp(da[i], axis=missmatch, **kwargs)
                            if self._get_position(da[i])!='T':
                                shift_mask = self._get_shift_mask(da[i], fill_value=bd_value, scaling=bd_scaling)
                                da_out_i = da_out_i.copy(data=np.nan_to_num(da_out_i,0) + shift_mask.values)
                            da_out_i *= self.nan_mask(da_out_i)
                        else:
                            raise Exception('Unknown handling of boundary values. Please use "interp" or "fill".')
                        da_out.append(da_out_i)
                else:
                    da_out.append(da[i])

        # Else interpolate the data variable along the axes to match the output_position
        else:
            if self._matching_pos(da, output_position):
                da_out = da
            else:
                missmatch = self._get_missmatch(da, output_position)
                if bd == 'interp':
                    da_out = self.interp(da.fillna(0),axis=missmatch, **kwargs)
                    da_out *= self._get_mask(da_out, maskargs=maskargs, **kwargs)
                elif bd == 'fill':
                    da_out = self.interp(da, axis=missmatch, **kwargs)
                    if self._get_position(da)!='T':
                        shift_mask = self._get_shift_mask(da, fill_value=bd_value, scaling=bd_scaling)
                        da_out = da_out.copy(data=np.nan_to_num(da_out, 0) + shift_mask.values)
                    da_out *= self.nan_mask(da_out)
                else:
                    raise Exception('Unknown handling of boundary values. Please use "interp" or "fill".')
        return da_out

    def _matching_pos(self, da, pos, skip=None):
        """
        Checks if the given data is on the indicated position

        :param da:      data variable or vector to test
        :param pos:     position to test for

        :return:        True, False or List(True/False, ...)
        """

        if type(da) == list:
            if skip != None:
                da = self._data_skip(da, skip)
            a_pos = self._get_dims(da)

            match = [any([expect == element for expect in self.points[pos]]) for element in a_pos]
            if all(match):
                return True
            else:
                return match  # raise Exception("""False elements %s do not match position %s""" % (match, pos))
        else:
            a_pos = self._get_dims(da)
            if a_pos in self.points[pos]:
                return True
            else:
                return False  # raise Exception("""The variable does not match position %s""" %pos)

    def _matching_dim(self, da1, da2, skip1=None, skip2=None):
        """
        Checks if the dimension of data variable da1 matches the dimension of data variable da2.

        :param da:      data variable or vector to test
        :param pos:     position to test for

        :return:        True, False or List(True/False, ...)
        """
        # Get dimensions and cut 't' and skip if given
        if skip1 != None:
            da1 = self._data_skip(da1, skip1)
        if skip2 != None:
            da2 = self._data_skip(da2, skip2)

        da1_dims = self._get_dims(da1)
        da2_dims = self._get_dims(da2)
        if 't' in da1_dims:
            ind = da1_dims.index('t')
            da1_dims = da1_dims[:ind] + da1_dims[ind + 1:]
        if 't' in da2_dims:
            ind = da2_dims.index('t')
            da2_dims = da2_dims[:ind] + da2_dims[ind + 1:]

        # Depending on type, make comparison
        if type(da1) is list and type(da2) is xr.DataArray:
            match = [da2_dims == da1_pos for da1_pos in da1_dims]
            if all(match):
                return True
            else:
                return match  # raise Exception("""False elements %s do not match position %s""" % (match, pos))
        elif type(da2) is list and type(da1) is xr.DataArray:
            match = [da1_dims == da2_pos for da2_pos in da2_dims]
            if all(match):
                return True
            else:
                return match
        elif type(da1) is list and type(da2) is list:
            if len(da1) == len(da2):
                if da1_dims == da2_dims:
                    return True
                else:
                    return False
            else:
                raise Exception("""Data variable 1 and Data variable 2 do not have the same length""")
        elif type(da1) is xr.DataArray and type(da2) is xr.DataArray:
            if da1_dims == da2_dims:
                return True
            else:
                return False

    def _get_mask(self, da, maskargs={},**kwargs):
        """Return a mask based on the dataarray provided."""
        
        maskargs = {**self.maskargs, **maskargs}

        if maskargs['mask'] is None:
            mask = 1
        elif maskargs['mask'] == 'nan':
            mask = self.nan_mask(da)
        elif maskargs['mask'] == 'zero':
            mask = self.zero_mask(da)
        elif maskargs['mask'] == 'boundary':
            if 'bd' in maskargs:
                mask = self.boundary_mask(da, maskargs['bd'])
            else:
                raise Exception('Please provide bd = [ax, index] as kwarg if boundary-mask is used!')
        elif maskargs['mask'] == 'usr_def':
            if 'mask_values' in maskargs:
                mask = maskargs['mask_values']
            else:
                raise Exception('Please provide the mask in the kwarg mask_values if usr_def is used!')
        elif maskargs['mask'] == 'mld':
            if 'mld' in maskargs and 'invert' in maskargs:
                mask = self.mld_mask(da, maskargs['mld'], invert=maskargs['invert'],**kwargs)
            elif 'mld' in maskargs:
                mask = self.mld_mask(da, maskargs['mld'],**kwargs)
            else:
                raise Exception('Please provide the mld and invert in maskargs dictionary!')
        elif maskargs['mask'] == 'sign':
            if 'invert' in maskargs:
                mask = self.sign_mask(da, invert=maskargs['invert'],**kwargs)
            else:
                mask = self.sign_mask(da, **kwargs)
        else:
            raise Exception('Unknown mask type')

        return mask

    def boundary_mask(self, da, bd):
        """
        Return a mask of the data array where the boundaries along an axis are set to nan.
        """
        dims = self.grid._get_dims_from_axis(da, bd.keys())
        mask_axes = {ax:np.ones(da.sizes[ax]) for ax in dims}
        if len(dims) > 1:
            for ax in enumerate(bd.keys()):
                for i in bd[ax[1]]:
                    mask_axes[dims[ax[0]]][i] = np.nan
                    
            return xr.DataArray(np.prod(np.meshgrid(*mask_axes.values(), indexing='ij'), axis=0),
                                dims=dims,
                                coords=dict(zip(dims,[da.coords[d] for d in dims])))
        
        else:
            for i in list(bd.values())[0]:
                mask_axes[dims[0]][i]= np.nan
                
            return xr.DataArray(mask_axes[dims[0]],
                                dims=dims,
                                coords=dict(zip(dims,[da.coords[d] for d in dims])))


    def zero_mask(self, da):
        """
        Create a mask of a data variable with 0 where the data is 0 or nan and 1 everywhere else.
        """
        da_one = da * 0 + 1
        da_one = da_one.where(np.isfinite(da), other=0)
        da_one = da_one.where(da != 0, other=0)
        mask = xr.DataArray(da_one.values, coords={dim: da[dim].values for dim in da.dims}, dims=da.dims)
        return mask

    def nan_mask(self, da):
        """
        Create a mask of a data variable with nan where the data is 0 and 1 everywhere else.
        """
        da_one = da * 0 + 1
        da_one = da_one.where(da != 0)
        mask = xr.DataArray(da_one.values, coords={dim: da[dim].values for dim in da.dims}, dims=da.dims)
        return mask

    def mld_mask(self, da, mld, invert=False, **kwargs):
        """

        :param da: Dataarray to get metrics from
        :param mld: Mixed-layer-depth values
        :param invert: If True, the mixed layer is masked to return below mld values only

        :return: nan-mask for values below the mixed layer depth
        """
        mask = da.copy(data=da.values * 0 + 1).rename('mld_mask')
        dim_z = self.grid._get_dims_from_axis(mask,'Z')[0]
        depth = self.get_depth_from_metric(da, **kwargs)
        if not invert:
            if np.all([d in mld.dims for d in da.dims if d!=dim_z]):
                mask = mask.where((depth<mld).transpose(*mask.dims))
            else:
                dim_order = []
                if 't' in mask.dims:
                    dim_order.append('t')
                    t=1
                else:
                    t=0
                for i in range(t,len(mask.dims)):
                    if mask.dims[i] in mld.dims or mask.dims[i]==dim_z:
                        dim_order.append(mask.dims[i])
                mask = mask.where((depth < mld).transpose(*tuple(dim_order)))
        elif invert:
            if np.all([d in mld.dims for d in da.dims if d!=dim_z]):
                mask = mask.where((depth>mld).transpose(*mask.dims))
            else:
                dim_order = []
                if 't' in mask.dims:
                    dim_order.append('t')
                    t=1
                else:
                    t=0
                for i in range(t,len(mask.dims)):
                    if mask.dims[i] in mld.dims or mask.dims[i]==dim_z:
                        dim_order.append(mask.dims[i])
                mask = mask.where((depth > mld).transpose(*tuple(dim_order)))
        return mask

    def sign_mask(self, da, invert=False, **kwargs):
        """

        :param da: Dataarray to be applied on
        :param invert: If True, the mixed layer is masked to return below mld values only

        :return: nan-mask for values greater/smaller zero
        """
        mask = da.copy(data=da.values * 0 + 1).rename('sign_mask')

        if not invert:
            mask = mask.where(da > 0)

        elif invert:
            mask = mask.where(da < 0)
        return mask
    
    def eq_mask(self, da, eq_scale=0.5):
        """Return mask where the equator is scaled with a specified value. Standard is 0.5"""
        dims = self._get_dims(da)
        if 'y_f' in dims:
            print('Data is on V-point, no mask will be applied')
            return 1
        elif 'y_c' not in dims:
            print('Data does not have a proper y-dimension')
            return 1
        else:
            mask_axes = {ax: np.ones(da.sizes[ax]) for ax in dims}
            mask_axes['y_c'][1] *= eq_scale
            return xr.DataArray(np.prod(np.meshgrid(*mask_axes.values(), indexing='ij'), axis=0),
                                dims=dims,
                                coords=dict(zip(dims,[da.coords[d] for d in dims])))

    def get_depth_from_metric(self, da, surface=None, **kwargs):
        """Return depth calculated from the grid metrics"""
        
        kwargs = {**self.boundary, **kwargs}
        pos = self._get_position(da)
        
        if pos == 'T':
            try:
                mw=self._get_metric_by_pos('Z','W')[0]
            except:
                raise Exception("W-Point metric is missing, required to calculate depth.")
            
            depth = self.grid.cumsum(mw,'Z',**kwargs)
            if surface:
                depth -= surface
            else:
                mt = self._get_metric_by_pos('Z','T')[0]
                if 't' in mt.dims:
                    depth -= 0.5*mt.mean('t').isel({self.grid._get_dims_from_axis(mt,'Z')[0]:0})
                else:
                    depth -= 0.5*mt.isel({self.grid._get_dims_from_axis(mt,'Z')[0]:0})
                    
                    
        elif pos == 'W':
            try:
                mt=self._get_metric_by_pos('Z','T')[0]
            except:
                raise Exception("T-Point metric is missing, required to calculate depth.")
            
            depth = self.grid.cumsum(mt,'Z',**kwargs)
            if surface:
                depth -= surface
            """
            else:
                mw = self._get_metric_by_pos('Z','W')[0]
                if 't' in mt.dims:
                    depth += 0.5*mw.mean('t').isel({self.grid._get_dims_from_axis(mw,'Z')[0]:0})
                else:
                    depth += 0.5*mw.isel({self.grid._get_dims_from_axis(mw,'Z')[0]:0})
            """        
        return depth
        
    def remap_numpy(self, da, depth_fr, depth_to):
        # Check dimensions and extract dimensions of the data with separation of the Z coordinate
        if self._matching_dim(da, depth_fr):
            pass
        dim_cut = list(da.dims)
        dim_z = self.grid._get_dims_from_axis(da, 'Z')
        dim_cut.remove(dim_z[0])

        # Create new data with dimensions of the old but the depth dimension replaced with the output depth
        # Iterate through elements and interpolate to new depth with numpy.interp
        da_out = np.zeros([da[dim_cut[i]].shape[0] for i in range(len(dim_cut))] + [len(depth_to)])
        if len(dim_cut) == 3:
            for i in range(da[dim_cut[0]].shape[0]):
                for j in range(da[dim_cut[1]].shape[0]):
                    for k in range(da[dim_cut[2]].shape[0]):
                        da_out[i, j, k] = np.interp(depth_to,
                                                    depth_fr.isel({dim_cut[0]: i, dim_cut[1]: j, dim_cut[2]: k}).values,
                                                    da.isel({dim_cut[0]: i, dim_cut[1]: j, dim_cut[2]: k}).values,
                                                    )#left=np.nan, right=np.nan)
        if len(dim_cut) == 2:
            for i in range(da[dim_cut[0]].shape[0]):
                for j in range(da[dim_cut[1]].shape[0]):
                    da_out[i, j] = np.interp(depth_to,
                                             depth_fr.isel({dim_cut[0]: i, dim_cut[1]: j}).values,
                                             da.isel({dim_cut[0]: i, dim_cut[1]: j}).values,
                                             )#left=np.nan, right=np.nan)
        if len(dim_cut) == 1:
            for i in range(da[dim_cut[0]].shape[0]):
                da_out[i] = np.interp(depth_to,
                                      depth_fr.isel({dim_cut[0]: i}).values,
                                      da.isel({dim_cut[0]: i}).values,
                                      )#left=np.nan, right=np.nan)

        # Create output DataArray with same dimensions as input but exchanged depth coordinate
        #if dim_z[0] == 'z_c':
        #    z = np.arange(len(depth_to))
        #elif dim_z[0] == 'z_f':
        #    z = np.arange(len(depth_to)) - 0.5
        dim_z[0]=str(depth_to.dims[0])
        da_out = xr.DataArray(da_out, coords=dict({dim_cut[i]: da[dim_cut[i]].values for i in range(len(dim_cut))},
                                                  **{dim_z[0]: depth_to.values}),
                              dims=dim_cut + dim_z)
        dims_out=list(da.dims)
        dims_out[1]=dim_z[0]
        da_out = da_out.transpose(*dims_out)

        return da_out

    def derivative(self, da, axis, **kwargs):
        """
        Compute the derivative along a given axis.

        Uses the xgcm derivative function
        """

        return self.grid.derivative(da, axis, **dict(self.boundary, **kwargs))

    def interp(self, da, axis, **kwargs):
        """
        Interpolation function based on the xgcm interpolation.
        Can be performed for up to three axes.

        :param da:      data variable to interpolate
        :param axis:    axis or List(axes) along which to interpolate, computed by given order
        :param kwargs:  additional keyword arguments for xgcm.interp

        :return:        interpolated data variable along given axis
        """
        # Merge kwargs
        kwargs = dict(self.boundary, **kwargs)
        if self.discretization == 'standard':
            if axis is None:
                da_int = self.grid.interp(da, )
            elif len(axis) == 1:
                da_int = self.grid.interp(da, axis=axis[0], **kwargs)
            elif len(axis) == 2:
                da_int = self.grid.interp(self.grid.interp(da, axis=axis[0], **kwargs),
                                          axis=axis[1], **kwargs)
            elif len(axis) == 3:
                da_int = self.grid.interp(self.grid.interp(self.grid.interp(da,
                                                                            axis=axis[0], **kwargs),
                                                           axis=axis[1], **kwargs),
                                          axis=axis[2], **kwargs)

            else:
                raise Exception("Unknown operation for axis: %s" % (axis))

        return da_int

    def dot(self, x, y):
        """
        Performs a dot product for two vector fields.

        x and y are vector fields given as a list of scalar fields. Matching dimensions are required.

        :param x:   vector field
        :param y:   vector filed

        :return:
        """
        dims = len(x) if len(x) == len(y) else False
        if dims == False:
            raise Exception("Vector dimensions do not match: x: %s != y:%s" % (len(x), len(y)))

        c = 0
        for i in range(dims):
            c += x[i] * y[i]

        return c

    def cross(self, x, y):
        """
        Performs a cross product for two 3d vector fields.

        x and y are vector fields as a type 'list' or 'np.ndarray" of scalar fields. Matching dimensions are required.

        :param a:
        :param b:
        :return:
        """

        dims = len(x) if len(x) == len(y) else False
        if dims == False:
            raise Exception("Vectors dimensions do not match: x: %s != y:%s" % (len(x), len(y)))

        if dims != 3:
            raise Exception("Operation is only implemented for 3D vectors, but dimensions are: %s" % dims)

        c = [0] * 3
        c[0] = x[1] * y[2] - x[2] * y[1]
        c[1] = x[2] * y[0] - x[0] * y[2]
        c[2] = x[0] * y[1] - x[1] * y[0]

        if type(x) == np.ndarray and type(y) == np.ndarray:
            c = np.ndarray(c)
        return c

    def divergence(self, x, axes=None, discretization=None, **kwargs):
        """
        Compute the divergence of a vector field.

        x is a scalar or a vector fields as a type 'list'.

        """
        if discretization is None:
            discretization = self.discretization

        if axes is None:
            axes = ['X', 'Y', 'Z']
        elif type(axes) is list:
            pass
        elif type(axes) is str or type(axes) is tuple:
            axes = list(axes)
        else:
            raise Exception("Input for 'axes' not understood, please provide a list, tuple, or string")

        if type(x) is not list:
            raise Exception("Input is not of type 'list', please provide a vector as type 'list'.")
        if axes == ['X', 'Y', 'Z'] and len(x) != 3:
            raise Exception("Vector is not 3dimensional, please provide axes to compute the gradient")

        # ========
        # Standard
        # ========
        if discretization == 'standard':
            y = sum([self.derivative(x[i], axes[i], **kwargs) for i in range(len(axes))])

        # ====
        # Nemo
        # ====
        elif discretization == 'nemo':
            weight_in = [1] * len(axes)
            x_d = [0] * len(axes)
            weight_out = [1] * len(axes)
            for i in range(len(axes)):
                # Get metrics from input position and multiply it to the data inside the difference
                if axes[i] == 'Z':
                    pass
                else:
                    weight_in[i] = self._combine_metric(x[i], axes, skip=axes[i])
                # Compute derivative
                x_d[i] = self.derivative(x[i] * weight_in[i], axes[i], **kwargs)
                # Get metrics for output position and divide the the derivative by it
                if axes[i] == 'Z':
                    pass
                else:
                    weight_out[i] = self._combine_metric(x_d[i], axes, skip=axes[i])

            y = sum([x_d[i] / weight_out[i] for i in range(len(axes))])

        return y

    def gradient(self, da, axes=None, **kwargs):
        """
        Compute the gradient of a data variable .

        If axes are not specified,  by default the gradient is taken along 'X', 'Y' and 'Z'.
        """
        if axes is None:
            axes = ['X', 'Y', 'Z']
        return [self.derivative(da, ax, **kwargs) for ax in axes]

    def curl(self, x, axes=None, **kwargs):
        """
        Compute the curl of a vector.

        If axes are not specified, by default the gradient is taken along 'X', 'Y' and 'Z'.
        """
        if len(x) != 3:
            raise Exception("Vector is not 3dimensional, please provide axes to compute the gradient")

        if axes is None:
            axes = ['X', 'Y', 'Z']
        elif len(axes) != 3:
            raise Exception("Input for axes is not 3 dimensional, please provide the appropriate axis order.")

        y = [0] * 3
        y[0] = self.derivative(x[2], axes[1], **kwargs) - self.derivative(x[1], axes[2], **kwargs)
        y[1] = self.derivative(x[0], axes[2], **kwargs) - self.derivative(x[2], axes[0], **kwargs)
        y[2] = self.derivative(x[1], axes[0], **kwargs) - self.derivative(x[0], axes[1], **kwargs)

        return y

    def average(self, P, axes, boundary=None, Vmask=None, **kwargs):
        """
        Compute the weighted average of a variable along the specified axes.
        The weights are taken from the metric provided through the grid object.

        P_average = Sum( P * m ) / Sum( m )
        With P the variable and m the respective weight
        """
        
        skip=0
        if 't' in P.dims:
            skip=1
        
        if len(P.dims[skip:]) in [1,0]:
            case = 'smallest'
        else:# len(P.dims[skip:])==3:
            case = 'largest'
            
        # Get metric from xgcm grid
        pos = self._get_position(P)
        dims = self.grid._get_dims_from_axis(P, axes)
        if P.dims != self._get_dims_order(P):
            P = P.transpose(*self._get_dims_order(P))
        if boundary is None:
            m = self._get_metric_by_pos(axes, pos ,combine=True, case=case)[0]
        else:
            for ax in axes:
                if ax not in boundary:
                    boundary[ax] = [None,None]
            m = self._get_metric_by_pos(axes, pos ,combine=True)[0].sel(
                {dims[i]: slice(*boundary[axes[i]]) for i in range(len(axes))})
            P = P.sel({dims[i]: slice(*boundary[axes[i]]) for i in range(len(axes))})

        # Get and apply mask
        mask = self._get_mask(P,**kwargs)
        # P = P * mask
        if Vmask is None:
            m = m * mask
        else:
            m = m * Vmask
        # Average along specified axes
        #dims = self.grid._get_dims_from_axis(P, axes)
        P_av = (P * m).sum(dims) / m.sum(dims)

        return P_av
