import xarray as xr
from os import listdir
import os
import os.path
import numpy as np


def dataset_domainmerge(path, file, coord_names, coord_vars, coord_level, dims_new, dims_old):
    run = True
    outputfile = file + "_merge.nc"
    if os.path.isfile(path + outputfile):
        print("Outputfile already exists")
        if input("Overwrite file? y or n:"):
            run = True
        else:
            run = False

    if run:
        dataset_full_dim_coord_update(path, coord_names, coord_vars, coord_level, dims_new, dims_old, name_init=file,
                                      specific_name=outputfile)
        """files= [file+"{}.nc".format(i) for i in range(8)]

        for f in files:
            ds = xr.open_dataset(path + f)
            dims = True
            for d in dims_new:
                if d not in list(dict(ds.dims).keys()):
                    dims = False

            if not dims:
                ds[i] = dataset_dim_convert(ds[i], dims_new, dims_old)

            ds[i] = dataset_coords_assign(ds[i], coord_names, coord_vars, coord_level)"""

        ds = [xr.open_dataset(path + file + "{}.nc".format(i)) for i in range(8)]
        ds = xr.combine_by_coords(ds)
        os.remove(path+outputfile)
        ds.to_netcdf(path=path+outputfile,mode="w")

    return path + file + "_merge.nc"


def dataset_coords_assign(dataset, coord_names, coord_vars, coord_level):
    if type(coord_level) == int:
        coord_names, coord_vars, coord_level = [coord_names], [coord_vars], [coord_level]
    for c in range(len(coord_names)):
        cn = coord_names[c]

        if cn not in list(dataset.coords):
            if coord_level[c] is None:
                dataset = dataset.assign_coords({cn: dataset[coord_vars[c]].values})
            elif coord_level[c] == 0:
                dataset = dataset.assign_coords({cn: dataset[coord_vars[c]].values[0]})
            elif coord_level[c] == 1:
                dataset = dataset.assign_coords({cn: dataset[coord_vars[c]].values[:, 0]})
            elif coord_level[c] == 2:
                dataset = dataset.assign_coords({cn: dataset[coord_vars[c]].values[:, :, 0]})

    return dataset


def dataset_dim_convert(dataset, dims_new, dims_old):
    """if dims_old is None:
        dims = list(dict(dataset.dims).keys())
        initials = [d[0] for d in dims_new]
        dims_old = list()
        for d in dims:
            if d[0] in initials:
                dims_old.append(d)"""
    dims_n = {}
    for d in range(len(dims_old)):
        dims_n[dims_old[d]] = dims_new[d]

    return dataset.rename_dims(dims_n)

def dataset_dimshift(path,
                     dims_values,
                     dims_new,
                     name_init=None,
                     specific_name=None):

    files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

    for f in files.copy():
        rm = False
        if f[-2:] != "nc":
            files.remove(f)
            rm = True
        if name_init is not None and not rm:
            if f[:len(name_init)] != name_init:
                files.remove(f)
                rm = True
        if specific_name is not None and not rm:
            if type(specific_name) == list or type(specific_name) == tuple:
                for s in specific_name:
                    if f == s:
                        files.remove(f)
                        break
            else:
                if f == specific_name:
                    files.remove(f)

    for f in files:
        ds = xr.open_dataset(path + f)
        vars = ds.variables
        dims = ds.dims

        for d in range(len(dims_new)):
            if dims_new[d] not in dims:
                ds = ds.assign_coords({dims_new[d]: dims_values[d]})
            else:
                pass

        suffix = [('u', 'u_0'), ('v', 'v_0'), ('w', 'w_0') , ('w_', 'w_1d') , ('f', 'f_0'), ('uw', 'uw_0'), ('vw', 'vw_0')]
        dims_old = ['x_c','y_c','z_c','z_c', ['x_c','y_c'], ['x_c','z_c'], ['y_c', 'z_c']]
        dims_shift = ['x_f','y_f','z_f', 'z_f', [ 'x_f','y_f'], ['x_f','z_f'], ['y_f', 'z_f']]
        track=0
        for v in vars:
            ind=[None,None]
            for i in [3,5,6]:
                if v[-4:] == suffix[i][1]:
                    ind[0]=i
                    ind[1]=1
                    break
                elif v[-2:] == suffix[i][0]:
                    ind[0]=i
                    ind[1]=0
                    break
            if ind[0] is None:
                for i in [0,1,2,3]:
                    if v[-3:] == suffix[i][1]:
                        ind[0]=i
                        ind[1]=1
                        break

                    elif v[-1] == suffix[i][0]:
                        ind[0]=i
                        ind[1]=0
                        break


            if ind[0] is not None:
                dims = list(ds[v].dims)
                if type(dims_old[ind[0]]) is str:
                    if dims_shift[ind[0]] in dims:
                        continue
                    else:
                        i = np.where(np.array(ds[v].dims) == dims_old[ind[0]])[0][0]

                        dims[i] = dims_shift[ind[0]]
                        ds[v] = xr.DataArray(data=ds[v].values, dims=dims, attrs=ds[v].attrs)
                        track += 1
                else:
                    if np.all(np.array([d in dims for d in dims_shift[ind[0]]])):
                        continue
                    else:
                        dnew=dims.copy()
                        for j in range(len(dims_old[ind[0]])):

                            if dims_old[ind[0]][j] in dims:
                                i = np.where(np.array(dims) == dims_old[ind[0]][j])[0][0]
                                dnew[i] = dims_shift[ind[0]][j]
                        ds[v] = xr.DataArray(data=ds[v].values, dims=dnew, attrs=ds[v].attrs)
                        track += 1

        print("{} variables changed dimensions".format(track))
        os.remove(path + f)
        ds.to_netcdf(path=path + f, mode="w")


def dataset_rename(dataset, coords_new, coords_old):
    coords = dict(zip(coords_old, coords_new))

    return dataset.rename(coords)


def dataset_full_dim_coord_update(path,
                                  coords_new,
                                  coords_old,
                                  coords_level,
                                  dims_new,
                                  dims_old,
                                  name_init=None,
                                  specific_name=None):
    files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in files.copy():
        rm = False
        if f[-2:] != "nc":
            files.remove(f)
            rm = True
        if name_init is not None and not rm:
            if f[:len(name_init)] != name_init:
                files.remove(f)
                rm = True
        if specific_name is not None and not rm:
            if type(specific_name) == list or type(specific_name) == tuple:
                for s in specific_name:
                    if f == s:
                        files.remove(f)
                        break
            else:
                if f == specific_name:
                    files.remove(f)


    for f in files:
        print(f)
        ds = xr.open_dataset(path + f)#, decode_times=False)
        vars = ds.variables
        dims = ds.dims
        coords = list(ds.coords)
        ci = list()
        track=[0,0,0]
        di = None
        dd = list()

        for d in dims:
            if d in dims_old:
                di = np.where(d == np.array(dims_old))[0][0]

                if dims_new[di] not in dims:
                    ds = ds.rename({d: dims_new[di]})
                    track[0]+=1
                    dims = ds.dims
                elif d in coords:
                    dd.append(d)
                else:
                    ds = ds.rename({d: dims_new[di]})
                    track[0]+=1
            """elif d in dims_old and d in vars:
                di = np.where(d == np.array(dims_old))[0][0]
                if dims_new[di] in dims:
                    ds = ds.drop(np.array(dims_old)[di])
                    pass
                else:
                    ds = ds.drop(np.array(dims_old)[di])
                    if d in ds.dims:
                        ds = ds.rename({d:dims_new[di]})"""

            """ds = ds.drop(np.array(dims_old)[dd])
            ds = dataset_rename(ds, np.array(dims_new)[di], np.array(dims_old)[di])
"""
        for c in range(len(coords_new)):
            if coords_new[c] not in coords:
                if coords_old[c] not in vars:
                    pass
                else:
                    ds = dataset_coords_assign(ds, np.array(coords_new)[c], np.array(coords_old)[c],
                                               np.array(coords_level)[c])
                    track[1]+=1
        """for c in ci:
            if coord_names[c] in coords:
                ci.remove(c)
                cd.append(c)"""

        cd = list(set.intersection(set(list(ds.coords)), set(coords_old)))
        for dv in set(cd + dd):
            ds = ds.drop(dv)
            track[2]+=1
        if track[0]==0 and track[1]==0:
            pass
        else:
            print("Updated {} dimensions and {} coordinates. {} variables were removed.".format(*track))
            os.remove(path + f)
            ds.to_netcdf(path=path + f, mode="w")

    return track[0]

def dataset_time_shift(path,time_var ,time_shift, name_init=None, specific_name=None):

    files = [f for f in listdir(path) if os.path.isfile(os.path.join(path, f))]

    for f in files.copy():
        rm = False
        if f[-2:] != "nc":
            files.remove(f)
            rm = True
        if name_init is not None and not rm:
            if f[:len(name_init)] != name_init:
                files.remove(f)
                rm = True
        if specific_name is not None and not rm:
            if type(specific_name) == list or type(specific_name) == tuple:
                for s in specific_name:
                    if f == s:
                        files.remove(f)
                        break
            else:
                if f == specific_name:
                    files.remove(f)

    for f in files:
        ds = xr.open_dataset(path + f , decode_times=False)
        if 'time_shift' in ds.t.attrs:
            if ds.t.attrs['time_shift'] != time_shift:
                attrs=ds.t.attrs
                ds[time_var] = ds[time_var] + time_shift - ds.t.attrs['time_shift']
                ds[time_var] = ds.t.assign_attrs({**attrs,'time_shift':time_shift})
                os.remove(path + f)
                ds.to_netcdf(path=path + f, mode="w")
                ds.close()
                print(f, 'time_shift updated')
        else:
            attrs = ds.t.attrs
            ds[time_var] = ds[time_var] + time_shift
            ds[time_var] = ds.t.assign_attrs({**attrs,'time_shift':time_shift})
            os.remove(path + f)
            ds.to_netcdf(path=path + f, mode="w")
            ds.close()
            print(f, 'time_shift updated')