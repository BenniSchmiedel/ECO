import os
import sys
import debugpy
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)

import diagnostics
from utils import str_or_none, config_parser, save_by_chunks, get_namelist, read_chunks, open_datasets

import numpy as np
import xgcm
import xarray as xr

def postprocess():
    print('Run postprocessing')
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    _metrics = {('X',): ['e1tm', 'e1um', 'e1vm', 'e1fm'],
        ('Y',): ['e2tm', 'e2um', 'e2vm', 'e2fm'],
        ('Z',): ['e3tm', 'e3um', 'e3vm', 'e3wm']}

    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = diagnostics.Grid_ops(grid, maskargs={'mask':'nan'})
    grid = grid_ops._update({'Z':ds['e3tm_1d']})

    p=diagnostics.Properties(grid_ops,{'X': ds.glamt, 'Y':ds.gphit,'Z': ds.depth_1d}, eos_properties=kwargs_sim['eos'])
    e=diagnostics.Energetics(grid_ops, p)

    print("h and dh")
    dh_T = p.dh_T(ds.to,ds.so,ds.depth,Z_r=ds.zg_0).compute()
    dh_S = p.dh_S(ds.to,ds.so,ds.depth,Z_r=ds.zg_0).compute()
    dhTdz = grid.derivative(dh_T,'Z',boundary='fill',fill_value=0).compute()
    dhSdz = grid.derivative(dh_S,'Z',boundary='fill',fill_value=0).compute()

    print("convection mask")
    ds_proc['mask_avt']= (grid_ops.nan_mask(ds.k_evd)*ds.k_evd/100).compute()
    ds_proc['mask_avt_invert']=((ds.mask_bd_w)*((-(ds_proc['mask_avt']-1)).fillna(1))).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', **kwargs_proc) 
    #ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Integration by hand to obtain downward flux")
    FT_bot_to_top = np.nan*np.ones(ds.e3wm.shape)
    FS_bot_to_top = np.nan*np.ones(ds.e3wm.shape)
    FT_surfaceflux = np.ones(ds.e3wm[:,0].shape)
    FS_surfaceflux = np.ones(ds.e3wm[:,0].shape)

    I = ds.z_c.size
    QT = (ds.ttrd_zdf*ds.e3tm).values
    QS = (ds.strd_zdf*ds.e3tm).values

    for i in range(1, I):
        if i==1:
            FT_bot_to_top[:,-1-i] = QT[:,-1-i]
            FS_bot_to_top[:,-1-i] = QS[:,-1-i]
        elif i==I-1:
            FT_bot_to_top[:,-1-i] = QT[:,0]*0# + QT[:,-1-i])
            FS_bot_to_top[:,-1-i] = QS[:,0]*0# + QS[:,-1-i])
            FT_surfaceflux = (FT_bot_to_top[:,1] + QT[:,0]) + (ds.ttrd_qns.values+ds.ttrd_qsr[:,0].values)*ds.e3tm[:,0].values
            FS_surfaceflux = (FS_bot_to_top[:,1] + QS[:,0]) + ds.strd_cdt.values*ds.e3tm[:,0].values
        else:
            FT_bot_to_top[:,-1-i] = (FT_bot_to_top[:,-i] + QT[:,-1-i])
            FS_bot_to_top[:,-1-i] = (FS_bot_to_top[:,-i] + QS[:,-1-i])
    FT_bot_to_top = xr.DataArray(FT_bot_to_top, coords=ds.e3wm.coords)
    FS_bot_to_top = xr.DataArray(FS_bot_to_top, coords=ds.e3wm.coords)
    FT_surfaceflux = xr.DataArray(FT_surfaceflux, coords=ds.e3wm[:,0].coords)
    FS_surfaceflux = xr.DataArray(FS_surfaceflux, coords=ds.e3wm[:,0].coords)

    print("KN2 conv/diff")
    KN2_h_bot_to_top_convective = - (dhTdz*FT_bot_to_top*ds_proc['mask_avt'].fillna(0) 
                                     +  dhSdz*FS_bot_to_top*ds_proc['mask_avt'].fillna(0)).compute() # Not shifted due to nan values
    KN2_h_bot_to_top_diffusive = - grid_ops._shift_position(dhTdz*FT_bot_to_top*ds_proc['mask_avt_invert'].fillna(0) 
                                    + dhSdz*FS_bot_to_top*ds_proc['mask_avt_invert'].fillna(0),'T').compute() 

    print("Gh fluxes")                    
    V_full = (ds.e3tm*ds.e2tm*ds.e1tm).sum(['z_c','y_c','x_c']).compute()
    GhT_surfaceflux = 1/V_full*(dh_T[:,0]*FT_surfaceflux*ds.e2tm*ds.e1tm).sum(['y_c','x_c']).compute()
    GhS_surfaceflux = 1/V_full*(dh_S[:,0]*FS_surfaceflux*ds.e2tm*ds.e1tm).sum(['y_c','x_c']).compute()
    
    GhT_qsr = dh_T*ds.ttrd_qsr
    GhT_qsr[:,0] *= 0

    GhT_hdiffusion = (dh_T*ds.ttrd_ldf).compute()
    GhS_hdiffusion = (dh_S*ds.strd_ldf).compute()

    maskT=ds.mask_bd_t
    maskW=ds.mask_bd_w
    print("Convective")  
    ds_proc["KN2_convective"] = -1/(9.81)*KN2_h_bot_to_top_convective.rolling({'t':2}).mean('t').compute()
    ds_proc["KN2_convective_yz"] = grid_ops.average(ds_proc["KN2_convective"],'X').compute()
    ds_proc["KN2_convective_gm"] = p.global_mean(ds_proc["KN2_convective"], Vmask=maskW).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Diffusive")  
    ds_proc["KN2_diffusive"] = -1/(9.81)*KN2_h_bot_to_top_diffusive.rolling({'t':2}).mean('t').compute()
    ds_proc["KN2_diffusive_yz"] = grid_ops.average(ds_proc["KN2_diffusive"],'X').compute()
    ds_proc["KN2_diffusive_gm"] = p.global_mean(ds_proc["KN2_diffusive"], Vmask=maskT).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Gh_surf")  
    ds_proc["Gh_surf"] = 1/(9.81)*(GhT_surfaceflux+GhS_surfaceflux).rolling({'t':2}).mean('t').compute()
    ds_proc["Gh_surf_yz"] = ds_proc["Gh_surf"]
    ds_proc["Gh_surf_gm"] = ds_proc["Gh_surf"]
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Gh_hdiff")  
    ds_proc["Gh_hdiff"] = 1/(9.81)*(GhT_hdiffusion+GhS_hdiffusion).rolling({'t':2}).mean('t').compute()
    ds_proc["Gh_hdiff_yz"] = grid_ops.average(ds_proc["Gh_hdiff"],'X').compute()
    ds_proc["Gh_hdiff_gm"] = p.global_mean(ds_proc["Gh_hdiff"], Vmask=maskT).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Gh_qsr")  
    ds_proc["Gh_qsr"] = 1/(9.81)*(GhT_qsr).rolling({'t':2}).mean('t').compute()
    ds_proc["Gh_qsr_yz"] = grid_ops.average(ds_proc["Gh_qsr"],'X').compute()
    ds_proc["Gh_qsr_gm"] = p.global_mean(ds_proc["Gh_qsr"], Vmask=maskT).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print("Conversion")  
    ds_proc["C"]= -1/(9.81*1026)*ds.ketrd_convP2K.rolling({'t':2}).mean('t').compute()
    ds_proc["C_yz"]= grid_ops.average(ds_proc["C"],'X').compute()
    ds_proc["C_gm"]= p.global_mean(ds_proc["C"], Vmask=maskT).compute()
    save_by_chunks(ds_proc, path_prefix=kwargs_proc['path_proc'], sub_prefix='postprocessed/postprocessed', mode='a', **kwargs_proc) 

    
if __name__ == '__main__':
    import argparse
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    from dask.distributed.client import  _get_global_client
    
    # Parse arguments from script execution
    parser = argparse.ArgumentParser(description='ECO Processing script.')
    parser.add_argument('-exp_family','-f','--f', type=str_or_none, default=None)
    parser.add_argument('-exp_name','-e','--e', type=str_or_none, default=None)
    #parser.add_argument('-sub_config','-c','--c', type=str_or_none, default=None)
    #parser.add_argument('-config_path','-p','--p', default='')
    parser.add_argument('-d','--d', type=bool, default=False)
    parser.add_argument('-mode','-m','--m', type=str_or_none, default=None)
    args = parser.parse_args()
    exp_family, exp_name, debug_mode, mode = args.f, args.e, args.d, args.m

    # Load configs
    kwargs_proc, kwargs_pre, kwargs_sim = config_parser(exp_family, exp_suffix=exp_name, log=True)
    if kwargs_sim['get_namelist']: kwargs_sim['namelist'] = get_namelist(exppath = kwargs_proc['path_nemo'])

    # Initialize debug-mode
    if debug_mode:
        print('Run Debug-Mode')
        debugpy.listen(57002)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()

    # Initialize dask cluster and client
    os.environ["MALLOC_TRIM_THRESHOLD_"] = '0'
    cluster = SLURMCluster(cores=kwargs_proc['n_cores'],
                            processes=kwargs_proc['n_processes'],
                            memory=kwargs_proc['memory'],
                            account=kwargs_proc['account'],
                            n_workers=kwargs_proc['n_workers'],
                            death_timeout= 300,
                            local_directory=kwargs_proc['local_directory'])
    if kwargs_proc['n_jobs']>1:
        cluster.scale(jobs=kwargs_proc['n_jobs'])
    client = _get_global_client() or Client(cluster)
    print('Dask setup finished')

    # Run processing
    startTime = datetime.now()
    # chunk_steps = round((kwargs_proc['time_stop']-kwargs_proc['time_start'])/kwargs_proc['n_chunks'])
    # for n_t in range(kwargs_proc['n_chunks']):
    #     if n_t == kwargs_proc['n_chunks']-1:
    #         t_slice = (kwargs_proc['time_start']+chunk_steps * n_t, kwargs_proc['time_stop'])
    #     else:
    #         t_slice = (kwargs_proc['time_start']+chunk_steps * n_t, kwargs_proc['time_start']+chunk_steps * (n_t+1))
    #     print('Open dataset timeslice {}-{}'.format(*t_slice))
    ds = open_datasets(kwargs_proc['exp_out'], kwargs_proc['exp_out_suffix'], components = ['all'])#, timestep_slice=t_slice)#, compat='override', coords='minimal')
    # ds = read_chunks(path_prefix=kwargs_proc['path_pre'], sub_prefix=kwargs_proc['exp_out'], sub_suffix=kwargs_proc['exp_out_suffix'], **kwargs_proc)
    #print('Postprocess dataset timeslice {}-{}'.format(*t_slice))
    postprocess()
    print('Postprocessing finished. Runtime: ', datetime.now() - startTime)