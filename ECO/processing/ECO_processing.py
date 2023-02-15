import os
import sys
import debugpy
from datetime import datetime
from pathlib import Path 

DATA_PROCESSED = Path('data/processed')
DATA_PREPROCESSED = Path('data/preprocessed')
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, parent_dir)

import diagnostics
from utils import drop_from_dict, split_by_chunks, str_or_none, get_number_of_chunks, existing_files_handler, config_parser
from xnemogcm import open_nemo_and_domain_cfg, get_metrics

import numpy as np
import xgcm
import xarray as xr
#import warnings
#warnings.filterwarnings('ignore')

def preprocess():
    import glob 
    global kwargs_proc

    # Get export path/files
    kwargs_proc['path_pre']= str(DATA_PREPROCESSED / kwargs_proc['exp_out'])
    kwargs_proc['path_proc']= str(DATA_PROCESSED / kwargs_proc['exp_out'])
    kwargs_proc['files_pre']='%s/%s/%s'%(DATA_PREPROCESSED,kwargs_proc['exp_out'],kwargs_proc['exp_out'])
    kwargs_proc['files_proc']='%s/%s/%s'%(DATA_PROCESSED,kwargs_proc['exp_out'],kwargs_proc['exp_out'])

    # Either load preprocessed data or original data
    if kwargs_pre['prioritize_existing']:
        # Check files exist
        if glob.glob(kwargs_proc['files_pre']+'*'):
            time_stop  = []
            time_start  = []
            # Get timestamps of files
            for f in glob.glob(kwargs_proc['files_pre']+'*'): 
                filename_parts = os.path.split(f[:-3])[1].split('_')
                time_stop.append(int(filename_parts[-1]))
                time_start.append(int(filename_parts[-2]))
            # If timestamps are insufficient, continue with original, otherwise skip
            if max(time_stop)>=kwargs_proc['time_stop'] and min(time_start)<=kwargs_proc['time_start']:
                if kwargs_pre['allow_rechunk']:
                    print('Get existing Datasets from %s* and rechunk.'%(kwargs_proc['files_pre']))
                    ds = xr.open_mfdataset(kwargs_proc['files_pre']+'*', decode_times=False, parallel = True)
                    get_original = False
                else:
                    print('Skip renaming and chunking')
                    return 
            else:
                print('Existing data does not have the sufficient time steps, start with renaming\nLoad original Datasets')
                get_original = True
        else:
            print('No preprocessed data found, start with renaming\nLoad original Datasets')
            get_original = True
    else:
        get_original = True

    if get_original:
        print('Load original data')
        ds = open_nemo_and_domain_cfg(
            nemo_files=kwargs_proc['nemo_path']+kwargs_proc['exp_in'],
            domcfg_files=kwargs_proc['domain_path']+kwargs_proc['exp_in'],
            nemo_kwargs={'decode_times':False, 'drop_variables': ['time_instant'], 'parallel':True}
        )
        ds = ds.chunk(ds.dims)
    # Cut data based on timeslice given and create datasets based on chunks
    print('Cut to timeslice %i-%i'%(kwargs_proc['time_start'],kwargs_proc['time_stop']))
    ds = ds.isel(t=slice(kwargs_proc['time_start'],kwargs_proc['time_stop']))
    
    # kwargs_proc['n_chunks'] = get_number_of_chunks(ds, **kwargs_proc)  skip because of snakemake
    print('Create %i timechunks of size %s nbytes'%(kwargs_proc['n_chunks'], ds.nbytes/kwargs_proc['n_chunks']) )

    ds = ds.chunk({'t':int(round((kwargs_proc['time_stop']-kwargs_proc['time_start'])/kwargs_proc['n_chunks'],0))})#.load()
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds, 
                                                                                         path_prefix=kwargs_proc['path_pre'],
                                                                                         sub_prefix=kwargs_proc['exp_out'], 
                                                                                         **kwargs_proc)
    
    
    # Save preprocessed data by chunks
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'],
                           path_prefix=kwargs_proc['path_pre'],
                           sub_prefix=kwargs_proc['exp_out'], **kwargs_proc)
    #xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'])
    print('Preprocess finished')

def prepare_dataset(): 
    ds = xr.open_mfdataset(kwargs_proc['files_pre']+'*', decode_times=False, parallel = True)
    ds = ds.isel(t=slice(kwargs_proc['time_start'],kwargs_proc['time_stop']))
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)
    
    print( 'Prepare key variable for processing' ) 
        
    bd_conf=kwargs_pre['boundary_mask']
    horiz_bd_conf = drop_from_dict(bd_conf, 'Z')

    grid = xgcm.Grid(ds, metrics=get_metrics(ds), periodic=False)
    grid_ops = diagnostics.Grid_ops(grid, boundary=kwargs_proc['xgcm'])

    print("Pass selected variables through")
    for var in kwargs_pre['vars_to_pass']:
        if var in ds.variables:
            ds_proc[kwargs_pre['vars_to_pass'][var]]=ds[var]

    print( 'Add halftime data' )
    t_s = xr.DataArray((ds.t-ds.t.diff('t').mean('t')/2).values,
                          coords={'t':(ds.t-ds.t.diff('t').mean('t')/2).values})
    ds['t_s'] = t_s
    for data in [ds.thetao,ds.so,ds.saltflx,ds.qt]:
        ds[data.name+'_s'] = ds_proc[data.name+'_s'] = xr.DataArray(data.interp({'t':t_s}).values,
                           coords=data.coords)
    ds_proc=ds_proc.rename({'thetao_s':'to_s'})

    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='properties/properties', **kwargs_proc)
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'], 
                           path_prefix=kwargs_proc['path_proc'], 
                           sub_prefix='properties/properties', **kwargs_proc)
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print( 'Add trends' )
    for var in kwargs_pre['vars_to_pass']:
        if '*' in var:
            for var1 in ds.variables:
                if var in var1:
                    ds_proc[var1]=ds[var1]
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='properties/trends', **kwargs_proc)
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'], 
                           path_prefix=kwargs_proc['path_proc'], 
                           sub_prefix='properties/trends', **kwargs_proc)
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print( 'Add masks' )
    ds['maskh_bdeq_t'] = grid_ops.boundary_mask(ds.e2t, horiz_bd_conf)*grid_ops.eq_mask(ds.e2t)
    ds['maskh_bdeq_u'] = grid_ops.boundary_mask(ds.e2u, horiz_bd_conf)*grid_ops.eq_mask(ds.e2u)
    ds['maskh_bdeq_v'] = grid_ops.boundary_mask(ds.e2v, horiz_bd_conf)*grid_ops.eq_mask(ds.e2v)
    ds['maskh_bdeq_f'] = grid_ops.boundary_mask(ds.e2f, horiz_bd_conf)*grid_ops.eq_mask(ds.e2f)
    ds['mask_bd_t'] = grid_ops.boundary_mask(ds.e3t, bd_conf)
    ds['mask_bd_u'] = grid_ops.boundary_mask(ds.e3u, bd_conf)
    ds['mask_bd_v'] = grid_ops.boundary_mask(ds.e3v, bd_conf)
    ds['mask_bd_w'] = grid_ops.boundary_mask(ds.e3w, bd_conf)
    #if kwargs_pre['pass_masks']:
    for var in ds.variables:
        if 'mask' in var: ds_proc[var] = ds[var]
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='domain/masks', **kwargs_proc)
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'], 
                           path_prefix=kwargs_proc['path_proc'], 
                           sub_prefix='domain/masks', **kwargs_proc)
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    #ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, path_prefix=kwargs_proc['files_proc'], **kwargs_proc)
    #xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   
    #ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)
    print( 'Add metrics' )
    ds['e1tm'] = ds_proc['e1tm']= ds.e1t*grid_ops.boundary_mask(ds.e1t, horiz_bd_conf)
    ds['e1um'] = ds_proc['e1um'] = ds.e1u*grid_ops.boundary_mask(ds.e1u, horiz_bd_conf)
    ds['e1vm'] = ds_proc['e1vm'] = ds.e1v*grid_ops.boundary_mask(ds.e1v, horiz_bd_conf)
    ds['e1fm'] = ds_proc['e1fm'] = ds.e1f*grid_ops.boundary_mask(ds.e1f, horiz_bd_conf)
    ds['e2tm'] = ds_proc['e2tm'] = ds.e2t*ds.maskh_bdeq_t
    ds['e2um'] = ds_proc['e2um'] = ds.e2u*ds.maskh_bdeq_u
    ds['e2vm'] = ds_proc['e2vm'] = ds.e2v*ds.maskh_bdeq_v
    ds['e2fm'] = ds_proc['e2fm'] = ds.e2f*ds.maskh_bdeq_f
    ds['e3tm'] = ds_proc['e3tm'] = ds.e3t*ds.mask_bd_t
    ds['e3um'] = ds_proc['e3um'] = ds.e3u*ds.mask_bd_u
    ds['e3vm'] = ds_proc['e3vm'] = ds.e3v*ds.mask_bd_v
    ds['e3wm'] = ds_proc['e3wm'] = ds.e3w*ds.mask_bd_w
    ds['e3tm_1d']= ds_proc['e3tm_1d'] = grid_ops.average(ds.e3tm,['Y','X'])

    ds['depth'] = ds_proc['depth'] = - (ds.gdepw_0.values+ds.e3tm/2)
    ds['depth_1d'] = ds_proc['depth_1d'] = ds.depth.mean(['t','x_c','y_c'])
    ds['depth_s'] = ds_proc['depth_s'] = xr.DataArray(ds['depth'].interp({'t':t_s}).values,coords=ds['depth'].coords)
    
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='domain/metrics', **kwargs_proc)
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'], 
                           path_prefix=kwargs_proc['path_proc'], 
                           sub_prefix='domain/metrics', **kwargs_proc)
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)
    #if kwargs_pre['pass_metrics']:
    #    for var in ds.variables:
    #        if var[:2] in ['e1','e2','e3'] and var[-2:] not in ['1d','_0']: ds_proc[var] = ds[var]
    #    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, path_prefix=kwargs_proc['files_proc'], **kwargs_proc)
    #    xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   
    #    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)
    
    # ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, path_prefix=kwargs_proc['files_proc'], **kwargs_proc)
    # xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   
    # ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    #print('Write data to: %s'%(kwargs_proc['files_proc']))
    #xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'])
    return ds

def process():
    global ds
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print( 'Initialize grid/metrics & diagnostics' )

    _metrics = {('X',): ['e1tm', 'e1um', 'e1vm', 'e1fm'],
     ('Y',): ['e2tm', 'e2um', 'e2vm', 'e2fm'],
     ('Z',): ['e3tm', 'e3um', 'e3vm', 'e3wm']}
    
    bd = {'boundary': 'fill', 'fill_value': 0}
    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = diagnostics.Grid_ops(grid, boundary=bd, maskargs={'mask':'nan'})

    #ds['e3tm_1d']=grid_ops.average(ds.e3tm,['Y','X'])
    grid = grid_ops._update({'Z':ds['e3tm_1d']})
    
    properties=diagnostics.Properties(grid_ops,{'X': ds.glamt,
                                        'Y': ds.gphit,
                                        'Z': ds.depth_1d}, eos_properties=kwargs_sim['eos'])
    power = diagnostics.Power(grid_ops)
    energetics=diagnostics.Energetics(grid_ops, properties)
    trends = {key: kwargs_sim[key] for key in ['T_trends','S_trends','K_trends','P_trends']}
    energetics_trend=diagnostics.Energetics_trends(grid_ops, properties,trends)
    #if kwargs_proc['spinup']:
    #    trends = {key: kwargs_sim[key] for key in ['T_trends','S_trends','K_trends','P_trends']}
    #    energetics_trend=diagnostics.Energetics_trends(grid_ops, properties,trends)
    #else:
    #    energetics_trend=diagnostics.Energetics_trends(grid_ops, properties,{'Configs/processes.ini'})
    transport = diagnostics.Transport(grid_ops)

    maskargs= {'mask':'usr_def','mask_values':ds.mask_bd_t}

    print( 'Add constants' )

    L = 61                        # [degrees] Approximative meridional extend of the basin
    atau         =   0.8          # [no unit]
    deltatau     =   5.77         # [degrees]
    tau0   =  0.1  #[0.1, 0.1333, 0.1666, .2]  

    ####
    print( 'Process, zg for total/global/horizontal mean' )
    
    ds_proc['t_hm'] = properties.horizontal_mean(ds['thetao'],**maskargs)
    ds_proc['s_hm'] = properties.horizontal_mean(ds['so'],**maskargs)

    ds_proc['rho_s'] = properties.density(ds['thetao_s'],ds['so_s'],ds['depth'],**maskargs)
    ds_proc['rho_hm'] = properties.density(ds_proc['t_hm'], ds_proc['s_hm'], ds['depth'])

    print('Write properties to: %s'%(kwargs_proc['files_proc'])) 
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='properties/properties', **kwargs_proc)
    xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    eta_mean=properties.horizontal_mean(ds.zos)
    functions= {#'M':'center_of_gravity_classical', 
                #'PE':'center_of_gravity_PE',
                'DE':'center_of_gravity_h'}

    ds_proc['t_gm'] = properties.global_mean(ds['thetao'],**maskargs)
    ds_proc['s_gm'] = properties.global_mean(ds['so'],**maskargs)
    ds_proc['rho'] = properties.density(ds['thetao'],ds['so'],ds['depth'],**maskargs)
    ds_proc['rho_gm'] = properties.density(ds_proc['t_gm'], ds_proc['s_gm'], ds['depth'])

    ds_proc['zg_eta']= energetics.center_of_gravity_eta(ds.zos,ds_proc.rho,eta_r=eta_mean, boussinesq=True)
    ds_proc['zg_0'] = energetics.center_of_gravity_classical(ds_proc.rho_gm).mean('t')    

    argsh = {#'M':  [ds_proc.rho],
             #'PE': [ds_proc.rho],
             'DE': [ds.thetao, ds.so, ds['depth']]}

    argsh_gm={#'M':  [ds_proc.rho_gm],
              #'PE': [ds_proc.rho_gm],
              'DE': [ds_proc.t_gm, ds_proc.s_gm, ds['depth']]}
    
    kwargsh = {#'M': {'boussinesq':True}, 
              #'PE': {'boussinesq':True,'Z_r':ds_proc['zg_0']}, 
              'DE': {'Z_r':ds_proc['zg_0']}}

    for f in functions:
        ds_proc['zg'+f] = getattr(energetics,functions[f])(*argsh[f],**kwargsh[f])
        ds_proc['zg'+f+'_gm']=getattr(energetics,functions[f])(*argsh_gm[f],**kwargsh[f])

    ###
    if kwargs_proc['dynamics']:
        print( 'Process z_g tendencies' )
        
        exceptions=[]#['tot', 'totad', 'ad', 'xad', 'yad', 'zad']#, 'zdf']
        ketrd_vars=['convp2k','tau']#,'zdf',]#'ldf',]#'atf','hpg','keg','zad','rvo','pvo',]

        T_trends={}
        S_trends={}
        for i in ds.variables:
            if (i[:2]=='tt' and not i[5:] in exceptions):
                T_trends[i[5:]]=ds[i].copy()
            elif (i[:2]=='st' and not i[5:] in exceptions):
                S_trends[i[5:]]=ds[i].copy()

        ds['ketrd_convp2k'] = ds.ketrd_convP2K
        ds = ds.drop('ketrd_convP2K')
        if kwargs_proc['kinetic_energy']:
            KE_trends={}
            for i in ketrd_vars:
                if i!='hpg':
                    KE_trends[i]= -ds['ketrd_'+i].copy()
        else:
            KE_trends={'convp2k':-ds['ketrd_convp2k'].copy()}
            if kwargs_proc['wind_input']:
                KE_trends['tau']= -ds['ketrd_tau'].copy()

        Zeros = np.zeros(ds.ketrd.shape)
        Zeros[:,0] =  1
        for tr in T_trends:
            if T_trends[tr].dims==('t','y_c','x_c'):
                T_trends[tr] = T_trends[tr].expand_dims({'z_c':ds.z_c},axis=1) * Zeros
            if tr == 'aft':
                T_trends[tr] = T_trends[tr]/2
        for tr in S_trends:
            if S_trends[tr].dims==('t','y_c','x_c'):
                S_trends[tr] = S_trends[tr].expand_dims({'z_c':ds.z_c},axis=1) * Zeros
            if tr == 'aft':
                S_trends[tr] = S_trends[tr]/2
        for tr in KE_trends:
            if KE_trends[tr].dims==('t','y_c','x_c'):
                KE_trends[tr] = KE_trends[tr].expand_dims({'z_c':ds.z_c},axis=1) * Zeros


        # Remove wind stress from zdf and separate it. 
        # Factor 0.5*1026 to correct wrong model output (0.5 for utau+utau_b mean)
        if kwargs_proc['wind_input']:
            KE_trends['tau'] *= 0.5*1026*ds.e3tm[:,0]
        if kwargs_proc['kinetic_energy']:
            KE_trends['zdf'] -= KE_trends['tau']#*0.5*1026*ds.e3tm[:,0]

        if kwargs_proc['wind_input']:
            print( 'Diagnose wind power input' )
            
            utau_t=tau0 * (- np.cos( ( 3*np.pi*ds.gphit)/( 2 * L ))+ atau * np.exp(-ds.gphit**2 / deltatau**2 ) )

            Ptaug = power.P_taug(utau_t, utau_t*0, ds.zos, ds.ff_t).transpose(*('t','y_c','x_c'))

            Zeros = np.zeros(ds.ketrd.shape)
            Zeros[:,0] =  1
            Ptaug3d = Ptaug.expand_dims({'z_c':ds.z_c},axis=1) * Zeros# / ds.e3tm[:,0]

            KE_trends['tau'] += Ptaug3d
            KE_trends['taug'] = -Ptaug3d

        kwargst= {'Vmask':ds.mask_bd_t}

        C=KE_trends['convp2k']
        if not kwargs_proc['kinetic_energy']:
            KE_trends=None
        dzgdt=energetics_trend.center_of_gravity_h_trend(0,ds.thetao_s, ds.so_s, ds.depth_s, Z_r=ds_proc.zg_0,
                                                          #T_trend=ds.ttrd_tot,#S_trend=dsstrd_tot,
                                                          #T_trend=dsttrd_tot,S_trend=dsstrd_tot,
                                                          C_trend=C,
                                                          T_trends=T_trends,
                                                          S_trends=S_trends,
                                                          C_trends=KE_trends,**kwargst)


        ds_proc['dzg_tot'] = dzgdt[0]

        proc=['T','S','K']
        for i in range(3):
            for key, value in dzgdt[1][i].items():
                ds_proc['dzg_'+proc[i]+key] = value
                    
        del Zeros
        del T_trends
        del S_trends
        del KE_trends

        print('Write tendencies to: %s'%(kwargs_proc['files_proc'])) 
        ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc,
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='properties/properties', **kwargs_proc)
        #ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, path_prefix=kwargs_proc['files_proc'], **kwargs_proc)
        xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   
        ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    print('Diagnose Meridional Overturning Streamfunction')
    
    psi = transport.compute_moc(ds.vo)
    psi = psi.chunk({d: ds.chunks[d] for d in psi.dims})  # To fix an error in y-chunks 
    ds_proc['psi'] = psi
    ds_proc['psi_maxz'] = psi.max('z_f')
    ds_proc['psi_dmoc'] = ds_proc['psi_maxz'].isel(y_f=slice(9,None)).max('y_f')
    ds_proc['psi_tmoc'] = ds_proc['psi_maxz'].isel(y_f=slice(None,9)).max('y_f')
    
    print('Write MOC to: %s'%(kwargs_proc['files_proc'])) 
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, 
                                                                                         path_prefix=kwargs_proc['path_proc'], 
                                                                                         sub_prefix='properties/moc', **kwargs_proc)
    existing_files_handler(ds_chunks, 
                           kwargs_proc['chunk_paths'], 
                           path_prefix=kwargs_proc['path_proc'], 
                           sub_prefix='properties/moc', **kwargs_proc)
    #xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'], mode='a')   

def postprocess():
    print('Run postprocessing')
    global ds
    ds_proc = xr.Dataset(coords = ds.coords, attrs = ds.attrs)

    _metrics = {('X',): ['e1tm', 'e1um', 'e1vm', 'e1fm'],
     ('Y',): ['e2tm', 'e2um', 'e2vm', 'e2fm'],
     ('Z',): ['e3tm', 'e3um', 'e3vm', 'e3wm']}
    
    bd = {'boundary': 'fill', 'fill_value': 0}
    grid = xgcm.Grid(ds, metrics=_metrics, periodic=False)
    grid_ops = diagnostics.Grid_ops(grid, boundary=bd, maskargs={'mask':'nan'})

    p=diagnostics.Properties(grid_ops,{'X': ds.glamt, 'Y':ds.gphit,'Z': ds.depth_1d})
    
    def clean(data):
        return (data.where(data!=0).chunk(dict(t=-1))).interpolate_na(dim=('t'),method='cubic')    

    def mean(d, mask):
        data = {}
        data['Forcing'] = p.global_mean(d.dzg_Tqns + d.dzg_Tqsr + d.dzg_Scdt,             Vmask=mask)
        data['Mixing'] = p.global_mean(d.dzg_Tzdf + d.dzg_Tldf + d.dzg_Szdf + d.dzg_Sldf,Vmask=mask)
        data['Conversion'] = p.global_mean(d.dzg_Kconvp2k,                                   Vmask=mask)
        if not kwargs_proc['spinup']:
            data['Atf']  = p.global_mean(d.dzg_Tatf + d.dzg_Satf,                          Vmask=mask)
            data['Total']= p.global_mean(d.dzg_Tzdf + d.dzg_Tldf + d.dzg_Szdf + d.dzg_Sldf +
                                                      d.dzg_Tqns + d.dzg_Tqsr + d.dzg_Scdt +
                                                      d.dzg_Tatf + d.dzg_Satf,             Vmask=mask)
        else:
            data['Total']= p.global_mean(d.dzg_Tzdf + d.dzg_Tldf + d.dzg_Szdf + d.dzg_Sldf +
                                                      d.dzg_Tqns + d.dzg_Tqsr + d.dzg_Scdt,Vmask=mask)
        if kwargs_proc['kinetic_energy']:
            data['Tau']  = p.global_mean(d.dzg_Ktau,                                       Vmask=mask)
            if kwargs_proc['wind_input']:
                data['Taug'] = p.global_mean(d.dzg_Ktaug,                                      Vmask=mask)
        return data
    
    proc = ['Forcing','Mixing','Conversion','Atf','Total','Tau', 'Taug']
    post_proc = mean(ds_proc, ds.mask_bd_t)
    post_procml = mean(ds_proc*grid_ops.mld_mask(ds.e3tm,ds.mldr10_1), ds.mask_bd_t)
    post_procbl = mean(ds_proc*grid_ops.mld_mask(ds.e3tm,ds.mldr10_1,invert=True), ds.mask_bd_t)
    for p in proc:
        if p=='Atf' and kwargs_proc['spinup']:
            continue
        elif p=='Taug' and not kwargs_proc['wind_input']:
            continue
        elif (p=='Tau' or p=='Taug') and not kwargs_proc['kinetic_energy']:
            continue
        ds_proc[p] = post_proc[p]
        ds_proc[p+'_ml'] = post_procml[p]
        ds_proc[p+'_bl'] = post_procbl[p] 
        
    ds_chunks, kwargs_proc['chunk_slices'], kwargs_proc['chunk_paths'] = split_by_chunks(ds_proc, path_prefix=kwargs_proc['files_proc'], **kwargs_proc)

    print('Write data to: %s'%(kwargs_proc['files_proc'])) 
    
    xr.save_mfdataset(datasets=ds_chunks, paths=kwargs_proc['chunk_paths'],mode='a')     
                          
if __name__ == '__main__':
    import argparse
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    from dask.distributed.client import  _get_global_client
    
    # Parse arguments from script execution
    parser = argparse.ArgumentParser(description='ECO Processing script.')
    parser.add_argument('-sub_config','-c','--c', type=str_or_none, default=None)
    parser.add_argument('-config_path','-p','--p', default='')
    parser.add_argument('-d','--d', type=bool, default=False)
    parser.add_argument('-mode','-m','--m', type=str_or_none, default=None)
    args = parser.parse_args()
    sub_config, config_path, debug_mode, mode = args.c, args.p, args.d, args.m

    # Load configs
    kwargs_proc, kwargs_pre, kwargs_sim = config_parser(config_path=config_path, sub_config=sub_config, log=True)

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
    if mode=='preprocess':
        preprocess()
    else:
        preprocess()
        ds = prepare_dataset()
        process()
        if kwargs_proc['postprocessing']: postprocess()
    print('Processing finished. Runtime: ', datetime.now() - startTime)