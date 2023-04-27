def get_number_of_chunks(ds, **kwargs) -> int: 
    # Function to evaluate the chunksize based on default number of chunks and maximum size
    n_chunks = kwargs['n_chunks']

    if kwargs['max_chunksize'][-3:] == 'GiB': 
        byte_convert = (1024**3)
    elif kwargs['max_chunksize'][-3:] == 'MiB': 
        byte_convert = (1024**2)
    else:
        print("Unknown sizetype %s" %(kwargs['max_chunksize']) )

    if ds.nbytes/byte_convert/n_chunks > float(kwargs['max_chunksize'][:-3]):
        while ds.nbytes/byte_convert/n_chunks > float(kwargs['max_chunksize'][:-3]):
            n_chunks += 1
        print('Adjust number of chunks to %i'%(n_chunks))
    return n_chunks

def split_by_chunks(dataset, path_prefix='.', sub_prefix='', sub_suffix ='', **kwargs):
    import itertools
    from pathlib import PosixPath
    export_path = PosixPath(path_prefix) / PosixPath(sub_prefix).parent #= os.path.split(filespath_prefix)[0]
    file_prefix = PosixPath(sub_prefix+sub_suffix).name
    
    
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            if start >= dataset.sizes[dim]:
                break
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices

    if not 't' in chunk_slices:
        n_chunk = kwargs['n_chunks']
        chunk = int((((kwargs['time_stop'])-kwargs['time_start'])/n_chunk))

        chunk_slices['t'] = [slice(i*chunk, (i+1)*chunk) if i<n_chunk-1
                             else slice(i*chunk, kwargs['time_stop']) for i in range(n_chunk)]
    
    ds_chunks=[]
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        ds_chunks.append(dataset[selection])
    
    
    chunk_paths = ['%s/%s_%i_%i.nc'%(str(export_path),file_prefix, chunk_slices['t'][i].start+kwargs['time_start'],
                                                chunk_slices['t'][i].stop+kwargs['time_start']) for i in range(len(chunk_slices['t']))]
    return [ds_chunks, chunk_slices['t'], chunk_paths]

def get_chunknames(sub_prefix = '', sub_suffix = '', **kwargs):
    from pathlib import PosixPath

    file_prefix = PosixPath(sub_prefix+sub_suffix).name
    
    file_names = []
    chunk_steps = round((kwargs['time_stop']-kwargs['time_start'])/kwargs['n_chunks'])
    for c in range(kwargs['n_chunks']):
        start = kwargs['time_start'] + chunk_steps * c
        if c == kwargs['n_chunks']-1:
            stop  = kwargs['time_stop']
        else:
            stop  = kwargs['time_start'] + chunk_steps * (c+1)
        file_names.append(f'{file_prefix}_{start}_{stop}.nc')
    return file_names

def existing_files_handler(datasets, datapaths, path_prefix='.', sub_prefix='', sub_suffix='', mode='w',  **kwargs):
    import os
    import xarray as xr
    from pathlib import PosixPath

    if type(datapaths) is list: datapaths_out = [p for p in datapaths]
    if datapaths is None: datapaths_out=''

    def save():
        print('Write data to: %s*'%(PosixPath(path_prefix) / PosixPath(sub_prefix+sub_suffix)))
        xr.save_mfdataset(datasets=datasets, paths=datapaths_out)

    export_path = PosixPath(path_prefix) / PosixPath(sub_prefix).parent 
    file_prefix = PosixPath(sub_prefix+sub_suffix).name
    
    # Check if directory exists
    if not export_path.exists():
        if mode=='w':
            print('Create output folder')
            export_path.mkdir()
            save()
            return
        elif mode=='r':
            print('Directory does not exist')
            return 1 # Read original
    
    # Check if any files exist
    existing_files = [f.name for f in list(export_path.glob(file_prefix+'*'))]
    if not existing_files:
        if mode=='w':
            print('No files exist yet, proceed')
            save()
            return
        elif mode=='r':
            print('Files do not exist')
            return 1 # Read original
        
    # Search existing files
    # Handle read mode first
    file_names = get_chunknames(sub_prefix = sub_prefix, sub_suffix = sub_suffix, **kwargs)
    if mode=='r':
        if all(l in existing_files for l in file_names):
            print('All files already exists')
            return 0 # Skip preprocess
        else:
            print('Existing files incomplete or chunked differently')
            # TODO - Function to check if sufficient data is there, otherwise fall back to original
            # Data with same timeframe but different chunks would be appropriate
            return 2 # Read existing
        
    # Handle write mode next    
    elif mode =='w':
        # If all expected files exist
        if all(l in existing_files for l in file_names):
            print('All files already exists')
            if kwargs['data_override']:
                files_to_remove = file_names
            else:
                print('Data override is disabled -> Skip')
                return
        else:
            # Check if single files already exist
            files_to_remove = []
            tmp_files = []
            for f in existing_files: 
                file_name_parts = f[:-3].split('_')
                if file_name_parts[-1]=='tmp': 
                    tmp_files.append(export_path+f)
                    continue
                stop = int(file_name_parts[-1])
                start = int(file_name_parts[-2])
                if stop < kwargs['time_start'] or start > kwargs['time_stop']:
                    continue
                files_to_remove.append(export_path / f)

    # Remove tmp files
    for f in tmp_files:
        os.remove(f)
    # If existing timestamps are before time_start -> skip
    if not files_to_remove:
        print('No files in selected timeframe exist yet')
        save()
        return
    else:
        # If override not active, ask for override permission
        if not kwargs['data_override']:
            input_text = 'Action required!\n\
                          There is single existing data of the selected simulation\n\
                          in the selected timeframe, but override is disabled.\n\
                          Allow override? [y/n]'
            choice = ''
            while choice not in ['y','n']:
                choice = input(input_text).lower()
            
            if choice == 'n':   
                print('Keep existing data. Duplicates may occur.')
                save()
                return
   
    datapaths_out = [p[:-3]+'_tmp.nc' for p in datapaths]
    save()
    for p in range(len(files_to_remove)):
        os.remove(files_to_remove[p])
    for p in range(len(datapaths)):
        os.rename(datapaths_out[p],datapaths[p])

def read_chunks(path_prefix='.', sub_prefix='', sub_suffix='', **kwargs):
    from pathlib import PosixPath
    import xarray as xr

    export_path = PosixPath(path_prefix) / PosixPath(sub_prefix).parent #= os.path.split(filespath_prefix)[0]
    file_prefix = PosixPath(sub_prefix+sub_suffix).name

    files_to_read = []
    for f in list(export_path.glob(file_prefix+'*')):
        file_name_parts = f.name[:-3].split('_')
        if file_name_parts[-1]=='tmp': continue
        stop = int(file_name_parts[-1])
        start = int(file_name_parts[-2])
        if stop < kwargs['time_start'] or start > kwargs['time_stop']:
            continue
        files_to_read.append(f)

    return xr.open_mfdataset(files_to_read, decode_times=False, parallel = True)

def save_by_chunks(ds, path_prefix='', sub_prefix='', sub_suffix='', mode='w', **kwargs):
    import xarray as xr
    # Split only possible if any variable is existing. If not then create a dummy
    #if not [var for var in ds.variables if var not in ds.coords]:
    #    ds['dummy'] = xr.zeros_like(ds.t).chunk(
    #        {'t':int(round((kwargs['time_stop']-kwargs['time_start'])/kwargs['n_chunks']))})# Otherwise add dummy# Split only possible if actual variables are present
    
    ds_chunks, chunk_slices, chunk_paths = split_by_chunks(ds,
                                                           path_prefix=path_prefix, 
                                                           sub_prefix=sub_prefix,
                                                           sub_suffix=sub_suffix, **kwargs)
    
    if mode=='w':
        existing_files_handler(ds_chunks, chunk_paths, path_prefix=path_prefix, sub_prefix=sub_prefix, sub_suffix=sub_suffix, mode='w', **kwargs)
    elif mode=='a':
        xr.save_mfdataset(datasets=ds_chunks, paths=chunk_paths, mode='a') 

def open_datasets(exp, exp_suffix, components='all', combine=True, parallel=True, timestep_slice=None):
    from pathlib import Path
    import xarray as xr
    import os,sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    sys.path.insert(0, parent_dir)

    # Get pairs of exp/exp_suffix if multiple simulations are requested
    pairs = []
    if type(exp) is str and type(exp_suffix) is str:
        pairs.append((exp,exp_suffix))
    elif type(exp) is list and type(exp_suffix) is str:
        for e in exp:
            pairs.append((e,exp_suffix))
    elif type(exp) is str and type(exp_suffix) is list:
        for e in exp_suffix:
            pairs.append((exp,e))
    elif type(exp) is list and type(exp_suffix) is list:
        if len(exp)!=len(exp_suffix):
            raise Exception("Could not match the lists exp_prefix and exp_suffix, provide lists of equal length")
        else:
            for i in range(len(exp)):
                pairs.append((exp[i],exp_suffix[i]))
    
    # Loop over simulation
    out={}
    for exp,exp_suffix in pairs:
        exp_full = exp+exp_suffix

        if type(timestep_slice)==dict: 
            try:    ts = timestep_slice[exp_suffix]
            except: ts = timestep_slice[exp_full]
        else:       ts = timestep_slice

        # Get path
        exp_full = exp+exp_suffix
        DATA = Path('data/processed') / exp / exp_full
        if DATA.is_dir():
            pass
        elif ('..' / DATA).is_dir():
            DATA = '..' / DATA

        # Get subdirectories
        directories = []
        for f in list(DATA.glob('*')): 
            if f.is_dir(): directories.append(f.name)

        # Collect components selected from the subdirectories
        datasets = []
        for comp in components:
            if comp == 'all':
                if ts is None: 
                    datasets.append(xr.open_mfdataset(list((DATA / 'domain').glob('*')), decode_times=False, parallel=parallel, data_vars='minimal') )
                    datasets.append(xr.open_mfdataset(list((DATA / 'properties').glob('*')), decode_times=False, parallel=parallel, data_vars='minimal') )
                else:
                    files_to_open = list(file for file in list((DATA / 'domain').glob('*')) if int(str(file)[:-3].split('_')[-2])<ts[1])
                    datasets.append(xr.open_mfdataset(files_to_open, decode_times=False, parallel=parallel, data_vars='minimal') )
                    files_to_open = list(file for file in list((DATA / 'properties').glob('*')) if int(str(file)[:-3].split('_')[-2])<ts[1])
                    datasets.append(xr.open_mfdataset(files_to_open, decode_times=False, parallel=parallel, data_vars='minimal') )

            elif comp in ['metrics', 'masks'] and 'domain' in directories:
                if ts is None: 
                    datasets.append(xr.open_mfdataset(f"{DATA / 'domain'}/{comp}*", decode_times=False, parallel=parallel, data_vars='minimal') )
                else:
                    files_to_open = list(file for file in list((DATA / 'domain' / comp).glob('*')) if int(str(file)[:-3].split('_')[-2])<ts[1])
                    datasets.append(xr.open_mfdataset(files_to_open, decode_times=False, parallel=parallel, data_vars='minimal') )
            elif comp in ['properties', 'moc', 'trends'] and 'properties' in directories:
                if ts is None: 
                    datasets.append(xr.open_mfdataset(f"{DATA / 'properties'}/{comp}*", decode_times=False, parallel=parallel, data_vars='minimal') )  
                else:
                    files_to_open = list(file for file in list((DATA / 'properties' / comp).glob('*')) if int(str(file)[:-3].split('_')[-2])<ts[1])
                    datasets.append(xr.open_mfdataset(files_to_open, decode_times=False, parallel=parallel, data_vars='minimal') )
        
        # Save in dictionary, merge components if combine==True
        if combine:
            if ts is None: out[exp_full] = xr.merge(datasets)
            else:          out[exp_full] = xr.merge(datasets).isel({'t':slice(*ts)})
        else:
            if ts is None: out[exp_full] = datasets
            else:          out[exp_full] = datasets.isel({'t':slice(*ts)})

    # Return dataset if only one simulation requested, otherwise return dictionary
    if len(out)==1:
        return list(out.values())[0]
    else:
        return out
    
def config_parser(exp_family, exp_suffix = None ,log=False):
    import yaml
    import os, sys
    from pathlib import Path 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, '..'))

    DATA = parent_dir / Path('data')
    CONFIGS = parent_dir / Path('configs')
    DATA_RAW = DATA / 'raw' 
    DATA_PROCESSED = DATA / 'processed'
    DATA_PREPROCESSED = DATA / 'preprocessed'

    # Load and store base config in kwargs
    with open(CONFIGS / exp_family / "base.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    kwargs_proc = cfg['processing']
    kwargs_pre = cfg['preprocessing']
    kwargs_sim = cfg['simulation']

    # update kwargs if subconfiguration file is provided
    if exp_suffix is None:
        print('Using processing configuration base.yml')
    else:
        if exp_family in exp_suffix: exp_suffix=exp_suffix.replace(exp_family,'')
        print(f'Update processing configuration with subconfig{exp_suffix}')
        exp_suffix = f'subconfig{exp_suffix}' if exp_suffix[-4:] == '.yml' else f'subconfig{exp_suffix}.yml'
        if (CONFIGS / exp_family / exp_suffix).exists(): # os.path.exists('%s/%s'%(config_path ,sub_config)):
            with open(CONFIGS / exp_family / exp_suffix, 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            if 'processing' in cfg: kwargs_proc = {**kwargs_proc,**cfg['processing']}
            if 'preprocessing' in cfg: kwargs_pre = {**kwargs_pre,**cfg['preprocessing']}
            if 'simulation' in cfg: kwargs_sim = {**kwargs_sim,**cfg['simulation']}
        else:
            print('Subconfiguration %s does not exist'%(exp_suffix))

    # Add export paths/files
    exp_full = kwargs_proc['exp_out']+kwargs_proc['exp_out_suffix']
    kwargs_proc['path_nemo']   = Path(kwargs_proc['nemo_path'])   / kwargs_proc['exp_in']
    kwargs_proc['path_domain'] = Path(kwargs_proc['domain_path']) / kwargs_proc['exp_in']
    if not kwargs_proc['path_nemo'].is_dir(): kwargs_proc['path_nemo'] = parent_dir / kwargs_proc['path_nemo'] 
    if not kwargs_proc['path_domain'].is_dir(): kwargs_proc['path_domain'] = parent_dir / kwargs_proc['path_domain'] 

    #kwargs_proc['path_raw']    = str(DATA_RAW          / kwargs_proc['exp_out'] / exp_full)
    kwargs_proc['path_pre']    = str(DATA_PREPROCESSED / kwargs_proc['exp_out'] / exp_full)
    kwargs_proc['path_proc']   = str(DATA_PROCESSED    / kwargs_proc['exp_out'] / exp_full)
    kwargs_proc['files_pre']   = str(DATA_PREPROCESSED / kwargs_proc['exp_out'] / exp_full / exp_full)
    
    if log:
        print("Input: %s"                   % kwargs_proc['exp_in'])
        print("Output: %s"                  % kwargs_proc['exp_out'])
        print("Output suffix: %s"           % kwargs_proc['exp_out_suffix'])
        print("Nemo_path: %s"               % kwargs_proc['path_nemo'])
        print("Domain_path: %s"             % kwargs_proc['path_domain'])
        print("Processed_path: %s"          % kwargs_proc['path_proc'])
        print("Preprocessed_path: %s"       % kwargs_proc['path_pre'])
        print("Time_start: %s"              % kwargs_proc['time_start'])
        print("Time_stop: %s"               % kwargs_proc['time_stop'])
        print("Dynamics: %s"                % kwargs_proc['dynamics'])
        print("Kinetic energy: %s"          % kwargs_proc['kinetic_energy'])
        print("Diagnose wind power: %s"     % kwargs_proc['wind_input'])
        #print("Spinup: %s"                  % kwargs_proc['spinup'])
        print("Postprocess: %s"             % kwargs_proc['postprocessing'])

    return kwargs_proc, kwargs_pre, kwargs_sim

def get_namelist(exppath = None):
    import f90nml, os

    try:
        nml     = f90nml.read(f'{exppath}/namelist_ref')
        try: nml_cfg = f90nml.read(f'{exppath}/namelist_cfg')
        except: nml_cfg = f90nml.read(f'{exppath}/namelist_cfg_run')
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
        nml     = f90nml.read(f'{parent_dir}/{exppath}/namelist_ref')
        try: nml_cfg = f90nml.read(f'{parent_dir}/{exppath}/namelist_cfg')
        except: nml_cfg = f90nml.read(f'{parent_dir}/{exppath}/namelist_cfg_run')
    
    return {**nml, **nml_cfg}

def config_preparation(exp='EXP00_test',suffix='',time_start=0,time_stop=360,n_chunks=4):
    from pathlib import Path 
    import yaml, os, shutil

    if not os.path.exists(f'configs/{exp}'):
        print(f'Initialize Configuration configs/{exp}/')
        os.mkdir(f'configs/{exp}')
        shutil.copyfile('configs/base.yml', f'configs/{exp}/base.yml')
    elif not list(Path(f'configs/{exp}/').glob('base.yml')):
        shutil.copyfile('configs/base.yml', f'configs/{exp}/base.yml')

    data = dict(
            preprocessing = dict(
                prioritize_existing = True
                ),
            processing = dict(
                exp_in = f"{exp}{suffix}",
                exp_out = exp,
                exp_out_suffix = suffix,
                n_chunks = n_chunks,
                time_start  = time_start,
                time_stop   = time_stop
                )
            )
    if 'rest' in suffix:
        data['processing']['dynamics'] = False
        data['processing']['nemo_path'] = f'data/raw/{exp}'
        data['processing']['domain_path'] = data['processing']['nemo_path']

    if list(Path(f'configs/{exp}/').glob(f'subconfig{suffix}.yml')):
        with open(f'configs/{exp}/subconfig{suffix}.yml') as infile:
            yaml_data = yaml.safe_load(infile)
        write = not yaml_data == data
    else:
        write = True
    if write:
        print('Update subconfiguration')
        with open(f'configs/{exp}/subconfig{suffix}.yml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    else:
        print('Subconfiguration already in place')

def drop_from_dict(d, k):
    d_out = d.copy()
    try:
        del d_out[k]
    except KeyError as ex:
        print("No such key: '%s'" % ex.message)
    return d_out

def str_or_none(s):
    return None if s=='None' else s

def combine_spinup(exp_family):
    import xarray as xr
    import numpy as np
    from pathlib import Path 
    import yaml

    print('Get Path')
    DATA_RAW = Path('data/raw')
    
    with open(f'configs/{exp_family}/base.yml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    DATA_SPINUP = Path(cfg['processing']['nemo_path'])

    p_from = DATA_SPINUP / f'{exp_family}_rest6m2000'
    p_to = DATA_RAW / f'{exp_family}' / f'{exp_family}_spinup'
    print('Read variables from', p_from / 'Spinup' / 'BASIN_6m_0*') 
    ds = xr.open_mfdataset(str(p_from / 'Spinup' / 'BASIN_6m_0*'), compat='override', decode_times=False)
    variables = []
    drops = []
    extra = ['depth', 'time_cen','time_ins']#,'bounds']
    for v in ds.variables:
        if v in ds.coords and v not in ds.dims:
            drops.append(v)
        elif np.any([d in v for d in extra]):
            drops.append(v)
        else:
            variables.append(v)
            
    year = np.arange(0,2050,50)
    for arakawa in ['T', 'U', 'V', 'W']:
        print(f'Combine {arakawa}-point')
        ds = [xr.open_dataset(str(p_from / 'Spinup' / f'BASIN_6m_{y}_grid_{arakawa}.nc'),
                                        decode_times=False, drop_variables=drops) for y in year]
        ds_full = xr.concat(ds,dim='time_counter')
        print(f'Write to: ', p_to / f'{exp_family}_spinup_grid_{arakawa}.nc')
        ds_full.to_netcdf(p_to / f'BASIN_spinup_grid_{arakawa}.nc')

        del ds, ds_full
        print(f'{arakawa} - OK')

    print('Copy domain_cfg* and namelist')
    import shutil

    for f in list((p_from).glob('domain_cfg*')):
        shutil.copyfile(f, p_to / f.name)

    shutil.copyfile(p_from / 'namelist_ref', p_to / 'namelist_ref')
    try: shutil.copyfile(p_from / 'namelist_cfg', p_to / 'namelist_cfg')
    except: shutil.copyfile(p_from / 'namelist_cfg_restart', p_to / 'namelist_cfg')

def combine_data(exp_family, exp_suffix):
    import xarray as xr
    import numpy as np
    from pathlib import Path 
    import yaml, os

    print('Get Path')
    DATA_RAW = Path('data/raw')
    
    with open(f'configs/{exp_family}/base.yml', "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    DATA_OUT = Path(cfg['processing']['nemo_path'])

    p_from = DATA_OUT / f'{exp_family}{exp_suffix}'
    p_to = DATA_RAW / f'{exp_family}' / f'{exp_family}{exp_suffix}'
    files_from = list((p_from / 'Data').glob('BASIN*'))
    files_prefix = '%s_%s'%tuple(files_from[0].name.split('_')[:2])
    years = list(set([int(f.name.split('_')[2]) for f in files_from]))
    years.sort()
    
    print('Read variables from', p_from / 'Data' / f'{files_prefix}_0*') 
    ds = xr.open_mfdataset(str(p_from / 'Data' / f'{files_prefix}_0*'), compat='override', decode_times=False)
    variables = []
    drops = []
    extra = ['depth', 'time_cen','time_ins']#,'bounds']
    for v in ds.variables:
         if v in ds.coords and v not in ds.dims:
             drops.append(v)
         elif np.any([d in v for d in extra]):
             drops.append(v)
         else:
             variables.append(v)
    
    for arakawa in ['T', 'U', 'V', 'W']:
        print(f'Combine {arakawa}-point')
        ds = [xr.open_dataset(str(p_from / 'Data' / f'{files_prefix}_{y}_grid_{arakawa}.nc'),
                                        decode_times=False, drop_variables=drops) for y in years]
        ds_full = xr.concat(ds,dim='time_counter')
        
        if not p_to.exists():
            os.mkdir(p_to)

        print(f'Write to: ', p_to / f'BASIN_combine_grid_{arakawa}.nc')
        ds_full.to_netcdf(p_to / f'BASIN_combine_grid_{arakawa}.nc')
        print(f'{arakawa} - OK')

    print('Copy domain_cfg* and namelist')
    import shutil

    for f in list((p_from).glob('domain_cfg*')):
        shutil.copyfile(f, p_to / f.name)

    shutil.copyfile(p_from / 'namelist_ref', p_to / 'namelist_ref')
    try: shutil.copyfile(p_from / 'namelist_cfg', p_to / 'namelist_cfg')
    except: shutil.copyfile(p_from / 'namelist_cfg_run', p_to / 'namelist_cfg')
