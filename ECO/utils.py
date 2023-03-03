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
    ds_chunks=[]
    for slices in itertools.product(*chunk_slices.values()):
        selection = dict(zip(chunk_slices.keys(), slices))
        ds_chunks.append(dataset[selection])

    chunk_paths = ['%s/%s_%i_%i.nc'%(str(export_path),file_prefix, chunk_slices['t'][i].start+kwargs['time_start'],
                                                chunk_slices['t'][i].stop+kwargs['time_start']) for i in range(len(chunk_slices['t']))]
    return [ds_chunks, chunk_slices['t'], chunk_paths]

def existing_files_handler(datasets, datapaths, path_prefix='.', sub_prefix='', sub_suffix='',  **kwargs):
    import os
    import xarray as xr
    from pathlib import PosixPath

    datapaths_out = [p for p in datapaths]
    def save():
        print('Write data to: %s*'%(PosixPath(path_prefix) / PosixPath(sub_prefix+sub_suffix)))
        xr.save_mfdataset(datasets=datasets, paths=datapaths_out)
        
    export_path = PosixPath(path_prefix) / PosixPath(sub_prefix).parent #= os.path.split(filespath_prefix)[0]
    file_prefix = PosixPath(sub_prefix+sub_suffix).name
    # Create directory if not existing
    #if not os.path.exists(export_path):
    if not export_path.exists():
        print('Create output folder')
        export_path.mkdir()
        save()
        return
    
    # Check if files do not exist -> skip
    if not list(export_path.glob(file_prefix+'*')):
        print('No files exist yet, proceed')
        save()
        return
    else:
        time_stop  = []
        time_start  = []
        file_paths = []
        tmp_files = []
        for f in list(export_path.glob(file_prefix+'*')): 
            file_name_parts = f.name[:-3].split('_')
            if file_name_parts[-1]=='tmp': 
                tmp_files.append(f)
                continue
            time_stop.append(int(file_name_parts[-1]))
            time_start.append(int(file_name_parts[-2]))
            file_paths.append(f)

        # Remove tmp files
        for f in tmp_files:
            os.remove(f)
        # If existing timestamps are before time_start -> skip
        if max(time_stop) < kwargs['time_start']:
            print('Timestamp of existing files is prior to selected time_start')
            save()
            return
        else:
            # If override not active, ask for override permission
            if not kwargs['data_override']:
                input_text = 'Action required!\n\
                              There is existing data of the selected simulation and timesteps,\n\
                              but override is disabled.\n\
                              Remove everything with timestamp time_start time_stop? [y/n]'
                choice = ''
                while choice not in ['y','n']:
                    choice = input(input_text).lower()
                
                # If permission not granted -> skip
                if choice == 'n':    
                    print('Keep existing data. Duplicates may occur.')
                    save()
                    return

            # If override granted -> remove files with timestamp later time_start or before time_stop
            files_to_remove=[]
            for i in range(len(file_paths)):  
                if (time_stop[i] > kwargs['time_start'] and time_stop[i] < kwargs['time_stop']):
                    files_to_remove.append(file_paths[i])
                elif (time_start[i] < kwargs['time_stop'] and time_start[i] >= kwargs['time_start']):
                    files_to_remove.append(file_paths[i])

            datapaths_out = [p[:-3]+'_tmp.nc' for p in datapaths]
            save()
            for p in range(len(datapaths)):
                #print('Remove & rename', files_to_remove[p])
                os.remove(files_to_remove[p])
            for p in range(len(datapaths)):
                os.rename(datapaths_out[p],datapaths[p])
             
def save_by_chunks(ds, path_prefix='', sub_prefix='', sub_suffix='', mode='w', **kwargs):
    ds_chunks, chunk_slices, chunk_paths = split_by_chunks(ds,
                                                           path_prefix=path_prefix, 
                                                           sub_prefix=sub_prefix,
                                                           sub_suffix=sub_suffix, **kwargs)
    if mode=='w':
        existing_files_handler(ds_chunks, chunk_paths, path_prefix=path_prefix, sub_prefix=sub_prefix, sub_suffix=sub_suffix, **kwargs)
    elif mode=='a':
        import xarray as xr
        xr.save_mfdataset(datasets=ds_chunks, paths=chunk_paths, mode='a')   

def open_datasets(exp, exp_suffix, components='all', combine=True, parallel=True):
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
                datasets.append(xr.open_mfdataset(f"{DATA / 'domain'}/*", decode_times=False, parallel=parallel, data_vars='minimal') )
                datasets.append(xr.open_mfdataset(f"{DATA / 'properties'}/*", decode_times=False, parallel=parallel, data_vars='minimal') )
            elif comp in ['metrics', 'masks'] and 'domain' in directories:
                datasets.append(xr.open_mfdataset(f"{DATA / 'domain'}/{comp}*", decode_times=False, parallel=parallel, data_vars='minimal') )
            elif comp in ['properties', 'moc', 'trends'] and 'properties' in directories:
                datasets.append(xr.open_mfdataset(f"{DATA / 'properties'}/{comp}*", decode_times=False, parallel=parallel, data_vars='minimal') )  
        
        # Save in dictionary, merge components if combine==True
        if combine:
            out[exp_full] = xr.merge(datasets)
        else:
            out[exp_full] = datasets

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
        print("Export_path: %s"             % kwargs_proc['path_proc'])
        print("Time_start: %s"              % kwargs_proc['time_start'])
        print("Time_stop: %s"               % kwargs_proc['time_stop'])
        print("Dynamics: %s"                % kwargs_proc['dynamics'])
        print("Kinetic energy: %s"          % kwargs_proc['kinetic_energy'])
        print("Diagnose wind power: %s"     % kwargs_proc['wind_input'])
        print("Spinup: %s"                  % kwargs_proc['spinup'])
        print("Postprocess: %s"             % kwargs_proc['postprocessing'])

    return kwargs_proc, kwargs_pre, kwargs_sim

def get_namelist(path = None):
    import f90nml
    from pathlib import Path
    if type(path)==str: path = Path(path)

    nml = f90nml.read(path / 'namelist_ref')
    nml_cfg = f90nml.read(path / 'namelist_cfg_run')
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
        print('Subconfiguuation already in place')

def drop_from_dict(d, k):
    d_out = d.copy()
    try:
        del d_out[k]
    except KeyError as ex:
        print("No such key: '%s'" % ex.message)
    return d_out

def str_or_none(s):
    return None if s=='None' else s
