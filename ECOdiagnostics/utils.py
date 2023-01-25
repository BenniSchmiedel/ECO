
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

def split_by_chunks(dataset, path_prefix='.', **kwargs):
    import itertools

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

    chunk_paths = ['%s_%i_%i.nc'%(path_prefix, chunk_slices['t'][i].start+kwargs['time_start'],
                                                chunk_slices['t'][i].stop+kwargs['time_start']) for i in range(len(chunk_slices['t']))]
    return [ds_chunks, chunk_slices['t'], chunk_paths]

def existing_files_handler(datasets, datapaths, filespath_prefix, **kwargs):
    import os
    import glob
    import xarray as xr

    def save():
        print('Write data to: %s'%(filespath_prefix))
        xr.save_mfdataset(datasets=datasets, paths=datapaths)
        
    export_path = os.path.split(filespath_prefix)[0]
    # Create directory if not existing
    if not os.path.exists(export_path):
        print('Create output folder')
        os.mkdir(export_path)
        save()
        return
    
    # Check if files do not exist -> skip
    if not glob.glob(filespath_prefix+'*'):
        print('No files exist yet, proceed')
        save()
        return
    else:
        time_stop  = []
        time_start  = []
        file_paths = []
        for f in glob.glob(filespath_prefix+'*'): 
            file_name_parts = os.path.split(f[:-3])[1].split('_')
            time_stop.append(int(file_name_parts[-1]))
            time_start.append(int(file_name_parts[-2]))
            file_paths.append(f)

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
                    print('Remove ', file_paths[i])
                    files_to_remove.append(file_paths[i])
                elif (time_start[i] < kwargs['time_stop'] and time_start[i] >= kwargs['time_start']):
                    print('Remove ', file_paths[i])
                    files_to_remove.append(file_paths[i])#os.remove(file_paths[i]) 

            datapaths_og = datapaths.copy()
            for p in range(len(datapaths)):
                datapaths[p] = datapaths[p][:-3]+'_temp.nc'
                print('Temporary save to %s'%(datapaths[p]))
            save()
            for f in files_to_remove:
                print('Remove ', f)
                os.remove(f)
            for p in range(len(datapaths)):
                print('Rename to %s'%(datapaths_og[p]))
                os.rename(datapaths[p],datapaths_og[p])
             
def drop_from_dict(d, k):
    d_out = d.copy()
    try:
        del d_out[k]
    except KeyError as ex:
        print("No such key: '%s'" % ex.message)
    return d_out

def str_or_none(s):
    return None if s=='None' else s

def config_parser(config_path='Configs/', sub_config=None ,log=False):
    import yaml
    import os

    kwargs_proc, kwargs_proc['config_path'], kwargs_proc['sub_config'] = {}, config_path, sub_config

    # Load and store base config in kwargs
    print("%s/base.yml"%(config_path))
    with open("%s/base.yml"%(config_path), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    
    kwargs_proc = {**kwargs_proc,**cfg['processing']}
    kwargs_pre = cfg['preprocessing']
    kwargs_sim = cfg['simulation']

    # update kwargs if subconfiguration file is provided
    if sub_config is None:
        print('Using processing configuration base.yml')
    else:
        print('Update processing configuration with %s' %(kwargs_proc['sub_config']))
        sub_config = sub_config if sub_config == '.yml' else sub_config+'.yml'
        if os.path.exists('%s/%s'%(config_path ,sub_config)):
            with open('%s/%s'%(config_path, sub_config), 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            if 'processing' in cfg: kwargs_proc = {**kwargs_proc,**cfg['processing']}
            if 'preprocessing' in cfg: kwargs_pre = {**kwargs_pre,**cfg['preprocessing']}
            if 'simulation' in cfg: kwargs_sim = {**kwargs_sim,**cfg['simulation']}
        else:
            print('Subconfiguration %s does not exist'%(sub_config))

    if log:
        print("Input: %s"                    % kwargs_proc['exp_in'])
        print("Output: %s"                   % kwargs_proc['exp_out'])
        print("Nemo_path: %s (default)"      % kwargs_proc['nemo_path'])
        print("Domain_path: %s (default)"    % kwargs_proc['domain_path'])
        print("Export_path: %s (default)"    % kwargs_proc['output_path'])
        print("Time_start: %s"               % kwargs_proc['time_start'])
        print("Time_stop: %s"                % kwargs_proc['time_stop'])
        print("Dynamics: %s"                 % kwargs_proc['dynamics'])
        print("Kinetic energy: %s"           % kwargs_proc['kinetic_energy'])
        print("Diagnose wind power: %s"      % kwargs_proc['wind_input'])
        print("Spinup: %s"                   % kwargs_proc['spinup'])
        print("Postprocess: %s"              % kwargs_proc['postprocessing'])

    return kwargs_proc, kwargs_pre, kwargs_sim