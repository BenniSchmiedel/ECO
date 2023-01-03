def update_plotstyle():
    import matplotlib
    
    matplotlib.rcParams['figure.figsize'] = [16,9]
    matplotlib.rcParams['axes.titlesize']=20
    matplotlib.rcParams['axes.labelsize']=18
    matplotlib.rcParams['lines.linewidth']=2.5
    matplotlib.rcParams['lines.markersize']=10
    matplotlib.rcParams['xtick.labelsize']=14
    matplotlib.rcParams['ytick.labelsize']=14
    matplotlib.rcParams['ytick.minor.visible']=True
    matplotlib.rcParams['ytick.direction']='inout'
    matplotlib.rcParams['ytick.major.size']=10
    matplotlib.rcParams['ytick.minor.size']=5
    matplotlib.rcParams['xtick.minor.visible']=True
    matplotlib.rcParams['xtick.direction']='inout'
    matplotlib.rcParams['xtick.major.size']=10
    matplotlib.rcParams['xtick.minor.size']=5
    
def cmap_OB():
    import matplotlib
    import numpy as np
    
    top = matplotlib.cm.get_cmap('OrRd', 128) # r means reversed version
    bottom = matplotlib.cm.get_cmap('Blues', 128)# combine it all
    newcolors = np.vstack((top(np.linspace(0, 1, 128)[::-1][10:-20]),
                       bottom(np.linspace(0, 1, 128))[10:-20]))# create a new colormaps with a name of OrangeBlue
    return matplotlib.colors.ListedColormap(newcolors, name='OrangeBlue')
    
def eos_config(kwargs,path=''):
    import numpy as np
    import argparse

    def clean_str(x): return x.replace('\t','').replace(' ','')
    eos_keys = [clean_str(i) for i in np.loadtxt(path+'eos.txt',dtype=str, delimiter=',',max_rows=1)]
    eos_id = np.where(np.array(eos_keys)==kwargs['eos'])[0][0]

    eos_parameters = np.loadtxt(path+'eos.txt',dtype="U30,f,f,f,f,f", delimiter=',', skiprows=1)
    eos_parameters = {clean_str(eos_parameters[i][0]):eos_parameters[i][eos_id] for i in range(len(eos_parameters))}
    return eos_parameters
