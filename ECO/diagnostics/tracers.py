from .transport import Transport
import gsw
import configparser

import numpy as np
import xarray as xr

class Tracers:

    def __init__(self,
                 grid_ops,
                 properties,
                 trends,
                 position_out='T',
                 interpolation_step='preceding',
                 ):

        self.position_out = position_out
        self.interpolation_step = interpolation_step
        self.grid_ops = grid_ops
        self.properties = properties
        self.transport = Transport(grid_ops)

        #config = configparser.ConfigParser()
        #config.read(trends_file)
        self.T_trends = trends['T_trends']#list()
        self.S_trends = trends['S_trends']#= list()
        #for process in config.options('T_processes'):
        #    if eval(config['T_processes'][process]):
        #        self.trends_T.append(process)
        #for process in config.options('S_processes'):
        #    if eval(config['S_processes'][process]):
        #        self.trends_S.append(process)

    def temperature_trend(self, **kwargs):
        
        T_trends = dict()
        for i in list(kwargs.keys()):
            if i not in self.T_trends:
                print('Process is deactivated or invalid keyword {}'.format(i))
        for i in self.T_trends:
            T_trends[i] = kwargs.get(i, None)
        loc = locals()
        T_trends = {
            i: getattr(self.transport, i)(*[loc[arg] for arg in getattr(self.transport, i).__code__.co_varnames])
            if T_trends[i] is None else T_trends[i] for i in T_trends}
        for i in T_trends:
            if np.all(np.isnan(T_trends[i])):
                T_trends[i]=0
                print(i+' is nan everywhere, process was removed.')
        return sum(T_trends.values()), T_trends

    def salinity_trend(self, **kwargs):

        S_trends = dict()
        for i in list(kwargs.keys()):
            if i not in self.S_trends:
                print('Process is deactivated or invalid keyword {}'.format(i))
        for i in self.S_trends:
            S_trends[i] = kwargs.get(i, None)
        loc = locals()
        S_trends = {
            i: getattr(self.transport, i)(*[loc[arg] for arg in getattr(self.transport, i).__code__.co_varnames])
            if S_trends[i] is None else S_trends[i] for i in S_trends}
        for i in S_trends:
            if np.all(np.isnan(S_trends[i])):
                S_trends[i]=0
                print(i+' is nan everywhere, process was removed.')
        return sum(S_trends.values()), S_trends
