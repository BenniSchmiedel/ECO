from .properties import Properties
from .transport import Transport
from .tracers import Tracers
import gsw

import numpy as np
import xarray as xr
import configparser

class Energetics:

    def __init__(self,
                 ops,
                 properties,
                 position_out='T',
                 interpolation_step='preceding',
                 eos='seos'
                 ):

        self.position_out = position_out
        self.interpolation_step = interpolation_step
        self.ops = ops
        self.properties = properties
        self.eos = eos

    def kinetic_energy(self, v, rho, output_position=None):
        """
        :param v: List of velocity vectors [u,v,w]
        :param rho: density field
        :param output_position: position on the grid for kinetic_energy
        :type v: list
        :type rho: xr.DataArray, np.ndarray
        :return: kineteic energy field (energy density)
        """

        if not output_position:
            output_position = self.position_out

        if all((pos==output_position for pos in self.ops._get_position(v))):
            v2sum = sum([v**2 for v in v])
        else:
            v2 = self.ops._shift_position([v**2 for v in v],output_position)
            if all((pos == output_position for pos in self.ops._get_position(v2))):
                v2sum = sum([v2 for v2 in v2])

        return 0.5 * v2sum * rho

    def potential_energy(self, rho, output_position=None, Z_r=0):
        if not output_position:
            output_position = self.position_out

        Z = self.properties.coords['Z']
        g = self.properties.constants['g']
        
        if self.ops._matching_pos(rho,output_position) or rho.dims == ('t',):
            return rho * g * (Z - Z_r)

        else:
            rho = self.ops._shift_position(rho, output_position)
            return rho * g * (Z - Z_r)

    def internal_energy(self, S, T, p):
        """
        Return the specific internal energy of seawater as a function of Conservative Temperature, Absolute Salinity and
        pressure. It uses the gsw_interal_energy-function from the teos10/gsw sea water package.

        :param S: Absoulte Salinity
        :param T: In-situ Temperature (will be converted to Conservative temperature)
        :param p: sea water pressure p = P - P0 with P the full pressure and P0 the atmosphericc pressure
        :return: specitic internal energy
        """
        CT = gsw.conversions.CT_from_pt(T,S)
        SA = S
        return gsw.internal_energy(SA,CT,p)

    def dynamic_enthalpy(self,  T, S, Z, eos=None, Z_r=0, T_ref=10, S_ref=35, output_position=None, **kwargs):
        if not output_position:
            output_position = self.position_out

        #z = self.properties.coords['Z']
        g = self.properties.constants['g']
        rho0 = self.properties.constants['rho0']

        a0 = self.properties.ocean_properties['thermal_expansion']
        b0 = self.properties.ocean_properties['haline_expansion']
        lambda1 = self.properties.ocean_properties['cabbeling_T']
        lambda2 = self.properties.ocean_properties['cabbeling_S']
        nu = self.properties.ocean_properties['cabbeling_TS']
        mu1 = self.properties.ocean_properties['thermobarric_T']
        mu2 = self.properties.ocean_properties['thermobarric_S']
        dT = T - T_ref
        dS = S - S_ref

        mask = self.ops._get_mask(T, **kwargs)

        Z = Z * mask
        
        if not eos:
            eos = self.eos
        if eos == '2eos':
            def h(dT, dS, Z):
                h = - g/rho0 * ((-a0 * (1+lambda1/2*dT) * dT + b0 * (1+lambda2/2*dS) * dS - nu*dT*dS) * (Z_r - Z)
                        + 0.5 * (-a0*mu1*dT + b0*mu2*dS) * (Z_r**2 - Z**2))
                return h

        elif eos == 'seos':
            def h(dT, dS, Z):
                h = - g/rho0 * ((-a0 * (1+lambda1/2*dT) * dT + b0*dS) * (Z_r - Z)
                        - 0.5*a0*mu1*dT * (Z_r**2 - Z**2))
                return h

        elif eos == 'leos':
            def h(dT, dS, Z):
                alpha = self.properties.ocean_properties['thermal_expansion_lin']
                beta = self.properties.ocean_properties['haline_expansion_lin']

                h = - ( - alpha*dT + beta*dS ) * (Z_r-Z) * g
                return h


        """if 't' in T.dims:
            h = np.zeros(T.shape)#b.copy(data=np.zeros(b.shape()))
        else:
            h = np.zeros(T.shape)"""
        """for x in range(self.properties.coords['X'].shape[0]):
            for y in range(self.properties.coords['Y'].shape[0]):
                for z in range(self.properties.coords['Z'].shape[0]):

                    if T.dims[0]=='t':
                        if np.all(np.isnan(T[:,z,y,x])):
                            pass
                        else:
                            h[:,z,y,x] = b(T[:,z,y,x],S[:,z,y,x],Z[:,z,y,x])#.sum(self.ops.grid._get_dims_from_axis(T,'Z')) #* Z[:,z,y,x]
                    elif T.dims[0] == 'z':
                        if np.all(np.isnan(T[z, y, x])):
                            pass
                        else:
                            h[z, y, x] = b(T[z, y, x], S[z, y, x], Z[z, y, x])#.sum(
                                #self.ops.grid._get_dims_from_axis(T, 'Z'))  # * Z[:,z,y,x]"""
        #h = T.copy(data= h )
        h = h(dT,dS,Z)
        if self.ops._matching_pos(h, output_position):
            return h

        else:
            return self.ops._shift_position(h, output_position)

    def center_of_gravity_PE(self, rho, Z_r=0, boussinesq=True, **kwargs):
        """
        Computes the position of the center of mass.
        Returns the location for each axis. Operates on a 3D grid.
        """
        try:
            pos = self.ops._get_position(rho)
        except:
            pos = 'T'
                
        dV = self.ops._get_metric_by_pos(['X','Y','Z'], pos, combine = True)[0]
        # Compute Sum(PE*dV) and Sum(rho*dV)

        mask = self.ops._get_mask(dV, **kwargs)

        rho = rho * mask

        if boussinesq:
            mass_full = (self.properties.constants['rho0'] * dV * mask).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z')))
        else:
            mass_full = (rho * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z')))

        PE = self.potential_energy(rho, Z_r=Z_r)
        TPE_full = (PE * dV * mask).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z')))

        g = self.properties.constants['g']

        # Return center of gravity
        return TPE_full / (g * mass_full)

    def center_of_gravity_h(self, T, S, z, eos, Z_r=0, T_ref=10, S_ref=35, output_position='T', **kwargs):
        """
        Computes the position of the center of mass.
        Returns the location for each axis. Operates on a 1D or 3D grid.
        """
        #Position check
        try:
            pos = self.ops._get_position(T)
        except:
            pos = 'T'
            
        if self.ops._matching_pos(z,'T'):
            pass
        else:
            Exception('depth has the wrong position {}. Please provide a depth on T position.'.format(self.ops._get_position(z)))

        #Compute dynamic_enthalpy
        h = self.dynamic_enthalpy(T, S, z, eos=eos, Z_r=Z_r, T_ref=T_ref, S_ref=S_ref, output_position=output_position, **kwargs)
        
        #Check dimension, 1D and 3D valid
        skip=0
        if 't' in h.dims:
            skip=1
        
        if len(h.dims[skip:]) == 1:
            return self.ops.average(h, ['Z'],**kwargs) / self.properties.constants['g']
        elif len(h.dims[skip:])==3:
            return self.ops.average(h, ['X','Y','Z'],**kwargs) / self.properties.constants['g']
        else:
            raise Exception("Data has to be 1D or 3D in space, please check dimensions or time variable name, expected: 't'")
            

    def center_of_gravity_eta(self, eta, rho, eta_r=0, boussinesq=False, **kwargs):
        """
        Computes the position of the center of mass.
        Returns the location for each axis. Operates on a 3D grid.
        """
        
        try:
            pos = self.ops._get_position(rho)
        except:
            pos = 'T'
            
        rho0 = self.properties.constants['rho0']
        dr = self.ops._get_metric_by_pos(['X', 'Y', 'Z'], pos)
        dA_s = dr[0]*dr[1]
        del dr

        mask = self.ops._get_mask(rho, **kwargs)

        dV = self.ops._get_metric_by_pos(['X','Y','Z'], pos, combine = True)[0] * mask
        if boussinesq:
            mass_full = (rho0 * dV).sum(self.ops._get_dims(dV))
        else:
            mass_full = (rho  * dV).sum(self.ops._get_dims(rho))
        del dV
        
        zg_eta = (0.5 * rho0 * dA_s* (eta**2-eta_r**2)).sum(self.ops.grid._get_dims_from_axis(eta, ('X', 'Y'))) / mass_full
        #self.ops.average(h, ['X','Y','Z'],mask='nan') / g

        return zg_eta

    def center_of_gravity_classical(self, rho, boussinesq=False, coords=None, **kwargs):
        """
        Computes the position of the center of mass.
        Returns the location for each axis. Operates on a 1D or 3D grid.
        """
        try:
            pos = self.ops._get_position(rho)
        except:
            pos = 'T'
        
        ### Check Case, treat 0D as 1D
        skip=0
        if 't' in rho.dims:
            skip=1
        
        if len(rho.dims[skip:]) in [1,0]:
            case = 1
        elif len(rho.dims[skip:])==3:
            case = 3
        else:
            raise Exception("Data has to be 1D or 3D in space, please check dimensions or time variable name ('t')")
            
        ### Get metrics & coordinate
        if case == 1:
            dV = self.ops._get_metric_by_pos('Z', pos, case = 'smallest')[0]
        elif case == 3:
            dV = self.ops._get_metric_by_pos(['X', 'Y', 'Z'], pos, combine = True)[0]
                    
        if coords is None:
            if case == 1:
                z = self.properties.coords['Z']
                
            elif case == 3:
                coords = [self.properties.coords[coord] for coord in ['X','Y','Z']]
                x, y, z = coords
                
        # Compute
        mask = self.ops._get_mask(dV, **kwargs)
        rho = rho * mask

        if case == 1:
            if boussinesq:
                rho0 = self.properties.constants['rho0']
                mass_full = (rho0*dV*mask).sum(self.ops.grid._get_dims_from_axis(dV, ('Z')))
            else:
                mass_full = (rho * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('Z')))
                
            R_z = (rho * z * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('Z'))) / mass_full
            
        if case == 3:
            if boussinesq:
                rho0 = self.properties.constants['rho0']
                mass_full = (rho0*dV*mask).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z')))
            else:
                mass_full = (rho * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z')))
                
            R_z = (rho * z * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full

        #mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('X', 'Y')))
        #mass_zonal = mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('X', 'Z')))
        #mass_meridional = mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('Y', 'Z')))

        # Get center of mass
        #R_x = (rho * x * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full
        #R_y = (rho * y * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full
        #R_x = (mass_meridional * x).sum(self.ops.grid._get_dims_from_axis(mass_c, ('X'))) / mass_full
        #R_y = (mass_zonal * y).sum(self.ops.grid._get_dims_from_axis(mass_c, ('Y'))) / mass_full
        #R_z = (mass_horizontal * z).sum(self.ops.grid._get_dims_from_axis(mass_c, 'Z')) / mass_full

        return R_z #[R_x, R_y, R_z]

class Energetics_trends:

    def __init__(self,
                 ops,
                 properties,
                 trends_file,
                 position_out='T',
                 interpolation_step='preceding',
                 eos='seos'
                 ):
        self.position_out = position_out
        self.interpolation_step = interpolation_step
        self.eos = eos
        self.ops = ops
        self.properties = properties
        self.transport = Transport(ops)
        #if trends_file=='default':
        #    trends_file=''
        self.tracers = Tracers(ops, properties, trends_file)

        config = configparser.ConfigParser()
        config.read(trends_file)
        self.trends_ke = list()
        self.trends_pe = list()
        for process in config.options('ke_processes'):
            if eval(config['ke_processes'][process]):
                self.trends_ke.append(process)
        for process in config.options('pe_processes'):
            if eval(config['pe_processes'][process]):
                self.trends_pe.append(process)


    def kinetic_energy_trend(self, v, rho, **kwargs):

        ke_trends = dict()
        for i in list(kwargs.keys()):
            if i not in self.trends_ke:
                print('Process is deactivated or invalid keyword {}'.format(i))
        for i in self.trends_ke:
            if i in self.trends_ke:
                ke_trends[i] = kwargs.get(i,None)
        loc = locals()
        #ke_trend =sum([getattr(self.transport, i)(*[loc[arg] for arg in getattr(self.transport, i).__code__.co_varnames])
        #             if processes[i] is None else processes[i] for i in list(processes.keys())])
        ke_trends = {i: getattr(self.transport, i)(*[loc[arg] for arg in getattr(self.transport, i).__code__.co_varnames])
                     if ke_trends[i] is None else ke_trends[i] for i in ke_trends}
        return sum(ke_trends.values()), ke_trends

    def potential_energy_trend(self, rho, **kwargs):

        processes_keys = ['adv_m', 'adv_z', 'adv_v', 'conv', 'diff_h', 'diff_v', 'bbl',
                          'geoth', 'surf_runoff', 'solar_pen', 'relax']
        pe_trends = dict()
        for i in list(kwargs.keys()):
            if i not in processes_keys:
                print('Invalid keyword {}'.format(i))
            elif i in processes_keys and i not in self.trends_pe:
                print('Process {} is deactivated but still given'.format(i))
        for i in self.trends_ke:
            pe_trends[i] = kwargs.get(i, None)
        loc = locals()
        pe_trends = {
            i: getattr(self.transport, i)(*[loc[arg] for arg in getattr(self.transport, i).__code__.co_varnames])
            if pe_trends[i] is None else pe_trends[i] for i in pe_trends}
        return sum(pe_trends.values()), pe_trends

    def internal_energy_trend(self):
        pass

    def center_of_gravity_h_trend(self, rho, v, T, S, Z, eos = None, Z_r=0, T_ref=10, S_ref=35,
                                  T_trend=None, T_trends=None,
                                  S_trend=None, S_trends=None,
                                  C_trend=None, C_trends=None,
                                  boundary=None, Vmask=None, **kwargs):

        ## Get T and S trends
        if T_trend is None and T_trends is None:
            raise Exception('Please provide either the total T_trend or the processes as dictionary named T_trends')
        elif not T_trends is None:
            T_trends = self.tracers.temperature_trend(**T_trends)
            T_trend = T_trends[0]

        if S_trend is None and S_trends is None:
            raise Exception('Please provide either the total S_trend or the processes as dictionary named S_trends')
        elif not S_trends is None:
            S_trends = self.tracers.salinity_trend(**S_trends)
            S_trend = S_trends[0]

        if not eos:
            eos = self.eos
        ## Get derivatives of h in T and S
        dh_T = self.properties.dh_T(T, S, Z, eos, Z_r=Z_r, T_ref=T_ref, S_ref=S_ref)
        dh_S = self.properties.dh_S(T, S, Z, eos, Z_r=Z_r, T_ref=T_ref, S_ref=S_ref)

        g=self.properties.constants['g']
        rho0=self.properties.constants['rho0']

        ## Get Conversion
        if C_trend is None and C_trends is None:
            raise Exception('Please provide either the total C_trend or the processes as dictionary named C_trends')
        elif not C_trends is None:
            C_trends = self.kinetic_energy_trend(v, rho, **C_trends)
            if C_trend is None:
                C_trend = C_trends[0]

        zg_trend = 1/g * (self.properties.global_mean(dh_T*T_trend, Vmask=Vmask, **kwargs) +
                          self.properties.global_mean(dh_S*S_trend, Vmask=Vmask, **kwargs) +
                          self.properties.global_mean(C_trend/rho0, Vmask=Vmask, **kwargs))

        zg_trends = []
        if not T_trends is None:
            zg_trends.append({i: T_trends[1][i] * dh_T / g for i in T_trends[1]})
        else:
            zg_trends.append({'T_tot':dh_T * T_trend / g})

        if not S_trends is None:
            zg_trends.append({i: S_trends[1][i] * dh_S / g for i in S_trends[1]})
        else:
            zg_trends.append({'S_tot':dh_S * S_trend / g})

        if not C_trends is None:
            zg_trends.append({i:  C_trends[1][i] / g / rho0 for i in C_trends[1]})
        else:
            zg_trends.append({'convp2k': C_trend / g / rho0})

        return zg_trend, zg_trends
