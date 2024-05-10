import numpy as np

class Properties:

    def __init__(self,
                 grid_ops,
                 coords,
                 eos_properties=None
                 ):
        self.ops = grid_ops
        if not eos_properties:
            self.eos_properties={'thermal_expansion': 0.1655,
                                   'haline_expansion' : 0.76554,
                                   'cabbeling_T' : 5.952e-2,
                                   'cabbeling_S' : 7.4914e-4,
                                   'cabbeling_TS' : 2.4341e-3,
                                   'thermobaric_T' : -1.497e-4,
                                   'thermobaric_S' : -1.109e-5
                                    }
        else:
            self.eos_properties=eos_properties
            
        self.constants = {'g': 9.81,
                          'rho0': 1026}
        self.coords = coords

    def horizontal_anomaly(self, P, Vmask=None, **kwargs):
        """
        Returns the horizontal anomalies of any property P.
        Horizontal anomalies are defined to be the anomaly compared to a horizontal mean.
        """
        P_hm = self.horizontal_mean(P, Vmask=Vmask, **kwargs)

        return  P - P_hm

    def horizontal_mean(self, P, Vmask=None, **kwargs):
        """
        Returns the horizontal mean for any property P.
        """

        P_hm = self.ops.average(P, ['X', 'Y'], Vmask=Vmask, **kwargs)

        return P_hm

    def global_mean(self, P, Vmask=None, **kwargs):
        """
        Returns the global mean for any property P.
        """

        P_gm = self.ops.average(P, ['X', 'Y', 'Z'], Vmask=Vmask, **kwargs)

        return P_gm

    def density(self, T, S, Z, T_ref=10, S_ref=35, rho0 = 1026, out = None, **kwargs):
        """
        Returns the non-linear sea water density defined by Roquet et al 2015a

        rhd(T,S,Z) = [ -a0*( 1+ lambda1/2*dT+ mu1*Z )*dT
                                        + b0*( 1+ lambda2/2*dT+ mu2*Z )*dS - nu*dT*dS] / rau0
        with dT = T - 10 and dS = S - 35 and depth Z
        with the following coefficients :
            thermal exp. coef.    rn_a0      =   0.16550
            saline  cont. coef.   rn_b0      =   0.76554
            cabbeling coef.       rn_lambda1 =   5.95200E-002
            cabbeling coef.       rn_lambda2 =   7.49140E-004
            thermobar. coef.      rn_mu1     =   1.49700E-004
            thermobar. coef.      rn_mu2     =   1.10900E-005
            2nd cabbel. coef.     rn_nu      =   2.43410E-003
        """

        a0 = self.eos_properties['thermal_expansion']
        b0 = self.eos_properties['haline_expansion']
        lambda1 = self.eos_properties['cabbeling_T']
        lambda2 = self.eos_properties['cabbeling_S']
        nu = self.eos_properties['cabbeling_TS']
        mu1 = self.eos_properties['thermobaric_T']
        mu2 = self.eos_properties['thermobaric_S']

        mask = self.ops._get_mask(T, **kwargs)    
        Z = Z * mask
        dT = (T - T_ref)*mask
        dS = (S - S_ref)*mask

        rhd = (-a0 * (1 + lambda1 / 2 * dT + mu1 * Z) * dT
                   + b0 * (1 - lambda2 / 2 * dS - mu2 * Z) * dS - nu * dT * dS) /rho0

        if out=='horizontal':
            rhd = self.horizontal_mean(rhd, **kwargs)
        elif out=='global':
            rhd = self.global_mean(rhd, **kwargs)

        return (rhd + 1) * rho0

    def brunt_vaisala(self, T, S, eos='seos'):
        """

        :param T:
        :param S:
        :return:
        """

        g = self.constants['g']
        e3 = self.ops._get_metric_by_pos('Z', 'T')[0]
        if self.ops._get_position(S) != 'W':
            S = self.ops._shift_position(S, 'W')
        if self.ops._get_position(T) != 'W':
            T = self.ops._shift_position(T, 'W')
        
        if eos=='seos':
            alpha = self.eos_properties['thermal_expansion']
            beta = self.eos_properties['haline_expansion']
        elif eos=='2eos':
            alpha = self.eos_properties['thermal_expansion']
            beta = self.eos_properties['haline_expansion'] 
        elif eos=='leos':
            alpha = self.eos_properties['thermal_expansion_lin']
            beta = self.eos_properties['haline_expansion_lin']

        return g/e3 *(beta * self.ops.grid.diff(S, 'Z', boundary='fill', fill_value=0) -
                      alpha * self.ops.grid.diff(T, 'Z', boundary='fill', fill_value=0))

    def dh_T(self, T, S, Z, Z_r=0, T_ref=10, S_ref=35):
        """
        :param T:
        :param S:
        :param eos:
        :return:
        """
        g = self.constants['g']
        rho0 = self.constants['rho0']

        a0 = self.eos_properties['thermal_expansion']
        l1 = self.eos_properties['cabbeling_T']
        nu = self.eos_properties['cabbeling_TS']
        mu1 = self.eos_properties['thermobaric_T']

        dT = T - T_ref
        dS = S - S_ref

        return - g/rho0 * ((-a0*(1+l1*dT) - nu*dS)*(Z_r- Z) - 0.5*a0*mu1*(Z_r**2-Z**2) )

    def dh_S(self, T, S, Z, Z_r=0, T_ref=10, S_ref=35):
        """
        :param T:
        :param S:
        :param eos:
        :return:
        """
        g = self.constants['g']
        rho0 = self.constants['rho0']

        b0 = self.eos_properties['haline_expansion']
        l2 = self.eos_properties['cabbeling_S']
        nu = self.eos_properties['cabbeling_TS']
        mu2 = self.eos_properties['thermobaric_S']

        dT = T - T_ref
        dS = S - S_ref

        return - g/rho0 * ((b0*(1-l2*dS) - nu*dT)*(Z_r- Z) - 0.5*b0*mu2*(Z_r**2-Z**2))

    def dh_T_new(self, T, S, Z, Z_r=0, T_ref=10, S_ref=35):
        """
        :param T:
        :param S:
        :param eos:
        :return:
        """
        g = self.constants['g']
        rho0 = self.constants['rho0']

        a0 = self.eos_properties['thermal_expansion']
        l1 = self.eos_properties['cabbeling_T']
        nu = self.eos_properties['cabbeling_TS']
        mu1 = self.eos_properties['thermobaric_T']

        dT = T - T_ref
        dS = S - S_ref

        return g/rho0 * ((-a0*(1+l1*dT) - nu*dS)*Z - 0.5*a0*mu1*Z**2 )

    def dh_S_new(self, T, S, Z, Z_r=0, T_ref=10, S_ref=35):
        """
        :param T:
        :param S:
        :param eos:
        :return:
        """
        g = self.constants['g']
        rho0 = self.constants['rho0']

        b0 = self.eos_properties['haline_expansion']
        l2 = self.eos_properties['cabbeling_S']
        nu = self.eos_properties['cabbeling_TS']
        mu2 = self.eos_properties['thermobaric_S']

        dT = T - T_ref
        dS = S - S_ref

        return g/rho0 * ((b0*(1-l2*dS) - nu*dT)*Z - 0.5*b0*mu2*Z**2)
    
    def dh_TZ(self, T, S, Z, T_ref=10, S_ref=35):

        g = self.constants['g']
        rho0 = self.constants['rho0']

        a0 = self.eos_properties['thermal_expansion']
        l1 = self.eos_properties['cabbeling_T']
        nu = self.eos_properties['cabbeling_TS']
        mu1= self.eos_properties['thermobaric_T']

        dT = (T-T_ref)
        dS = (S-S_ref)
        
        return (- g/rho0 * (-(a0*(1+l1*dT) + nu*dS) + a0*mu1*Z ))

    def dh_SZ(self, T, S, Z, T_ref=10, S_ref=35):

        g = self.constants['g']
        rho0 = self.constants['rho0']

        b0 = self.eos_properties['haline_expansion']
        l2 = self.eos_properties['cabbeling_S']
        nu = self.eos_properties['cabbeling_TS']
        mu2 = self.eos_properties['thermobaric_S']

        dT = (T-T_ref)
        dS = (S-S_ref)
        
        return (- g/rho0 * ((-b0*(1-l2*dS) + nu*dT) + b0*mu2*Z ))

    def dh_TT(self, Z, Z_r=0):
        
        g = self.constants['g']
        rho0 = self.constants['rho0']
        a0 = self.eos_properties['thermal_expansion']
        l1 = self.eos_properties['cabbeling_T']
        
        return - g/rho0 * (-a0*l1*(Z_r- Z))
    
    def dh_SS(self, Z, Z_r=0):
        
        g = self.constants['g']
        rho0 = self.constants['rho0']
        b0 = self.eos_properties['haline_expansion']
        l2 = self.eos_properties['cabbeling_S']
        
        return - g/rho0 * (b0*l2*(Z_r- Z))

    def center_of_mass(self, rho, boussinesq=False, coords=None, mask=None):
        """
        Computes the position of the center of mass.
        Returns the location for each axis. Operates on a 3D grid.
        """
        try:
            pos = self.ops._get_position(rho)
        except:
            pos = 'T'
        dr = self.ops._get_metric_by_pos(['X', 'Y', 'Z'], pos)
        dV = np.prod(dr)
        if coords is None:
            coords = [self.coords[coord] for coord in ['X','Y','Z']]
        x, y, z = coords
        # Compute Sum ( rho * V * D)
        if mask is None:
            mask = 1
        rho = rho * mask
        if boussinesq:
            rho0 = self.constants['rho0']
            mass_full = (rho0*dV*mask).sum(self.ops.grid._get_dims_from_axis(rho, ('X', 'Y', 'Z')))
        else:
            mass_full = (rho * dV).sum(self.ops.grid._get_dims_from_axis(rho, ('X', 'Y', 'Z')))
        #mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('X', 'Y')))
        #mass_zonal = mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('X', 'Z')))
        #mass_meridional = mass_c.sum(self.ops.grid._get_dims_from_axis(mass_c, ('Y', 'Z')))

        # Get center of mass
        R_x = (rho * x * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full
        R_y = (rho * y * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full
        R_z = (rho * z * dV).sum(self.ops.grid._get_dims_from_axis(dV, ('X', 'Y', 'Z'))) / mass_full
        #R_x = (mass_meridional * x).sum(self.ops.grid._get_dims_from_axis(mass_c, ('X'))) / mass_full
        #R_y = (mass_zonal * y).sum(self.ops.grid._get_dims_from_axis(mass_c, ('Y'))) / mass_full
        #R_z = (mass_horizontal * z).sum(self.ops.grid._get_dims_from_axis(mass_c, 'Z')) / mass_full

        return R_z#[R_x, R_y, R_z]
