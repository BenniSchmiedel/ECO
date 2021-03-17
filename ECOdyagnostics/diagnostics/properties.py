import numpy as np

class Properties:

    def __init__(self,
                 grid_ops,
                 coords
                 ):
        self.ops = grid_ops
        self.ocean_properties={'thermal_expansion_lin': 2 * 10**-4,
                               'haline_expansion_lin' : 8 * 10**-4,
                               'thermal_expansion_2nd': 0.1655,
                               'haline_expansion_2nd' : 0.76554,
                               'cabbeling_T' : 5.952e-2,
                               'cabbeling_S' : 7.4914e-4,
                               'cabbeling_TS' : 2.4341e-3,
                               'thermobarric_T' : 1.497e-4,
                               'thermobarric_S' : 1.109e-5
                                }
        self.constants = {'g': 9.81,
                          'rho0': 1026}
        self.coords = coords

    def horizontal_anomaly(self, P, **maskargs):
        """
        Returns the horizontal anomalies of any property P.
        Horizontal anomalies are defined to be the anomaly compared to a horizontal mean.
        """
        P_hm = self.horizontal_mean(P, **maskargs)
        P_ha = P - P_hm

        return P_ha

    def horizontal_mean(self, P, **maskargs):
        """
        Returns the horizontal mean for any property P.
        """

        P_hm = self.ops.average(P, ['X', 'Y'], **maskargs)

        return P_hm

    def global_mean(self, P, **maskargs):
        """
        Returns the global mean for any property P.
        """

        P_gm = self.ops.average(P, ['X', 'Y', 'Z'], **maskargs)

        return P_gm

    def basin_mean(self, P ,boundary, **maskargs):
        """
        Returns the global mean for any property P.
        """

        P_gm = self.ops.average(P, ['X', 'Y', 'Z'], boundary=boundary, **maskargs)

        return P_gm

    def density(self, T, S, eos, T_ref=10, S_ref=35, rho0 = 1026):
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
        a0 = self.ocean_properties['thermal_expansion_2nd']
        b0 = self.ocean_properties['haline_expansion_2nd']
        lambda1 = self.ocean_properties['cabbeling_T']
        lambda2 = self.ocean_properties['cabbeling_S']
        nu = self.ocean_properties['cabbeling_TS']
        mu1 = self.ocean_properties['thermobarric_T']
        mu2 = self.ocean_properties['thermobarric_S']

        if eos == '2nd-eos':
            Z = self.coords['Z']
            dT = T - T_ref
            dS = S - S_ref
            rhd = (-a0 * (1 + lambda1 / 2 * dT + mu1 * Z) * dT
                   + b0 * (1 - lambda2 / 2 * dS - mu2 * Z) * dS - nu * dT * dS) /rho0

        elif eos == 's-eos':
            Z = self.coords['Z']
            dT = T - T_ref
            dS = S - S_ref
            rhd = (-a0 * (1 + lambda1 / 2 * dT + mu1 * Z) * dT
                   + b0 * dS ) / rho0

        elif eos == 'lin-eos':
            alpha = self.ocean_properties['thermal_expansion_lin']
            beta = self.ocean_properties['haline_expansion_lin']
            dT = T - T_ref
            dS = S - S_ref
            rhd = - alpha * dT + beta * dS



        return (rhd + 1) * rho0

    def dh_T(self, T, S, Z, eos, Z_r=0, T_ref=10, S_ref=35):
        """

        :param T:
        :param S:
        :param eos:
        :return:
        """

        g = self.constants['g']
        rho0 = self.constants['rho0']

        a0 = self.ocean_properties['thermal_expansion_2nd']
        b0 = self.ocean_properties['haline_expansion_2nd']
        lambda1 = self.ocean_properties['cabbeling_T']
        lambda2 = self.ocean_properties['cabbeling_S']
        nu = self.ocean_properties['cabbeling_TS']
        mu1 = self.ocean_properties['thermobarric_T']
        mu2 = self.ocean_properties['thermobarric_S']

        dT = T - T_ref
        dS = S - S_ref

        if eos == '2nd-eos':
            # rho = -a (1 +.5 lam1 T +
            dh_T = - g/rho0 * ((-a0*(1+lambda1*dT) - nu*dS)*(Z_r- Z) - 0.5*a0*mu1*(Z_r**2-Z**2) )
        elif eos == 's-eos':
            dh_T = - g/rho0 * (-a0*(1+lambda1*dT)*(Z_r- Z) - 0.5*a0*mu1*(Z_r**2-Z**2) )
        elif eos == 'lin-eos':
            alpha = self.ocean_properties['thermal_expansion_lin']
            dh_T =  alpha * g * (Z_r - Z)
        return dh_T

    def dh_S(self, T, S, Z, eos, Z_r=0, T_ref=10, S_ref=35):
        """

        :param T:
        :param S:
        :param eos:
        :return:
        """
        g = self.constants['g']
        rho0 = self.constants['rho0']

        a0 = self.ocean_properties['thermal_expansion_2nd']
        b0 = self.ocean_properties['haline_expansion_2nd']
        lambda1 = self.ocean_properties['cabbeling_T']
        lambda2 = self.ocean_properties['cabbeling_S']
        nu = self.ocean_properties['cabbeling_TS']
        mu1 = self.ocean_properties['thermobarric_T']
        mu2 = self.ocean_properties['thermobarric_S']

        dT = T - T_ref
        dS = S - S_ref
        if eos == '2nd-eos':
            dh_S = - g/rho0 * ((b0*(1-lambda2*dS) - nu*dT)*(Z_r- Z) - 0.5*b0*mu2*(Z_r**2-Z**2) )
        elif eos == 's-eos':
            dh_S = - g/rho0 * b0 * (Z_r - Z)
        elif eos == 'lin-eos':
            beta = self.ocean_properties['haline_expansion_lin']
            dh_S = - beta * g * (Z_r - Z)
        return dh_S

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