import numpy as np

class Transport:

    def __init__(self, grid_ops, position_out='F', interpolation_step='consecutive'):
        self.position_out = position_out
        self.interpolation_step = 'consecutive'
        self.ops = grid_ops

    def ekman_transport(self, tau, f_in, rho_0=1026, output_position=None, f_interpolation=False):
        """
        Computes the ekman transport from the surface wind stress tau.

        U_ek = -(k x tau) / (rho_0 * f)
        with k the vertical unit and tau the windstress.

        :param tau:   windstress vector 3D
        :param rho_0:   surface density
        :param g:   gravitational acceleration
        :param output_position:
        :return:
        """

        # Calculate crossproduct k x tau
        k = [0, 0, 1]
        tau_x_k = [0]*3
        tau_x_k[0] = k[1] * tau[2] - k[2] * tau[1]
        tau_x_k[1] = k[2] * tau[0] - k[0] * tau[2]
        tau_x_k[2] = k[0] * tau[1] - k[1] * tau[0]

        # Adjust f if necessary or desired to match dimensions
        # Skip the z-coordinate since it is zero but would mess up dimension comparison
        if f_interpolation:
            if self.ops._matching_dim(tau_x_k, f_in, skip1 = 2, skip2 =2) == True:
                f=f_in
            else:
                f = [np.nan] * 3
                for i in  range(2):
                    f[i] = self.ops._shift_position(f_in, self.ops._get_position(tau_x_k[i]))
        else:
            f=f_in

        # Calculate U_Ek = tau_x_k /(rho*f), but check if f and tau_x_k have the same dimensions
        # Skip the z-coordinate since it is zero but would mess up dimension comparison
        if self.ops._matching_dim(tau_x_k, f, skip1=2, skip2=2) == True:
            U_Ek = []
            for ax in range(3):
                if type(f) is list:
                    U_Ek.append(- tau_x_k[ax] / (rho_0 * f[ax]))
                else:
                    U_Ek.append(- tau_x_k[ax] / (rho_0 * f))
        else:
            raise Exception("""Dimensions of tau x k and f do not match %s - %s
                """ % (self.ops._get_dims(tau_x_k), self.ops._get_dims(f)))

        return U_Ek

    def ekman_velocity(self, U_ek):
        """
        Computes the ekman transport from the surface wind stress tau.

        w_Ek = Grad_h[ -(k x tau)/(rho_o * f) ] ) Grad_h [ U_Ek ]
        with k the vertical unit and tau the windstress.

        :return:
        """
        # Calculate horizontal gradient of ekman transport
        w_Ek = self.ops.dot( [1,1,0],
                             [self.ops.grid.derivative(U_ek[0], 'X',boundary='fill',fill_value=0),
                              self.ops.grid.derivative(U_ek[1], 'Y',boundary='fill',fill_value=0), 0]
                             #[self.ops.derivative(U_ek[0],axis='X'),self.ops.derivative(U_ek[1],axis='Y'),0]
                             )
        return w_Ek

    def moc_streamfunction(self, v, Vmask=None, **maskargs):

        #Integrate over longitudes
        v_x = self.ops.grid.integrate(v, axis='X')

        #Cumint in depth (bottom to top)
        try:
            sf = self.ops.grid.cumint(v_x, axis='Z' , boundary="fill", fill_value=0)
        except:
            z_t = self.ops.average(self.ops.grid.get_metric(v, 'Z'), ['X','Y'] , Vmask=Vmask, **maskargs )
            sf = self.ops.grid.cumsum(v_x*z_t, axis='Z', boundary="fill", fill_value=0)

        sf = sf - sf.isel({self.ops.grid._get_dims_from_axis(sf, 'Z')[0]: -1})

        return sf

    def conv(self,):
        pass

    def enhanced_diffusion_mask(self, trd_zdf, case, case_data, condition=1e-12, invert=False):
        """
        Returns a mask that masks regions affected by enhanced vertical diffusion which mimics convection.
        Convection is applied when the condition N2 < -1e-12 is true, where N2 is the Brunt-Väisälä-frequency.
        Alternatively the mask is generated based on the buoyancy forcing, where it is applied when buoyancy forcing is negative.
        :param trd_zdf:
        :param T:
        :param S:
        :param alpha:
        :param beta:
        :return:
        """

        mask = trd_zdf.copy(data=np.ones(trd_zdf.shape))
        if case == 'Brunt_Vaisala':
            if not invert:
                mask = mask.where(case_data <= condition, other=np.nan)
            elif invert:
                mask = mask.where(case_data > condition, other=np.nan)

        elif case == 'Buoyancy_forcing':
            if not invert:
                mask = mask.where(case_data <= condition, other=np.nan)
            elif invert:
                mask = mask.where(case_data > condition, other=np.nan)

        return mask

    def compute_moc(self, v, X="X", Z="Z"):
        psi = self._compute_moc_with_v_at_cst_depth(v, X="X", Z="Z")
        return psi

    def _compute_moc_with_v_at_cst_depth(self, v, X="X", Z="Z"):
        """
        Compute the meridional overturning streamfunction.
        """
        v_x_dx = self.ops.grid.integrate(v, axis=X)  # (vo_to * domcfg_to['y_f_dif']).sum(dim='x_c')
        # integrate from top to bot
        psi = self.ops.grid.cumint(v_x_dx, axis=Z, boundary="fill", fill_value=0) * 1e-6
        # convert -> from bot to top
        psi = psi - psi.isel({self.ops.grid._get_dims_from_axis(psi,Z)[0]: -1})
        return psi
