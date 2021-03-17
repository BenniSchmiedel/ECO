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
                pass
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
                    U_Ek.append(tau_x_k[ax] / (rho_0 * f[ax]))
                else:
                    U_Ek.append(tau_x_k[ax] / (rho_0 * f))
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
                             [self.ops.derivative(U_ek[0],axis='X'),self.ops.derivative(U_ek[1],axis='Y'),0]
                             )
        return w_Ek

    def conv(self,):
        pass
