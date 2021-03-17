from .transport import Transport

class Power:

    def __init__(self,
                 grid_ops,
                 position_out='T',
                 interpolation_step='preceding',
                 ):

        self.position_out = position_out
        self.interpolation_step = interpolation_step
        self.ops = grid_ops
        self.transport = Transport(grid_ops)

    def P_taug(self, tau_x, tau_y, eta, f, rho_0=1026, g=9.81, output_position=None, f_interpolation=False, **kwargs):
        """
        Computes the direct rate of power input into the geostrophic circulation.

        It is calculated from the surface wind stress (tau_x, tau_y) and the sea surface height eta with:
        P_tau = rho_0 * g * [ U_ek * Grad(eta) ] = rho_0 * g * [ ( (k x tau) / (g * f) ) * Grad(eta) ]

        Computation is performed with interpolation of the data variables to give the result on a specified
        grid position.
        Note: The windstress z-coordinate has a default 0-input to stay consistent with the crossproduct operator.

        When interpolation is done preceding to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} --interp--> 'T' ===>>> U_Ek {['T','T',__]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} --interp--> 'F' ===>>> U_Ek {['F','F',__]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        When interpolation is done consecutive to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['T','T', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['F','F', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        :param tau_x:           wind stress x-direction
        :param tau_y:           wind stress y-direction
        :param eta:             sea surface height eta
        :param f:               coriolis parameter
        :param rho_0:           surface water density
        :param g:               gravitational acceleration
        :param position_out:    position on grid of output P_taug

        :return:                P_taug
        """
        tr = self.transport
        tau = [tau_x, tau_y, 0]
        ax_order = ('X', 'Y', 'Z')

        if output_position is None:
            output_position = self.position_out


        if self.interpolation_step == 'preceding':
            # Interpolate tau to position_out
            tau=self.ops._shift_position(tau,output_position,elements=[True,True,False])
            #Calculate U_ek
            U_Ek = self.transport.ekman_transport(tau, f, output_position=output_position, f_interpolation=f_interpolation)
            # Compute Grad(eta), fill z-coordinate with 0
            grad_eta = []
            for i in range(2):
                grad_eta.append(self.ops.derivative(eta, axis=ax_order[i], **kwargs))
            grad_eta.append(0)
            # Interpolate Grad(eta) to position_out
            grad_eta=self.ops._shift_position(grad_eta,output_position,elements=[True,True,False])


        elif self.interpolation_step == 'consecutive':
            # Calculate U_ek
            U_Ek = self.transport.ekman_transport(tau, f, f_interpolation=f_interpolation)

            print([U_Ek[i].dims for i in range(2)])
            # Interpolate tU_ek to position_out
            U_Ek = self.ops._shift_position(U_Ek, output_position, elements=[True, True, False])
            # Compute Grad(eta), fill z-coordinate with 0
            grad_eta = []
            for i in range(2):
                grad_eta.append(self.ops.derivative(eta, axis=ax_order[i], **kwargs))
            grad_eta.append(0)
            # Interpolate Grad(eta) to position_out
            grad_eta = self.ops._shift_position(grad_eta, output_position, elements=[True, True, False])



        p_tau = rho_0 * g * self.ops.dot(U_Ek, grad_eta)

        return p_tau

    def P_down(self, tau_x, tau_y, eta, f, rho_0=1026, g=9.81, output_position=None, f_interpolation=True, **kwargs):
        """
        Computes the downward flux into the geostrophic circulation.

        It is calculated from the surface wind stress (tau_x, tau_y) and the sea surface height eta with:
        P_down = -rho_0 * g * eta * Grad_h( U_ek ) = Grad x [tau / (rho_0 * f) ]

        Computation is performed with interpolation of the data variables to give the result on a specified
        grid position.
        Note: The windstress z-coordinate has a default 0-input to stay consistent with the crossproduct operator.

        When interpolation is done preceding to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau/rho*f {'U' and 'V'}  ===>>> k x tau/rho*f {['VW','UW', __ ]} --interp--> {['U','V', __ ]}
        ===>>> w_Ek {'T'}
        eta {'T'}
        ===>>> P_down {'T'}

        'F'-point:
        tau/rho*f {'U' and 'V'} ===>>> k x tau/rho*f {['VW','UW', __ ]} --interp--> {['V','U', __ ]}
        ===>>> w_Ek {'F'}
        eta {'T'} --interp--> eta {'F'}
        ===>>> P_down {'F'}

        When interpolation is done consecutive to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['T','T', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['F','F', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        :param tau_x:           wind stress x-direction
        :param tau_y:           wind stress y-direction
        :param eta:             sea surface height eta
        :param f:               coriolis parameter
        :param rho_0:           surface water density
        :param g:               gravitational acceleration
        :param position_out:    position on grid of output P_taug

        :return:                P_taug
        """
        tau = [tau_x, tau_y, 0]

        if output_position is None:
            output_position = self.position_out

        # Get positions of elements

        if self.interpolation_step == 'preceding':
            #Calculate U_ek and shift to output
            U_Ek = self.transport.ekman_transport(tau, f, output_position=output_position, f_interpolation=f_interpolation)
            if output_position == 'T':
                U_Ek = self.ops._shift_position(U_Ek,['U','V','W'],elements=[True,True,False])
            elif output_position == 'F':
                U_Ek = self.ops._shift_position(U_Ek,['V','U','W'],elements=[True,True,False])

            #Compute w_Ek
            w_Ek = self.transport.ekman_velocity(U_Ek)

            # Interpolate eta if 'F'
            if output_position == 'F':
                eta=self.ops._shift_position(eta,'F')


        elif self.interpolation_step == 'consecutive':
            pass


        p_down = - rho_0 * g * eta * w_Ek

        return p_down

    def P_lat(self, tau_x, tau_y, eta, f, rho_0=1026, g=9.81, output_position=None, f_interpolation=True, **kwargs):
        """
        Computes the downward flux into the geostrophic circulation.

        It is calculated from the surface wind stress (tau_x, tau_y) and the sea surface height eta with:
        P_down = Grad_h( rho_0 * g * eta * U_ek )

        Computation is performed with interpolation of the data variables to give the result on a specified
        grid position.
        Note: The windstress z-coordinate has a default 0-input to stay consistent with the crossproduct operator.

        When interpolation is done preceding to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau/rho*f {'U' and 'V'}  ===>>> U_ek {['VW','UW', __ ]} --interp--> {['U','V', __ ]}

        eta {'T'} --interp--> {['U','V', __ ]}
        ===>>> P_lat {'T'}

        'F'-point:
        tau/rho*f {'U' and 'V'} ===>>> k x tau/rho*f {['VW','UW', __ ]} --interp--> {['V','U', __ ]}
        eta {'T'} --interp--> eta {['V','U', __ ]}
        ===>>> P_lat {'F'}

        When interpolation is done consecutive to the computation the following order is
        followed to end up on the points 'T' or 'F':

        'T'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['T','T', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['T','T', __ ]}
        ===>>> P_tau {'T'}

        'F'-point:
        tau {'U' and 'V'} ===>>> U_Ek {['VW','UW','F']}  --interp--> {['F','F', __ ]}
        eta {'T'} ===>>> Grad(eta) {['U','V', __ ]} --interp--> {['F','F', __ ]}
        ===>>> P_tau {'F'}

        :param tau_x:           wind stress x-direction
        :param tau_y:           wind stress y-direction
        :param eta:             sea surface height eta
        :param f:               coriolis parameter
        :param rho_0:           surface water density
        :param g:               gravitational acceleration
        :param position_out:    position on grid of output P_taug

        :return:                P_taug
        """
        tau = [tau_x, tau_y, 0]

        if output_position is None:
            output_position = self.position_out

        # Get positions of elements
        #(tau_pos, eta_pos) = (self.ops._get_position(tau), self.ops._get_position(eta))

        if self.interpolation_step == 'preceding':
            #Calculate U_ek and shift to output
            U_Ek = self.transport.ekman_transport(tau, f, output_position=output_position, f_interpolation=f_interpolation)
            if output_position == 'T':
                U_Ek = self.ops._shift_position(U_Ek,['U','V','W'],elements=[True,True,False])
            elif output_position == 'F':
                U_Ek = self.ops._shift_position(U_Ek,['V','U','W'],elements=[True,True,False])

            #Compute w_Ek
            w_Ek = self.transport.ekman_velocity(U_Ek)

            # Interpolate eta. distinguish between x and y to contain dimensions
            if output_position == 'T':
                eta_x=self.ops._shift_position(eta,'U')
                eta_y=self.ops._shift_position(eta,'V')
            if output_position == 'F':
                eta_x = self.ops._shift_position(eta, 'V')
                eta_y = self.ops._shift_position(eta, 'U')

            # Compute lateral flux phi_lat
            phi_lat_x = rho_0 * g * eta_x * U_Ek[0]
            phi_lat_y = rho_0 * g * eta_y * U_Ek[1]
            #phi_lat = rho_0 * g *eta * U_Ek

        elif self.interpolation_step == 'consecutive':
            pass


        p_lat = self.ops.dot([1, 1, 0],[self.ops.derivative(phi_lat_x, axis='X'),
                                        self.ops.derivative(phi_lat_y, axis='Y'),
                                        0])

        return p_lat, [phi_lat_x,phi_lat_y]
