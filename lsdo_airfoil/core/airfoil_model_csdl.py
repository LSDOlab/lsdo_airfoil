from csdl import Model
import csdl
import numpy as np
import importlib
from lsdo_airfoil.core.pre_processing.coordinate_processing import CoordinateProcessing
from lsdo_airfoil.utils.get_airfoil_model import get_airfoil_models
from lsdo_airfoil.utils.load_control_points import load_control_points
from lsdo_airfoil.core.airfoil_models import ClModel, CdModel, CmModel, CpLowerModel, CpUpperModel, DeltaStarLowerModel, DeltaStarUpperModel, ThetaLowerModel, ThetaUpperModel
from lsdo_airfoil import CONTROL_POINTS_FOLDER


X_min_numpy_prestall = np.array([
                -3.87714524e-03, -6.21114345e-03, -3.65010835e-03, -5.48448414e-04,
                1.04316720e-03, -3.44090629e-04,  -1.91351119e-03, -2.22159643e-03,
                -2.74770800e-03, -4.31647711e-03, -5.94483502e-03, -9.50526167e-03,
                -1.18035562e-02, -1.22448076e-02, -8.20550136e-03, -3.83688067e-03,
                -2.66918913e-04,  5.67624951e-03,  1.92390252e-02,  2.55834870e-02,
                3.14692594e-02,  3.43126804e-02,   3.81270386e-02,  4.34582904e-02,
                4.47864607e-02,  4.02273424e-02,   3.80498208e-02,  2.97566336e-02,
                2.03249976e-02,  1.10881981e-02,   4.26956685e-03, -5.45227900e-04,
                -8.00000000e+00,  1.00000000e+05,  0.00000000e+00]
            )

X_min_numpy_poststall = np.array([
                -3.87714524e-03, -6.21114345e-03, -3.65010835e-03, -5.48448414e-04,
                1.04316720e-03, -3.44090629e-04,  -1.91351119e-03, -2.22159643e-03,
                -2.74770800e-03, -4.31647711e-03, -5.94483502e-03, -9.50526167e-03,
                -1.18035562e-02, -1.22448076e-02, -8.20550136e-03, -3.83688067e-03,
                -2.66918913e-04,  5.67624951e-03,  1.92390252e-02,  2.55834870e-02,
                3.14692594e-02,  3.43126804e-02,   3.81270386e-02,  4.34582904e-02,
                4.47864607e-02,  4.02273424e-02,   3.80498208e-02,  2.97566336e-02,
                2.03249976e-02,  1.10881981e-02,   4.26956685e-03, -5.45227900e-04,
                -90.00000000e+00,  1.00000000e+05,  0.00000000e+00]
            )

X_max_numpy_prestall = np.array([
                1.64971128e-03, 5.25048282e-03, 1.47131169e-02, 3.03167850e-02,
                4.73764949e-02, 6.15255609e-02, 7.35139325e-02, 8.21573734e-02,
                8.81158486e-02, 9.02919322e-02, 8.93072858e-02, 8.19384754e-02,
                7.00145736e-02, 5.29626682e-02, 3.25598940e-02, 1.39800459e-02,
                1.76265929e-02, 4.02436182e-02, 7.17813671e-02, 1.06165685e-01,
                1.40150547e-01, 1.67483926e-01, 1.88060194e-01, 2.04852015e-01,
                2.15405628e-01, 2.26217642e-01, 2.21330181e-01, 1.99092031e-01,
                1.54896125e-01, 1.00657888e-01, 4.71989214e-02, 1.23437112e-02,
                17, 8.00000000e+06, 6.00000000e-01]
                )

X_max_numpy_poststall = np.array([
                1.64971128e-03, 5.25048282e-03, 1.47131169e-02, 3.03167850e-02,
                4.73764949e-02, 6.15255609e-02, 7.35139325e-02, 8.21573734e-02,
                8.81158486e-02, 9.02919322e-02, 8.93072858e-02, 8.19384754e-02,
                7.00145736e-02, 5.29626682e-02, 3.25598940e-02, 1.39800459e-02,
                1.76265929e-02, 4.02436182e-02, 7.17813671e-02, 1.06165685e-01,
                1.40150547e-01, 1.67483926e-01, 1.88060194e-01, 2.04852015e-01,
                2.15405628e-01, 2.26217642e-01, 2.21330181e-01, 1.99092031e-01,
                1.54896125e-01, 1.00657888e-01, 4.71989214e-02, 1.23437112e-02,
                90., 8.00000000e+06, 6.00000000e-01]
                )


class AirfoilModelCSDL(Model):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('airfoil_raw_shape', types=tuple, default=None, allow_none=True)
        self.parameters.declare('airfoil_name', types=str, allow_none=True)
        self.parameters.declare('compute_control_points', types=bool, default=True)

    def define(self):
        airfoil_raw_shape = self.parameters['airfoil_raw_shape']
        airfoil_name = self.parameters['airfoil_name']
        
        neural_net_dict = get_airfoil_models()
        num_nodes = self.parameters['num_nodes']

        compute_control_points = self.parameters['compute_control_points']
        if compute_control_points:
            airfoil_upper = self.declare_variable('airfoil_upper', shape=airfoil_raw_shape)
            airfoil_lower = self.declare_variable('airfoil_lower', shape=airfoil_raw_shape)

            airfoil_camber = self.register_output('airfoil_camber', 0.5 * (airfoil_upper + airfoil_lower))
            airfoil_thickness = self.register_output('airfoil_thickness', ((airfoil_upper - airfoil_lower)**2)**0.5)

            cpts_camber, cpts_thickness_raw = csdl.custom(airfoil_camber, airfoil_thickness, op=CoordinateProcessing(airfoil_raw_shape=airfoil_raw_shape))
            self.register_output('control_points_camber', cpts_camber)
            self.register_output('control_points_thickness_raw', cpts_thickness_raw)

            cpts_thickness_raw_declared = self.declare_variable('control_points_thickness_raw', shape=(18, 1))
            cpts_thickness = (cpts_thickness_raw_declared**2)**0.5
            self.register_output('control_points_thickness', cpts_thickness)
       

            control_points = self.create_output('control_points', shape=(32, 1), val=0)
            control_points[0:16, 0] = cpts_camber[1:17, 0]
            control_points[16:, 0] = cpts_thickness[1:17, 0]

        else:
            available_airfoils = ['boeing_vertol_vr_12', 'Clark_y', 'NACA_4412', 'NASA_langley_ga_1']
            if airfoil_name not in available_airfoils:
                raise Exception(f"Unknown airfoil '{airfoil_name}'. Pre-computed camber and thickness B-spline control points exist for the following airfoils: {available_airfoils}")
            else:
                control_points_numpy = np.load(CONTROL_POINTS_FOLDER / f'{airfoil_name}.npy')
                control_points = self.create_input('control_points', shape=(32, 1), val=control_points_numpy)


        # Min and max for normalizing the data (this is based on the training data)
        X_min_prestall = csdl.expand(self.declare_variable(
            name='X_min_prestall',    
            val=X_min_numpy_prestall
        ), (num_nodes, 35), 'i->ji')


        X_max_prestall = csdl.expand(self.declare_variable(
            name='X_max_prestall',
            val=X_max_numpy_prestall,
        ), (num_nodes, 35), 'i->ji') 


        X_min_poststall = csdl.expand(self.declare_variable(
            name='X_min_poststall',    
            val=X_min_numpy_poststall
        ), (num_nodes, 35), 'i->ji')


        X_max_poststall = csdl.expand(self.declare_variable(
            name='X_max_poststall',
            val=X_max_numpy_poststall,
        ), (num_nodes, 35), 'i->ji') 


        M = self.declare_variable('mach_number', shape=(num_nodes, ))
        Re = self.declare_variable('reynolds_number', shape=(num_nodes, ))
        AoA = self.declare_variable('angle_of_attack', shape=(num_nodes, ))
        control_points_exp = csdl.expand(csdl.reshape(control_points, (32, )), (num_nodes, 32), 'i->ji')

        inputs = self.create_output('airfoil_inputs', shape=(num_nodes, 35), val=0)
        inputs[:, 0:32] = control_points_exp
        inputs[:, 32] = csdl.reshape(AoA, (num_nodes, 1))
        inputs[:, 33] = csdl.reshape(Re, (num_nodes, 1))
        inputs[:, 34] = csdl.reshape(M, (num_nodes, 1))


        # Scaling the variables
        scaled_inputs_prestall = (inputs - X_min_prestall) / (X_max_prestall - X_min_prestall)
        x = self.register_output('neural_net_input', scaled_inputs_prestall)

        scaled_inputs_poststall = (inputs - X_min_poststall) / (X_max_poststall - X_min_poststall)
        x_extrap = self.register_output('neural_net_input_extrap', scaled_inputs_poststall)
        


        # ------------------- HARD CODE INPUTS FOR PARTIALS TEST ------------------- #
        # scaled_inputs_cp = self.create_input('scaled_cp', val=np.array(
        #     [[0.75327905, 0.81741833, 0.57204811, 0.34830446, 0.24415865 ,0.23976231,
        #     0.23986748, 0.23642913, 0.24136649, 0.26048643, 0.28095821, 0.34019981,
        #     0.4111598 , 0.48562833, 0.52528305, 0.41864856, 0.42387705 ,0.97575927,
        #     0.83776899, 0.82634967 ,0.75395031, 0.74818546, 0.74461437 ,0.73406119,
        #     0.72887614, 0.70057835, 0.69333178 ,0.70430033, 0.70485218 ,0.68205206,
        #     0.57749769, 0.84025751]]))
        # self.add_design_variable('scaled_cp')
       
        # scaled_inputs =  val=np.array(
        #     [[0.75327905, 0.81741833, 0.57204811, 0.34830446, 0.24415865 ,0.23976231,
        #     0.23986748, 0.23642913, 0.24136649, 0.26048643, 0.28095821, 0.34019981,
        #     0.4111598 , 0.48562833, 0.52528305, 0.41864856, 0.42387705 ,0.97575927,
        #     0.83776899, 0.82634967 ,0.75395031, 0.74818546, 0.74461437 ,0.73406119,
        #     0.72887614, 0.70057835, 0.69333178 ,0.70430033, 0.70485218 ,0.68205206,
        #     0.57749769, 0.84025751, 1.2, 0.24050633, 0.16666667]])
        # print(np.tile(scaled_inputs.T, 4).T)
        # exit()
        
        
        # scaled_inputs_aoa = self.create_input('scaled_aoa', val=np.array([[1.2]]))
        # self.add_design_variable('scaled_aoa')
        # scaled_inputs_Re = self.create_input('scaled_Re', val=np.array([[0.24050633]]))
        # self.add_design_variable('scaled_Re')
        # scaled_inputs_M = self.create_input('scaled_M', val=np.array([[0.16666667]]))
    
        # x = self.create_output('neural_net_input', val=0, shape=(1, 35))
        # x[0, 0:32] = scaled_inputs_cp
        # x[0, 32] = scaled_inputs_aoa
        # x[0, 33] = scaled_inputs_Re
        # x[0, 34] = scaled_inputs_M
        
        # x = self.create_input('neural_net_input', val=np.tile(scaled_inputs.T, 100).T)
        # self.add_design_variable('neural_net_input')

        # ------------------- HARD CODE INPUTS FOR PARTIALS TEST ------------------- #

        cl = csdl.custom(x_extrap, op=ClModel(
                neural_net=neural_net_dict['Cl'],
                num_nodes=num_nodes,     
            )
        )
        self.register_output('Cl', cl)

        cd = csdl.custom(x_extrap, op=CdModel(
                neural_net=neural_net_dict['Cd'],
                num_nodes=num_nodes,

            )
        )
        self.register_output('Cd', cd)

        delta_star_upper = csdl.custom(x, op=DeltaStarUpperModel(
            neural_net=neural_net_dict['delta_star_upper'],
            num_nodes=num_nodes,
            X_min=X_min_numpy_prestall,
            X_max=X_max_numpy_prestall,
        ))
        self.register_output('DeltaStarUpper', delta_star_upper)

        delta_star_lower = csdl.custom(x, op=DeltaStarLowerModel(
            neural_net=neural_net_dict['delta_star_lower'],
            num_nodes=num_nodes,
            X_min=X_min_numpy_prestall,
            X_max=X_max_numpy_prestall,
        ))
        self.register_output('DeltaStarLower', delta_star_lower)


        theta_upper = csdl.custom(x, op=ThetaUpperModel(
            neural_net=neural_net_dict['theta_upper'],
            num_nodes=num_nodes,
        ))
        self.register_output('ThetaUpper', theta_upper)

        theta_lower = csdl.custom(x, op=ThetaLowerModel(
            neural_net=neural_net_dict['theta_lower'],
            num_nodes=num_nodes,
        ))
        self.register_output('ThetaLower', theta_lower)
