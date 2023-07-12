import csdl
import torch
import numpy as np
import copy
import scipy
import time


class ClModel(csdl.CustomExplicitOperation):
    def initialize(self):
        # The neural net will be a pre-trained model
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neural_net')


    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input_extrap', shape=(num_nodes, 35))
        self.add_output('Cl', shape=(num_nodes, ))

        self.declare_derivatives('Cl', 'neural_net_input_extrap')

    
    def compute(self, inputs, outputs):
        neural_net = self.parameters['neural_net']        
        neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
        # print('neural_net_input', neural_net_input)
        neural_net_prediction = neural_net(neural_net_input).detach().numpy()
        
        # print('Cl', neural_net_input)

        outputs['Cl'] =  neural_net_prediction #cl_output 
        
    
    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']        
        neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
        first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivatives['Cl', 'neural_net_input_extrap'] =  first_derivative_numpy#  scipy.linalg.block_diag(*derivatives_list)
       

class CdModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('neural_net')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input_extrap', shape=(num_nodes, 35))
        self.add_output('Cd', shape=(num_nodes, ))

        self.declare_derivatives('Cd', 'neural_net_input_extrap')

    
    def compute(self, inputs, outputs):
        neural_net = self.parameters['neural_net']        
        neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])
        neural_net_prediction = neural_net(neural_net_input).detach().numpy()
        # print(neural_net_prediction)
        outputs['Cd'] =  neural_net_prediction # cd_output 
        
        # neural_net_input = torch.Tensor(inputs['neural_net_input'])
        # outputs['Cd'] = neural_net(neural_net_input).detach().numpy().flatten()

    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']        
        neural_net_input = torch.Tensor(inputs['neural_net_input_extrap'])

        first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivatives['Cd', 'neural_net_input_extrap'] = first_derivative_numpy #scipy.linalg.block_diag(*derivatives_list)#

class CmModel(csdl.CustomExplicitOperation):
    def initialize(self):
        # The neural net will be a pre-trained model
        self.parameters.declare('num_nodes')
        self.parameters.declare('neural_net')

    def define(self):
        self.add_input('neural_net_input', shape=(1, 35))
        self.add_output('Cm')

        # self.declare_derivatives('*', '*')
    
    def compute(self, inputs, outputs):
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'])
        outputs['Cm'] = neural_net(neural_net_input).detach().numpy().flatten()

    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.tensor(inputs['neural_net_input'], dtype=torch.float32, requires_grad=True)
        outputs = neural_net(neural_net_input)

        derivatives['Cm', 'neural_net_input'] =  torch.autograd.grad(outputs, neural_net_input)[0].detach().numpy().reshape(1, 35)


# Vector valued regressions

class CpUpperModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('neural_net')
        self.parameters.declare('num_nodes')
        self.parameters.declare('X_max')
        self.parameters.declare('X_min')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('CpUpper', shape=(num_nodes, 100))

        self.declare_derivatives('CpUpper', 'neural_net_input')

    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        outputs['CpUpper'] = neural_net(neural_net_input).detach().numpy().flatten()

    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.tensor(inputs['neural_net_input'], dtype=torch.float32, requires_grad=True)

        derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivatives['CpUpper', 'neural_net_input'] =  derivative_numpy # torch.autograd.grad(outputs, neural_net_input)[0].detach().numpy().reshape(1, 35)

class CpLowerModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('neural_net')
        self.parameters.declare('X_max')
        self.parameters.declare('X_min')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('CpLower',shape=(num_nodes, 100))

        self.declare_derivatives('CpLower', 'neural_net_input')


    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        outputs['CpLower'] = neural_net(neural_net_input).detach().numpy().flatten()

    def compute_derivatives(self, inputs, derivatives):
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.tensor(inputs['neural_net_input'], dtype=torch.float32, requires_grad=True)

        derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivatives['CpLower', 'neural_net_input'] = derivative_numpy # torch.autograd.grad(outputs, neural_net_input)[0].detach().numpy().reshape(1, 35)


num_pts = 100
x_range = np.linspace(0, 1, num_pts)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)
positive_stall_disp_thickness = np.exp(1.*x_interp)-0.999
negative_stall_disp_thickness = 1e-4 * np.ones((num_pts, ))


class DeltaStarUpperModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('neural_net')
        self.parameters.declare('num_nodes')
        self.parameters.declare('X_max')
        self.parameters.declare('X_min')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('DeltaStarUpper', shape=(num_nodes, 100))

        self.declare_derivatives('DeltaStarUpper', 'neural_net_input')

    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        X_max = self.parameters['X_max']
        X_min = self.parameters['X_min']
        
        # Unscale the data
        neural_net_input_unscaled = inputs['neural_net_input'] * (X_max-X_min) + X_min
        alpha = neural_net_input_unscaled[:, -3].reshape(num_nodes, 1)
        
        self.positive_stall_indices = np.where(alpha>17)
        self.negative_stall_indices = np.where(alpha<-8)
    
        output = neural_net(neural_net_input).detach().numpy()
        output[self.positive_stall_indices[0]] = positive_stall_disp_thickness
        output[self.negative_stall_indices[0]] = negative_stall_disp_thickness
        outputs['DeltaStarUpper'] = output
    
    def compute_derivatives(self, inputs, derivatives):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        neural_net_input.requires_grad = True
        
        derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivative_numpy[self.positive_stall_indices[0], :, :, self.positive_stall_indices[0], :, :] = 0
        derivative_numpy[self.negative_stall_indices[0], :, :, self.negative_stall_indices[0], :, :] = 0
        derivatives['DeltaStarUpper', 'neural_net_input'] = derivative_numpy 

class DeltaStarLowerModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('neural_net')
        self.parameters.declare('num_nodes')
        self.parameters.declare('X_max')
        self.parameters.declare('X_min')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('DeltaStarLower', shape=(num_nodes, 100))

        self.declare_derivatives('DeltaStarLower', 'neural_net_input')


    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))

        X_max = self.parameters['X_max']
        X_min = self.parameters['X_min']
        
        # Unscale the data
        neural_net_input_unscaled = inputs['neural_net_input'] * (X_max-X_min) + X_min
        alpha = neural_net_input_unscaled[:, -3].reshape(num_nodes, 1)

        self.positive_stall_indices = np.where(alpha>17)
        self.negative_stall_indices = np.where(alpha<-8)

        output = neural_net(neural_net_input).detach().numpy()
        output[self.positive_stall_indices[0]] = negative_stall_disp_thickness
        output[self.negative_stall_indices[0]] = positive_stall_disp_thickness / 5
        outputs['DeltaStarLower'] = output

        # outputs['DeltaStarLower'] = neural_net(neural_net_input).detach().numpy().flatten()

    def compute_derivatives(self, inputs, derivatives):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        neural_net_input.requires_grad = True
        
        derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input).detach().numpy()
        derivative_numpy[self.positive_stall_indices[0], :, :, self.positive_stall_indices[0], :, :] = 0
        derivative_numpy[self.negative_stall_indices[0], :, :, self.negative_stall_indices[0], :, :] = 0
        derivatives['DeltaStarLower', 'neural_net_input'] = derivative_numpy 


class ThetaUpperModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('neural_net')
        self.parameters.declare('num_nodes')
        # self.parameters.declare('X_max')
        # self.parameters.declare('X_min')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('ThetaUpper', shape=(num_nodes, 100))

    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        outputs['ThetaUpper'] = neural_net(neural_net_input).detach().numpy().flatten()


class ThetaLowerModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('neural_net')
        self.parameters.declare('num_nodes')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        self.add_input('neural_net_input', shape=(num_nodes, 35))
        self.add_output('ThetaLower', shape=(num_nodes, 100))

    def compute(self, inputs, outputs):
        num_nodes = self.parameters['num_nodes']
        neural_net = self.parameters['neural_net']
        neural_net_input = torch.Tensor(inputs['neural_net_input'].reshape(num_nodes, 1, 35))
        outputs['ThetaLower'] = neural_net(neural_net_input).detach().numpy().flatten()
































#--------------OLD CODE----------------#

            # elif
                # print('\n')
                # print(outputs)
                # print(tiled_inputs)
                # print(first_derivative_numpy)
                # exit()
                
                # # raise Exception('smoothing')
                # aoa_stall_p = self.alpha_stall_rad[i]
                # Cl_stall_p = self.cl_stall[i]
                # A2_p = (Cl_stall_p - self.Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * np.sin(aoa_stall_p) / (np.cos(aoa_stall_p)**2)
                # mat_cl_p = np.array([
                #     [(aoa_stall_p-eps)**3, (aoa_stall_p-eps)**2, (aoa_stall_p-eps), 1],
                #     [(aoa_stall_p+eps)**3, (aoa_stall_p+eps)**2, (aoa_stall_p+eps), 1],
                #     [3 * (aoa_stall_p-eps)**2, 2*(aoa_stall_p-eps), 1, 0],
                #     [3 * (aoa_stall_p+eps)**2, 2*(aoa_stall_p+eps), 1, 0],
                # ])

                # neural_net_scalar_input = neural_net_input_copy_stall_minus[i, :]
                # entry_3 = torch.autograd.functional.jacobian(neural_net, neural_net_scalar_input)[0].detach().numpy()[-3]

                # lhs_cl_p = np.array([
                #     [neural_net_eval_stall_minus[i][0]],
                #     [ self.A1 * np.sin(2 * (aoa_stall_p+eps)) + A2_p * np.cos(aoa_stall_p+eps)**2 / np.sin(aoa_stall_p+eps)],
                #     [entry_3],
                #     [2 * self.A1 * np.cos(2 * (aoa_stall_p+eps)) - A2_p * (np.cos(aoa_stall_p+eps) * (1+1/(np.sin(aoa_stall_p+eps))**2))],
                # ])
                
                # coeff_cl_p = np.linalg.solve(mat_cl_p, lhs_cl_p)
                

                # # coeff_cl_p = self.coeff_mat[i, :]
                # d_dalpha = coeff_cl_p[2] + 2 * coeff_cl_p[1] * self.alpha[i] + 3 * coeff_cl_p[0] * self.alpha[i]**2
                # first_derivative_numpy = np.zeros((1, 35))
                # first_derivative_numpy[0, -3] = d_dalpha * np.pi/180 * (X_max[-3] -  X_min[-3])
                # first_derivative_numpy[0, -2] = first_derivative_numpy_test[i, -2]


                
            # else
                # aoa_stall_p = self.alpha_stall_rad[i]
                # Cl_stall_p = self.cl_stall[i]
                # A2_p = (Cl_stall_p - self.Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * np.sin(aoa_stall_p) / (np.cos(aoa_stall_p)**2)
                # aoa =  inputs['neural_net_input'][i, -3]
                # d_dalpha = (2 * self.A1 * np.cos(2 * self.alpha[i]) - A2_p * (np.cos(self.alpha[i]) * (1+1/(np.sin(self.alpha[i]))**2))) * np.pi/180 * (X_max[-3] -  X_min[-3])


                # first_derivative_numpy = np.zeros((1, 35))
                # # first_derivative_numpy[0, -1] = first_derivative_numpy_test[0, -1]
                # # first_derivative_numpy[0, -2] = first_derivative_numpy_test[0, -2]
                # first_derivative_numpy[0, -3] = d_dalpha
                # print('first_derivative_numpy', first_derivative_numpy)
                # derivatives_list.append(first_derivative_numpy)

# CL COMPUTE

 # # Stall considerations 
        # X_max = self.parameters['X_max']
        # X_min = self.parameters['X_min']
        
        # # Unscale the data
        # self.neural_net_input_unscaled = inputs['neural_net_input'] * (X_max-X_min) + X_min
        # self.alpha = np.deg2rad(self.neural_net_input_unscaled[:, -3].reshape(num_nodes, 1))
        # self.Re = self.neural_net_input_unscaled[:, -2].reshape(num_nodes, 1)
        # self.M = self.neural_net_input_unscaled[:, -1].reshape(num_nodes, 1)

        # # Compute cl_stall, aoa_stall for given Mach and Reynolds number
        # # print('np.hstack((self.Re, self.M))', np.hstack((self.Re, self.M)))
        # self.cl_stall =  cl_stall_interp(np.hstack((self.Re, self.M))) # 1.5 * np.ones((num_nodes, ))#
        # alpha_stall_deg = alpha_stall_interp(np.hstack((self.Re, self.M)))
        # self.alpha_stall_rad =   np.deg2rad(alpha_stall_deg) # np.deg2rad(16) * np.ones((num_nodes, ))#

        # cl_stall_minus = -1. * np.ones((num_nodes, ))
        # alpha_stall_m_deg = -8. * np.ones((num_nodes, ))
        # self.alpha_stall_m_rad = np.deg2rad(alpha_stall_m_deg)


        # # For smothing, evaluate airfaoil ML model wihthin smoothing region
        # alpha_stall_minus_eps_scaled = ((alpha_stall_deg - np.rad2deg(eps)) - X_min[-3]) / (X_max[-3] - X_min[-3])
        # neural_net_input_copy_stall_minus = copy.deepcopy(neural_net_input)
        # neural_net_input_copy_stall_minus[:, -3] = torch.Tensor(alpha_stall_minus_eps_scaled)
        # neural_net_eval_stall_minus = neural_net(neural_net_input_copy_stall_minus).detach().numpy()


        # alpha_stall_minus_eps_scaled_negative= ((alpha_stall_m_deg - np.rad2deg(eps)) - X_min[-3]) / (X_max[-3] - X_min[-3])
        # neural_net_input_copy_stall_minus_negative = copy.deepcopy(neural_net_input)
        # neural_net_input_copy_stall_minus_negative[:, -3] = torch.Tensor(alpha_stall_minus_eps_scaled_negative)
        # neural_net_eval_stall_minus_negative = neural_net(neural_net_input_copy_stall_minus_negative).detach().numpy()
       

        # # Viterna Extrapolation 
        # AR = 10.
        # self.Cd_max = 1.11 + 0.018 * AR
        # self.A1 = self.Cd_max / 2
        # B1 = self.Cd_max

        

        # self.coeff_mat = np.zeros((num_nodes, 4))

        # cl_output = np.zeros((num_nodes, 1))        
        # for i in range(num_nodes):
        #     if self.alpha[i] <= (self.alpha_stall_m_rad[i] - eps):
        #         aoa_stall_m = self.alpha_stall_m_rad[i]
        #         Cl_stall_m = cl_stall_minus[i]
        #         A2_m = (Cl_stall_m - self.Cd_max * np.sin(aoa_stall_m) * np.cos(aoa_stall_m)) * np.sin(aoa_stall_m) / (np.cos(aoa_stall_m)**2)
        #         cl_output[i, 0] = self.A1 * np.sin(2 * self.alpha[i]) + A2_m * np.cos(self.alpha[i])**2 / np.sin(self.alpha[i])
                
                

        #     elif (self.alpha[i] > (self.alpha_stall_m_rad[i] - eps)) & (self.alpha[i] <= (self.alpha_stall_m_rad[i]+eps)):
        #         aoa_stall_m = self.alpha_stall_m_rad[i]
        #         Cl_stall_m = cl_stall_minus[i]
        #         A2_m = (Cl_stall_m - self.Cd_max * np.sin(aoa_stall_m) * np.cos(aoa_stall_m)) * np.sin(aoa_stall_m) / (np.cos(aoa_stall_m)**2)
                
        #         mat_cl_m = np.array([
        #             [(aoa_stall_m-eps)**3, (aoa_stall_m-eps)**2, (aoa_stall_m-eps), 1],
        #             [(aoa_stall_m+eps)**3, (aoa_stall_m+eps)**2, (aoa_stall_m+eps), 1],
        #             [3 * (aoa_stall_m-eps)**2, 2*(aoa_stall_m-eps), 1, 0],
        #             [3 * (aoa_stall_m+eps)**2, 2*(aoa_stall_m+eps), 1, 0],
        #         ])
                
        #         neural_net_scalar_input = neural_net_input_copy_stall_minus_negative[i, :]
        #         entry_4 = torch.autograd.functional.jacobian(neural_net, neural_net_scalar_input)[0].detach().numpy()[-3]

        #         lhs_cl_m = np.array([
        #             [self.A1 * np.sin(2 * (aoa_stall_m-eps)) + A2_m * np.cos(aoa_stall_m-eps)**2 / np.sin(aoa_stall_m-eps)],
        #             [neural_net_eval_stall_minus_negative[i][0]],
        #             [2 * self.A1 * np.cos(2 * (aoa_stall_m-eps)) - A2_m * (np.cos(aoa_stall_m-eps) * (1+1/(np.sin(aoa_stall_m-eps))**2))],
        #             [entry_4],
        #         ])
        #         self.coeff_cl_m = np.linalg.solve(mat_cl_m, lhs_cl_m)
        #         cl_output[i, 0] = self.coeff_cl_m[3] + self.coeff_cl_m[2] * self.alpha[i] + self.coeff_cl_m[1] * self.alpha[i]**2 + self.coeff_cl_m[0] * self.alpha[i]**3

        #     elif (self.alpha[i] > (self.alpha_stall_m_rad[i]+eps)) & (self.alpha[i] <= (self.alpha_stall_rad[i] - eps)):
        #         cl_output[i, 0] = neural_net_prediction[i]
        #         # print('pre_stall')
            
        #     elif (self.alpha[i] > (self.alpha_stall_rad[i] - eps)) & (self.alpha[i] < (self.alpha_stall_rad[i] + eps)):
        #         # print('smoothing region')
                
        #         aoa_stall_p = self.alpha_stall_rad[i]
        #         Cl_stall_p = self.cl_stall[i]
        #         A2_p = (Cl_stall_p - self.Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * np.sin(aoa_stall_p) / (np.cos(aoa_stall_p)**2)
        #         mat_cl_p = np.array([
        #             [(aoa_stall_p-eps)**3, (aoa_stall_p-eps)**2, (aoa_stall_p-eps), 1],
        #             [(aoa_stall_p+eps)**3, (aoa_stall_p+eps)**2, (aoa_stall_p+eps), 1],
        #             [3 * (aoa_stall_p-eps)**2, 2*(aoa_stall_p-eps), 1, 0],
        #             [3 * (aoa_stall_p+eps)**2, 2*(aoa_stall_p+eps), 1, 0],
        #         ])

        #         neural_net_scalar_input = neural_net_input_copy_stall_minus[i, :]
        #         entry_3 = torch.autograd.functional.jacobian(neural_net, neural_net_scalar_input)[0].detach().numpy()[-3]

        #         lhs_cl_p = np.array([
        #             [neural_net_eval_stall_minus[i][0]],
        #             [ self.A1 * np.sin(2 * (aoa_stall_p+eps)) + A2_p * np.cos(aoa_stall_p+eps)**2 / np.sin(aoa_stall_p+eps)],
        #             [entry_3],
        #             [2 * self.A1 * np.cos(2 * (aoa_stall_p+eps)) - A2_p * (np.cos(aoa_stall_p+eps) * (1+1/(np.sin(aoa_stall_p+eps))**2))],
        #         ])
                
        #         coeff_cl_p = np.linalg.solve(mat_cl_p, lhs_cl_p)
        #         cl_output[i, 0] = coeff_cl_p[3] + coeff_cl_p[2] * self.alpha[i] + coeff_cl_p[1] * self.alpha[i]**2 + coeff_cl_p[0] * self.alpha[i]**3
        #         self.coeff_mat[i, :] = coeff_cl_p.reshape(4, )
        #     else:
        #         # print('viterna extrapolation')
        #         aoa_stall_p = self.alpha_stall_rad[i]
        #         Cl_stall_p = self.cl_stall[i]
        #         A2_p = (Cl_stall_p - self.Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * np.sin(aoa_stall_p) / (np.cos(aoa_stall_p)**2)
        #         cl_output[i, 0] = self.A1 * np.sin(2 * self.alpha[i]) + A2_p * np.cos(self.alpha[i])**2 / np.sin(self.alpha[i])

        # print(cl_output)
        # print(self.alpha * 180/np.pi)
        # exit()        

# CL COMPUTE DERIVATIVES 
# for i in range(num_nodes):
#             first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input[i, :])[0].detach().numpy().reshape(1, 35)
#             derivatives_list.append(first_derivative_numpy)
            # if (self.alpha[i] > (self.alpha_stall_m_rad[i]+eps)) & (self.alpha[i] <= (self.alpha_stall_rad[i] - eps)):
            #     # print(self.alpha[i])                
            #     first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input[i, :])[0].detach().numpy().reshape(1, 35)
            #     derivatives_list.append(first_derivative_numpy)
            
        #     else: # (self.alpha[i] > (self.alpha_stall_rad[i] - eps)): # & (self.alpha[i] < (self.alpha_stall_rad[i] + eps)):
        #         inputs_copy = copy.copy(inputs)
        #         tiled_inputs = np.tile(inputs_copy['neural_net_input'][i, :].reshape(35, 1), 35).T

        #         I_delta_x = delta_x * np.eye(35)
        #         perturbed_inputs_p = tiled_inputs + I_delta_x
        #         perturbed_inputs_m = tiled_inputs - I_delta_x
        #         first_derivative_numpy = np.zeros((1, 35))
        #         for j in range(35):
        #             outputs_p = {}
        #             outputs_p['Cl'] = np.zeros((num_nodes, ))
        #             inputs_copy_p = copy.deepcopy(inputs)
        #             inputs_copy_p['neural_net_input'][:, :] = np.tile(perturbed_inputs_p[j, :].reshape(35, 1), num_nodes).T
        #             self.compute(inputs_copy_p, outputs=outputs_p)
        #             Cl_perturbed_p = outputs_p['Cl'][0]

        #             outputs_m = {}
        #             outputs_m['Cl'] = np.zeros((num_nodes, ))
        #             inputs_copy_m = copy.deepcopy(inputs)
        #             inputs_copy_m['neural_net_input'][:, :] = np.tile(perturbed_inputs_m[j, :].reshape(35, 1), num_nodes).T
        #             self.compute(inputs_copy_m, outputs=outputs_m)
        #             Cl_perturbed_m = outputs_m['Cl'][0]
                    
        #             first_derivative_numpy[0, j] = (Cl_perturbed_p - Cl_perturbed_m) / 2 / delta_x
        #             print(first_derivative_numpy)

        #         derivatives_list.append(first_derivative_numpy)



# CD COMPUTE

# # # Stall considerations 
        # X_max = self.parameters['X_max']
        # X_min = self.parameters['X_min']
        
        # # Unscale the data
        # self.neural_net_input_unscaled = inputs['neural_net_input'] * (X_max-X_min) + X_min
        # self.alpha = np.deg2rad(self.neural_net_input_unscaled[:, -3].reshape(num_nodes, 1))
        # self.Re = self.neural_net_input_unscaled[:, -2].reshape(num_nodes, 1)
        # self.M = self.neural_net_input_unscaled[:, -1].reshape(num_nodes, 1)

        # # Compute cd_stall, aoa_stall for given Mach and Reynolds number
        # self.cd_stall =  cd_stall_interp(np.hstack((self.Re, self.M))) # 1.5 * np.ones((num_nodes, ))#
        # alpha_stall_deg = alpha_stall_interp(np.hstack((self.Re, self.M)))
        # self.alpha_stall_rad =   np.deg2rad(alpha_stall_deg) # np.deg2rad(16) * np.ones((num_nodes, ))#

        
        # alpha_stall_m_deg = -5. * np.ones((num_nodes, ))
        # self.alpha_stall_m_rad = np.deg2rad(alpha_stall_m_deg)

        # # For smothing, evaluate airfaoil ML model wihthin smoothing region
        # alpha_stall_minus_eps_scaled = ((alpha_stall_deg - np.rad2deg(eps)) - X_min[-3]) / (X_max[-3] - X_min[-3])
        # neural_net_input_copy_stall_minus = copy.deepcopy(neural_net_input)
        # neural_net_input_copy_stall_minus[:, -3] = torch.Tensor(alpha_stall_minus_eps_scaled)
        # neural_net_eval_stall_minus = neural_net(neural_net_input_copy_stall_minus).detach().numpy()

        # alpha_stall_minus_eps_scaled_negative= ((alpha_stall_m_deg - np.rad2deg(eps)) - X_min[-3]) / (X_max[-3] - X_min[-3])
        # neural_net_input_copy_stall_minus_negative = copy.deepcopy(neural_net_input)
        # neural_net_input_copy_stall_minus_negative[:, -3] = torch.Tensor(alpha_stall_minus_eps_scaled_negative)
        # neural_net_eval_stall_minus_negative = neural_net(neural_net_input_copy_stall_minus_negative).detach().numpy()

        # cd_stall_minus = neural_net_eval_stall_minus_negative.reshape(num_nodes, )

        # # exit()

        # # Viterna Extrapolation 
        # AR = 10.
        # self.Cd_max = 1.11 + 0.018 * AR
        # self.A1 = self.Cd_max / 2
        # B1 = self.Cd_max

        # self.coeff_mat = np.zeros((num_nodes, 4))
        # eps = np.deg2rad(0.05)
        # cd_output = np.zeros((num_nodes, 1))        
        # for i in range(num_nodes):
        #     if self.alpha[i] <= (self.alpha_stall_m_rad[i] - eps):
        #         aoa_stall_m = self.alpha_stall_m_rad[i]
        #         Cd_stall_m = cd_stall_minus[i]
        #         B2_m = (Cd_stall_m - self.Cd_max * np.sin(aoa_stall_m)**2) / np.cos(aoa_stall_m)
        #         cd_output[i, 0] = B1 * np.sin(self.alpha[i])**2 + B2_m * np.cos(self.alpha[i])
        #         # print(cd_output[i, 0])

        #     elif (self.alpha[i] > (self.alpha_stall_m_rad[i] - eps)) & (self.alpha[i] <= (self.alpha_stall_m_rad[i]+eps)):
        #         aoa_stall_m = self.alpha_stall_m_rad[i]
        #         Cd_stall_m = cd_stall_minus[i]
        #         B2_m = (Cd_stall_m - self.Cd_max * np.sin(aoa_stall_m)**2) / np.cos(aoa_stall_m)
                
        #         mat_cd_m = np.array([
        #             [(aoa_stall_m-eps)**3, (aoa_stall_m-eps)**2, (aoa_stall_m-eps), 1],
        #             [(aoa_stall_m+eps)**3, (aoa_stall_m+eps)**2, (aoa_stall_m+eps), 1],
        #             [3 * (aoa_stall_m-eps)**2, 2*(aoa_stall_m-eps), 1, 0],
        #             [3 * (aoa_stall_m+eps)**2, 2*(aoa_stall_m+eps), 1, 0],
        #         ])
                
        #         neural_net_scalar_input = neural_net_input_copy_stall_minus_negative[i, :]
        #         entry_4 = torch.autograd.functional.jacobian(neural_net, neural_net_scalar_input)[0].detach().numpy()[-3]

        #         lhs_cd_m = np.array([
        #             [B1 * np.sin(aoa_stall_m-eps)**2 + B2_m * np.cos(aoa_stall_m-eps)],
        #             [neural_net_eval_stall_minus_negative[i][0]],
        #             [B1 * np.sin(2 * (aoa_stall_m-eps)) - B2_m * np.sin(aoa_stall_m-eps)],
        #             [entry_4],
        #         ])
        #         self.coeff_cd_m = np.linalg.solve(mat_cd_m, lhs_cd_m)
        #         cd_output[i, 0] = self.coeff_cd_m[3] + self.coeff_cd_m[2] * self.alpha[i] + self.coeff_cd_m[1] * self.alpha[i]**2 + self.coeff_cd_m[0] * self.alpha[i]**3
            
        #     elif (self.alpha[i] > (self.alpha_stall_m_rad[i]+eps)) & (self.alpha[i] <= (self.alpha_stall_rad[i] - eps)):
        #         cd_output[i, 0] = neural_net_prediction[i]
        #         # print('pre_stall')
            
        #     elif (self.alpha[i] > (self.alpha_stall_rad[i] - eps)) & (self.alpha[i] < (self.alpha_stall_rad[i] + eps)):
        #         # print('smoothing region')
                
        #         aoa_stall_p = self.alpha_stall_rad[i]
        #         Cd_stall_p = self.cd_stall[i]
        #         B2_p = (Cd_stall_p - self.Cd_max * np.sin(aoa_stall_p)**2) / np.cos(aoa_stall_p)
                
        #         mat_cd_p = np.array([
        #             [(aoa_stall_p-eps)**3, (aoa_stall_p-eps)**2, (aoa_stall_p-eps), 1],
        #             [(aoa_stall_p+eps)**3, (aoa_stall_p+eps)**2, (aoa_stall_p+eps), 1],
        #             [3 * (aoa_stall_p-eps)**2, 2*(aoa_stall_p-eps), 1, 0],
        #             [3 * (aoa_stall_p+eps)**2, 2*(aoa_stall_p+eps), 1, 0],
        #         ])

        #         neural_net_scalar_input = neural_net_input_copy_stall_minus[i, :]
        #         entry_3 = torch.autograd.functional.jacobian(neural_net, neural_net_scalar_input)[0].detach().numpy()[-3]

        #         lhs_cd_p = np.array([
        #             [neural_net_eval_stall_minus[i][0]],
        #             [B1 * np.sin(aoa_stall_p+eps)**2 + B2_p * np.cos(aoa_stall_p+eps)],
        #             [entry_3],
        #             [B1 * np.sin(2 * (aoa_stall_p+eps)) - B2_p * np.sin(aoa_stall_p+eps)],
        #         ])
                
        #         coeff_cd_p = np.linalg.solve(mat_cd_p, lhs_cd_p)
        #         cd_output[i, 0] = coeff_cd_p[3] + coeff_cd_p[2] * self.alpha[i] + coeff_cd_p[1] * self.alpha[i]**2 + coeff_cd_p[0] * self.alpha[i]**3
        #         self.coeff_mat[i, :] = coeff_cd_p.reshape(4, )
            
        #     else:
        #         # print('viterna extrapolation')
        #         aoa_stall_p = self.alpha_stall_rad[i]
        #         Cd_stall_p = self.cd_stall[i]
        #         B2_p = (Cd_stall_p - self.Cd_max * np.sin(aoa_stall_p)**2) / np.cos(aoa_stall_p)
        #         cd_output[i, 0] = B1 * np.sin(self.alpha[i])**2 + B2_p * np.cos(self.alpha[i])                


# CD COMPUTE DERIVATIVES

         # derivatives_list = []
        # delta_x = 1e-3
        # for i in range(num_nodes):
        #     first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input[i, :])[0].detach().numpy().reshape(1, 35)
        #     derivatives_list.append(first_derivative_numpy)
            # if (self.alpha[i] > (self.alpha_stall_m_rad[i]+eps)) & (self.alpha[i] <= (self.alpha_stall_rad[i] - eps)):
            #     # print(self.alpha[i])                
            #     first_derivative_numpy = torch.autograd.functional.jacobian(neural_net, neural_net_input[i, :])[0].detach().numpy().reshape(1, 35)
            #     derivatives_list.append(first_derivative_numpy)
            
            # else: # (self.alpha[i] > (self.alpha_stall_rad[i] - eps)): # & (self.alpha[i] < (self.alpha_stall_rad[i] + eps)):
            #     inputs_copy = copy.copy(inputs)
            #     tiled_inputs = np.tile(inputs_copy['neural_net_input'][i, :].reshape(35, 1), 35).T

            #     I_delta_x = delta_x * np.eye(35)
            #     perturbed_inputs_p = tiled_inputs + I_delta_x
            #     perturbed_inputs_m = tiled_inputs - I_delta_x
            #     first_derivative_numpy = np.zeros((1, 35))
            #     for j in range(35):
            #         outputs_p = {}
            #         outputs_p['Cd'] = np.zeros((num_nodes, ))
            #         inputs_copy_p = copy.deepcopy(inputs)
            #         inputs_copy_p['neural_net_input'][:, :] = np.tile(perturbed_inputs_p[j, :].reshape(35, 1), num_nodes).T
            #         self.compute(inputs_copy_p, outputs=outputs_p)
            #         Cd_perturbed_p = outputs_p['Cd'][0]

            #         outputs_m = {}
            #         outputs_m['Cd'] = np.zeros((num_nodes, ))
            #         inputs_copy_m = copy.deepcopy(inputs)
            #         inputs_copy_m['neural_net_input'][:, :] = np.tile(perturbed_inputs_m[j, :].reshape(35, 1), num_nodes).T
            #         self.compute(inputs_copy_m, outputs=outputs_m)
            #         Cd_perturbed_m = outputs_m['Cd'][0]
                    
            #         first_derivative_numpy[0, j] = (Cd_perturbed_p - Cd_perturbed_m) / 2 / delta_x
                

            #     derivatives_list.append(first_derivative_numpy)