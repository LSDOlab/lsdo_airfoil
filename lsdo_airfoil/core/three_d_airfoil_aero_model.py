import numpy as np
import torch
import csdl_alpha as csdl
from lsdo_airfoil.core.generate_xfoil_data import run_xfoil
from lsdo_airfoil.core.airfoil_training import train_three_d_airfoil_model, train_three_d_airfoil_model_vector_valued
import os
from typing import Union
from scipy.sparse import block_diag
from lsdo_airfoil import UIUC_AIRFOILS


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)


class ThreeDAirfoilMLModelMaker:
    def __init__(
        self,
        airfoil_name: str,
        aoa_range: Union[list, np.ndarray],
        reynolds_range: Union[list, np.ndarray],
        mach_range: Union[list, np.ndarray],
        transition_top="free",
        transition_bottom="free",
        pane: int=250,
        num_interp: int = 120,
        x_spacing: str = "sin",
        power = 2,
        save_xfoil_data: bool = True, 
        force_regenerate_xfoil_data: bool = False,
    ) -> None:
        csdl.check_parameter(airfoil_name, "airfoil_name", types=str)
        self.airfoil_name = airfoil_name
        self.aoa_range = aoa_range
        self.reynolds_range = reynolds_range
        self.mach_range = mach_range
        self.num_interp = 120
        self.x_spacing = x_spacing

        if x_spacing == "linear":
            self.x_interp = np.linspace(0., 1., num_interp)
        elif x_spacing == "power":
            self.x_interp = np.linspace(0., 1, num_interp)**power
        elif x_spacing == "sin":
            self.x_interp = 0.5 + 0.5*np.sin(np.pi*(np.linspace(0., 1., num_interp)-0.5))

        # Run xfoil or load saved results
        self.coeffs, self.Cp_data, self.Ue_data, \
            self.Cf_data, self.delta_star_data, self.theta_data, \
                self.shape_parameter_data, self.inputs, self.parent_dir = run_xfoil(
            airfoil=airfoil_name, 
            aoa_range=aoa_range, 
            reynolds_range=reynolds_range, 
            mach_range=mach_range,
            transition_bottom=transition_bottom,
            transition_top=transition_top,
            pane=pane,
            save_data=save_xfoil_data,
            x_spacing=x_spacing,
            force_regenerate_data=force_regenerate_xfoil_data,
        )

        if os.path.isdir('regressions'):
            pass
        else:
            os.makedirs('regressions')


    def get_airfoil_model(
        self,
        quantities: str,
        tune_hyper_parameters: bool=False,
        num_trials: int = 500,
        force_retrain: bool = False,
    ):
        available_quantities =  ["Cl", "Cd", "Cp", "Ue", "Cf", "theta", "delta_star", 
                            "shape_parameter", "all", "scalar_valued", "vector_valued"]
        if not all([quantity for quantity in available_quantities]):
            raise ValueError(f"Unknown quantity. Available quantities are {available_quantities}")
        
        model_list = []
        out_shape_list = []
        X_max_list = []
        X_min_list = []

        for quantity in quantities:
            if quantity == "Cl":
                model, X_max, X_min = train_three_d_airfoil_model(
                    raw_inputs=self.inputs,
                    raw_outputs=self.coeffs,
                    Re_range=self.reynolds_range,
                    Ma_range=self.mach_range,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    tune_hyper_parameters=tune_hyper_parameters,
                    quantity=quantity,
                    num_trials=num_trials,
                    force_retrain=force_retrain,
                )
            elif quantity == "Cd":
                model, X_max, X_min = train_three_d_airfoil_model(
                    raw_inputs=self.inputs,
                    raw_outputs=self.coeffs,
                    Re_range=self.reynolds_range,
                    Ma_range=self.mach_range,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    tune_hyper_parameters=tune_hyper_parameters,
                    quantity=quantity,
                    num_trials=num_trials,
                    force_retrain=force_retrain,
                )

            elif quantity == "alpha_Cl_min_max":
                model, X_max, X_min = train_three_d_airfoil_model(
                    raw_inputs=self.inputs,
                    raw_outputs=self.coeffs,
                    Re_range=self.reynolds_range,
                    Ma_range=self.mach_range,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    tune_hyper_parameters=tune_hyper_parameters,
                    quantity=quantity,
                    num_trials=num_trials,
                    force_retrain=force_retrain,
                )

            elif quantity == "Cp":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.Cp_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )

            elif quantity == "Cf":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.Cf_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )

            elif quantity == "Ue":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.Ue_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )

            elif quantity == "delta_star":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.delta_star_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )

            elif quantity == "theta":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.theta_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )

            elif quantity == "shape_parameter":
                model, X_max, X_min = train_three_d_airfoil_model_vector_valued(
                    raw_inputs=self.inputs,
                    raw_outputs=self.shape_parameter_data,
                    data_directory_path=f"{UIUC_AIRFOILS}/{self.airfoil_name}/regressions",
                    quantity=quantity,
                    tune_hyper_parameters=tune_hyper_parameters,
                    force_retrain=force_retrain,
                    num_trials=num_trials,
                )
            
            else:
                raise NotImplementedError("Unkown airfoil model output quantity")

            if quantity in ["Cl", "Cd"]:
                out_shape = (1, )
            elif quantity == "alpha_Cl_min_max":
                out_shape = (2, )
            else:
                out_shape = (2 * self.num_interp, )

            out_shape_list.append(out_shape)
            model_list.append(model)
            X_max_list.append(X_max)
            X_min_list.append(X_min)


        airfoil_model = AirfoilModel(
            model_list=model_list,
            X_min_list=X_min_list,
            X_max_list=X_max_list,
            quantities=quantities,
            out_shape_list=out_shape_list,
        )

        os.chdir(self.parent_dir)

        return airfoil_model

class AirfoilModel:
    def __init__(
        self, model_list, X_min_list, X_max_list, quantities, out_shape_list,
    ):
        self.model_list = model_list
        self.X_min_list = X_min_list
        self.X_max_list = X_max_list
        self.quantities = quantities
        self.out_shape_list = out_shape_list
    
    def evaluate(self, alpha, Re, Ma):
        outputs = []
        for i, model in enumerate(self.model_list):
            ml_custom_operation = MLAirfoilCustomOp(
                model=self.model_list[i],
                X_min=self.X_min_list[i],
                X_max=self.X_max_list[i],
                quantity=self.quantities[i],
                out_shape=self.out_shape_list[i],
            )

            outputs.append(ml_custom_operation.evaluate(alpha, Re, Ma))

        if len(self.model_list) == 1:
            return outputs[0]
        else:
            return tuple(outputs)
    
class MLAirfoilCustomOp(csdl.CustomExplicitOperation):
    def __init__(self, model, X_min, X_max, quantity, out_shape):
        self.model = model
        self.X_min = X_min
        self.X_max = X_max
        self.quantity = quantity
        self.out_shape = out_shape
        super().__init__()

    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)

        input_shape = alpha.shape
        if len(input_shape) == 3:
            indices = np.arange(input_shape[0] * input_shape[1] * input_shape[2])

        elif len(input_shape) == 2:
            indices = np.arange(input_shape[0] * input_shape[1])

        elif len(input_shape) == 1:
            indices = np.arange(input_shape[0])

        else:
            raise NotImplementedError

        if self.out_shape == (1, ):
            output = self.create_output(self.quantity, shape=input_shape)
            self.declare_derivative_parameters(self.quantity, "alpha", rows=indices, cols=indices)
            self.declare_derivative_parameters(self.quantity, "Re", rows=indices, cols=indices)
            self.declare_derivative_parameters(self.quantity, "Ma", rows=indices, cols=indices)
        elif self.out_shape == (2, ):
            output = self.create_output(self.quantity, shape=input_shape+ self.out_shape)
            self.declare_derivative_parameters(self.quantity, "alpha", rows=indices, cols=indices, dependent=False)
            self.declare_derivative_parameters(self.quantity, "Re")
            self.declare_derivative_parameters(self.quantity, "Ma")
        else:
            output = self.create_output(self.quantity, input_shape + self.out_shape)
            self.declare_derivative_parameters(self.quantity, "alpha", sparse=True)#, rows=np.arange(480), cols=np.arange(480))
            self.declare_derivative_parameters(self.quantity, "Re", sparse=True)#, rows=np.arange(480), cols=np.arange(480))#
            self.declare_derivative_parameters(self.quantity, "Ma", sparse=True)#, rows=np.arange(480), cols=np.arange(480))#

        return output
    
    def compute(self, input_vals, output_vals):
        model = self.model

        alpha = input_vals["alpha"]
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        input_shape = alpha.shape
        num_nodes = alpha.flatten().shape[0]

        if self.quantity == "alpha_Cl_min_max":
            input_tensor = np.zeros((num_nodes, 2))
            input_tensor[:, 0] = Re.flatten()
            input_tensor[:, 1] = Ma.flatten()

            scaled_input_tensor = (input_tensor - self.X_min[0, 1:]) \
                / (self.X_max[0, 1:] - self.X_min[0, 1:])
            input_tensor_torch = torch.Tensor(scaled_input_tensor).to(device)
        else:
            input_tensor = np.zeros((num_nodes, 3))
            input_tensor[:, 0] = alpha.flatten()
            input_tensor[:, 1] = Re.flatten()
            input_tensor[:, 2] = Ma.flatten()

            scaled_input_tensor = (input_tensor - self.X_min) \
                / (self.X_max - self.X_min)
            input_tensor_torch = torch.Tensor(scaled_input_tensor).to(device)

        output = model(input_tensor_torch).cpu().detach().numpy()

        

        if self.out_shape == (1, ):
            output_vals[self.quantity] = output.reshape(input_shape)
        else:
            output_vals[self.quantity] = output.reshape(input_shape + self.out_shape)

    def compute_derivatives(self, input_vals, outputs, derivatives):
        model = self.model

        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        num_nodes = Re.flatten().shape[0]
        
        if self.quantity == "alpha_Cl_min_max":
            input_tensor = np.zeros((num_nodes, 2))
            input_tensor[:, 0] = Re.flatten()
            input_tensor[:, 1] = Ma.flatten()

            scaled_input_tensor = (input_tensor - self.X_min[0, 0:2]) \
                / (self.X_max[0, 0:2] - self.X_min[0, 0:2])
            input_tensor_torch = torch.Tensor(scaled_input_tensor).to(device)

        else:
            alpha = input_vals["alpha"]
            input_tensor = np.zeros((num_nodes, 3))
            input_tensor[:, 0] = alpha.flatten()
            input_tensor[:, 1] = Re.flatten()
            input_tensor[:, 2] = Ma.flatten()
            
            scaled_input_tensor = (input_tensor - self.X_min) \
                / (self.X_max - self.X_min)
            input_tensor_torch = torch.Tensor(scaled_input_tensor).to(device)

        if self.out_shape == (1, ):
            # d_model_d_scaled_tensor = torch.autograd.functional.jacobian(model, input_tensor_torch).cpu().detach().numpy()
            d_model_d_scaled_tensor = torch.func.vmap(torch.func.jacrev(model))(input_tensor_torch).cpu().detach().numpy()
            # d_model_d_scaled_tensor = d_model_d_scaled_tensor.reshape((num_nodes, num_nodes, 3))

            d_scaled_tensor_d_tensor = (1/(self.X_max - self.X_min)).reshape((1, 3))
            # d_tensor_d_inputs = np.ones((num_nodes, 3))

            d_model_d_inputs = np.einsum('ijk, lk->ijk', d_model_d_scaled_tensor, d_scaled_tensor_d_tensor).reshape((-1, 3))
            # d_model_d_tensor = np.einsum('ijk, lk->ijk', d_model_d_scaled_tensor, d_scaled_tensor_d_tensor)
            # d_model_d_inputs = np.einsum('ijk, jk->ik', d_model_d_tensor, d_tensor_d_inputs)

            derivatives[self.quantity, "alpha"] = d_model_d_inputs[:, 0]
            derivatives[self.quantity, "Re"] = d_model_d_inputs[:, 1]
            derivatives[self.quantity, "Ma"] = d_model_d_inputs[:, 2]

        else:
            d_model_d_scaled_tensor = torch.func.vmap(torch.func.jacfwd(model))(input_tensor_torch).cpu().detach().numpy()
            if self.out_shape == (2, ):
                if len(self.X_max.shape) == 1:
                    d_scaled_tensor_d_tensor = (1/(self.X_max[0:2] - self.X_min[0:2])).reshape((1, 2))
                else:
                    d_scaled_tensor_d_tensor = (1/(self.X_max[:,0:2] - self.X_min[:,0:2])).reshape((1, 2))
                d_model_d_inputs = np.einsum('ijk, mk->ijk', d_model_d_scaled_tensor, d_scaled_tensor_d_tensor)

                derivatives[self.quantity, "Re"] = block_diag(tuple([block.reshape((2, 1)) for block in d_model_d_inputs[:, :, 0]]))
                derivatives[self.quantity, "Ma"] = block_diag(tuple([block.reshape((2, 1)) for block in d_model_d_inputs[:, :, 1]]))

            else:
                d_scaled_tensor_d_tensor = (1/(self.X_max - self.X_min)).reshape((1, 3))
                d_model_d_inputs = np.einsum('ijk, mk->ijk', d_model_d_scaled_tensor, d_scaled_tensor_d_tensor)

                derivatives[self.quantity, "alpha"] = block_diag(tuple([block.reshape((240, 1)) for block in d_model_d_inputs[:, :, 0]]))
                derivatives[self.quantity, "Re"] = block_diag(tuple([block.reshape((240, 1)) for block in d_model_d_inputs[:, :, 1]]))
                derivatives[self.quantity, "Ma"] = block_diag(tuple([block.reshape((240, 1)) for block in d_model_d_inputs[:, :, 2]]))
