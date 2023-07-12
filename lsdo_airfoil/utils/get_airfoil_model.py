import torch
import numpy as np
from lsdo_airfoil.core.models.scalar_valued_regressions.scalar_valued_neural_nets import scaler_valued_nn_model, scaler_valued_nn_model_gelu
from lsdo_airfoil.core.models.vector_valued_regressions.vector_valued_neural_nets import LSTM
from lsdo_airfoil import MODELS_FOLDER
import copy
from torch import nn


# cl_model = nn.Sequential(
#             nn.Linear(35, 180), nn.ReLU(),
#             nn.Linear(180, 180), nn.ReLU(),
#             nn.Linear(180, 160), nn.SELU(),
#             nn.Linear(160, 140), nn.SELU(),
#             nn.Linear(140, 120), nn.SELU(),
#             nn.Linear(120, 120), nn.ReLU(),
#             nn.Linear(120, 100), nn.GELU(),
#             nn.Linear(100, 80), nn.ReLU(),
#             nn.Linear(80, 50), nn.ReLU(),
#             nn.Linear(50, 1),
# )


# cl_model = nn.Sequential(
#             nn.Linear(35, 180), nn.ReLU(),
#             nn.Linear(180, 160), nn.SELU(),
#             nn.Linear(160, 140), nn.SELU(),
#             nn.Linear(140, 120), nn.SELU(),
#             nn.Linear(120, 120), nn.ReLU(),
#             nn.Linear(120, 100), nn.GELU(),
#             nn.Linear(100, 80), nn.LeakyReLU(),
#             nn.Linear(80, 1),
# )

cl_model = cd_model = nn.Sequential(
            nn.Linear(35, 140), nn.ReLU(),
            nn.Linear(140, 180), nn.SELU(),
            nn.Linear(180, 180), nn.GELU(),
            nn.Linear(180, 140), nn.ReLU(),
            nn.Linear(140, 120), nn.SELU(),
            nn.Linear(120, 120), nn.ReLU(),
            nn.Linear(120, 100), nn.GELU(),
            nn.Linear(100, 80), nn.LeakyReLU(),
            nn.Linear(80, 50), nn.ReLU(),
            nn.Linear(50, 1),
)

neural_net_model_dict = dict()
input_dim = 35
hidden_dim_rnn = 128
output_dim_rnn = 100
num_layers_rnn = 1

def get_airfoil_models():    
    scalar_valued_models = ['Cd'] #['Cl', 'Cd']#, 'Cm']
    vector_valued_models = ['Cp', 'Ue', 'delta_star', 'theta']
    
    cl_net = cl_model
    cl_net.load_state_dict(torch.load(MODELS_FOLDER / f'scalar_valued_regressions/Cl_num_epoch_1300_batch_size_250_lr_0001_NEW_2nd_extrapolated_data_attempt_2'))
    cl_net.eval()
    cl_net.requires_grad_(False)
    neural_net_model_dict['Cl'] = copy.deepcopy(cl_net)

    cd_net = cd_model
    cd_net.load_state_dict(torch.load(MODELS_FOLDER / f'scalar_valued_regressions/Cd_num_epoch_1000_batch_size_250_lr_0001_NEW_2nd_extrapolated_data_attempt_1'))
    cd_net.eval()
    cd_net.requires_grad_(False)
    neural_net_model_dict['Cd'] = copy.deepcopy(cd_net)

    # for model in scalar_valued_models:
    #     neural_net_model = scaler_valued_nn_model
    #     neural_net_model.load_state_dict(torch.load(MODELS_FOLDER / f'scalar_valued_regressions/{model}_model', map_location=torch.device('cpu')))
    #     neural_net_model.eval()
    #     neural_net_model.requires_grad_(False)
    #     neural_net_model_dict[model] = copy.deepcopy(neural_net_model)
    

    for model in vector_valued_models:
        neural_net_upper_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim_rnn, output_dim=output_dim_rnn, num_layers=num_layers_rnn)
        neural_net_lower_model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim_rnn, output_dim=output_dim_rnn, num_layers=num_layers_rnn)
        neural_net_upper_model.eval()
        neural_net_lower_model.eval()
        neural_net_upper_model.requires_grad_(False)
        neural_net_lower_model.requires_grad_(False)

        neural_net_upper_model.load_state_dict(torch.load(MODELS_FOLDER / f'vector_valued_regressions/{model}_upper_model', map_location=torch.device('cpu')))
        neural_net_lower_model.load_state_dict(torch.load(MODELS_FOLDER / f'vector_valued_regressions/{model}_lower_model', map_location=torch.device('cpu')))

        neural_net_model_dict[f"{model}_upper"] = copy.deepcopy(neural_net_upper_model)
        neural_net_model_dict[f"{model}_lower"] = copy.deepcopy(neural_net_lower_model)
        
    return neural_net_model_dict



