import os
import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import optuna
from sklearn.model_selection import train_test_split
import pickle


# Run on GPU if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_default_dtype(torch.float64)

def define_model(
        trial: optuna.trial, 
        in_features_data : int=3,
        out_features_data: int=1,
        max_layers: int=6,
        min_units: int=5,
        max_units: int=100,
        add_drop_out: bool=False,
    
    ):
    n_layers = trial.suggest_int("n_layers", 2, max_layers)
    layers = []
    in_features = in_features_data

    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), min_units, max_units)
        layers.append(nn.Linear(in_features, out_features))
        activation_fun = trial.suggest_categorical(f"activation_fun_{i}", ["ReLU", "LeakyReLU", "GELU", "SELU", "Tanh", "Softplus"])
        layers.append(getattr(nn, activation_fun)())
        if add_drop_out:
            p = trial.suggest_float("dropout_l{}".format(i), 0., 0.5)
            layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, out_features_data))

    return nn.Sequential(*layers)

def train_three_d_airfoil_model_vector_valued(
    raw_inputs,
    raw_outputs,
    data_directory_path,
    quantity,
    tune_hyper_parameters,
    num_trials,
    force_retrain=False,
):
    xt = raw_inputs
    yt = raw_outputs

    # Convert to aoa to radians
    xt[:, 0] = np.deg2rad(xt[:, 0])

    output_data_dim = yt.shape[1]

    if os.path.isfile(f"{data_directory_path}/{quantity}_model"):
        if force_retrain:
            print(f":::::::::::::::::::::::::::TRAINING {quantity} MODEL:::::::::::::::::::::::::::")
            model = get_ml_model(xt, yt, data_directory_path, quantity, tune_hyper_parameters, num_trials)
            model.eval()

        else:
            if os.path.isfile(f"{data_directory_path}/tuned_{quantity}_model_params.pickle"):
                with open(f"{data_directory_path}/tuned_{quantity}_model_params.pickle", "rb") as pickle_file:
                    params_dict = pickle.load(pickle_file)
            else:
                with open(f"{data_directory_path}/general_vector_valued_model_params.pickle", "rb") as pickle_file:
                    params_dict = pickle.load(pickle_file)

            model = build_model_from_parameters(params_dict, None, None, None, None, None, None, train=False, output_data_dim=output_data_dim)
            model.eval()
            model.load_state_dict(
                torch.load(f"{data_directory_path}/{quantity}_model", map_location=torch.device("cpu"))
            )

    else:
        print(f":::::::::::::::::::::::::::TRAINING {quantity} MODEL:::::::::::::::::::::::::::")
        model = get_ml_model(xt, yt, data_directory_path, quantity, tune_hyper_parameters, num_trials)
        model.eval()

    X_max = np.max(raw_inputs, axis=0, keepdims=True) 
    X_min = np.min(raw_inputs, axis=0, keepdims=True)

    return model, X_max, X_min

def train_three_d_airfoil_model(
        quantity, raw_inputs, raw_outputs, Re_range, Ma_range, 
        data_directory_path, tune_hyper_parameters, 
        num_trials, force_retrain=False, plot=False,
    ):
    training_inputs = np.zeros((1, 3))
    training_outputs = np.zeros((1, 2))
    if plot:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    inputs_list = []
    training_inputs_alpha_Cl_min_max = np.zeros((1, 2))
    training_outputs_alpha_Cl_min_max = np.zeros((1, 2))
    counter = 0
    for i in range(len(Re_range)):
        for j in range(len(Ma_range)):
            # Parse input data into sweeps across Mach and Reynolds
            condition_1 = raw_inputs[:, -1] == Ma_range[j]
            condition_2 = raw_inputs[:, -2] == Re_range[i]

            combined_condition = np.logical_and.reduce((condition_1, condition_2))
            filter = np.where(combined_condition)[0]

            if len(filter) > 0:
                aoa_raw = np.deg2rad(raw_inputs[filter, -3])

                Cl = np.array([x for _, x in sorted(zip(aoa_raw, raw_outputs[filter, 0]))])
                Cd = np.array([x for _, x in sorted(zip(aoa_raw, raw_outputs[filter, 1]))]) 
                aoa_sorted = np.sort(aoa_raw)

                Cl_stall_p = np.max(Cl)
                Cl_stall_m = np.min(Cl)

                aoa_stall_p = aoa_sorted[np.where(Cl==Cl_stall_p)[0][0]]
                Cd_stall_p = Cd[np.where(Cl==Cl_stall_p)[0][0]]

                aoa_stall_m = aoa_sorted[np.where(Cl==Cl_stall_m)[0]][0]
                Cd_stall_m = Cd[np.where(Cl==Cl_stall_m)[0][0]]

                inputs_alpha_Cl_min_max = np.zeros((1, 2))
                inputs_alpha_Cl_min_max[0, 0] = Re_range[i]
                inputs_alpha_Cl_min_max[0, 1] = Ma_range[j]

                outputs_alpha_Cl_min_max = np.zeros((1, 2))
                outputs_alpha_Cl_min_max[0, 0] = aoa_stall_m
                outputs_alpha_Cl_min_max[0, 1] = aoa_stall_p
                training_inputs_alpha_Cl_min_max = np.vstack((training_inputs_alpha_Cl_min_max, inputs_alpha_Cl_min_max))
                training_outputs_alpha_Cl_min_max = np.vstack((training_outputs_alpha_Cl_min_max, outputs_alpha_Cl_min_max))

                pre_stall_p = aoa_sorted < aoa_stall_p
                pre_stall_m = aoa_sorted > aoa_stall_m

                pre_stall_filter = np.logical_and.reduce((pre_stall_p, pre_stall_m))

                aoa_pre_stall = aoa_sorted[pre_stall_filter]
                Cl_pre_stall = Cl[pre_stall_filter]
                Cd_pre_stall = Cd[pre_stall_filter]

                AR = 10
                Cd_max = 1.11 + 0.018 * AR
                A1 = Cd_max / 2
                B1 = Cd_max
                A2_p = (Cl_stall_p - Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * (np.sin(aoa_stall_p)/np.cos(aoa_stall_p)**2)
                A2_m = (Cl_stall_m - Cd_max * np.sin(aoa_stall_m) * np.cos(aoa_stall_m)) * (np.sin(aoa_stall_m)/np.cos(aoa_stall_m)**2)

                B2_p = (Cd_stall_p - Cd_max * np.sin(aoa_stall_p)**2) / (np.cos(aoa_stall_p))
                B2_m = (Cd_stall_m - Cd_max * np.sin(aoa_stall_m)**2) / (np.cos(aoa_stall_m))


                num_post_stall_points = 15
                i_vec = np.arange(0, num_post_stall_points)
                half_cos = 1 - np.cos(i_vec * np.pi / (2 * (num_post_stall_points - 1)))

                aoa_post_stall_plus = aoa_stall_p + np.deg2rad(0.5) - (aoa_stall_p + np.deg2rad(0.5) - np.deg2rad(90)) * half_cos
                aoa_post_stall_minus = np.flip(aoa_stall_m - np.deg2rad(0.5)  - (aoa_stall_m - np.deg2rad(0.5) + np.deg2rad(90)) * half_cos)

                Cl_post_stall_m = A1 * np.sin(2 * aoa_post_stall_minus) + A2_m * np.cos(aoa_post_stall_minus)**2 / np.sin(aoa_post_stall_minus)
                Cl_post_stall_p = A1 * np.sin(2 * aoa_post_stall_plus) + A2_p * np.cos(aoa_post_stall_plus)**2 / np.sin(aoa_post_stall_plus)

                Cd_post_stall_m = B1 * np.sin(aoa_post_stall_minus)**2 +  B2_m * np.cos(aoa_post_stall_minus)
                Cd_post_stall_p = B1 * np.sin(aoa_post_stall_plus)**2 +  B2_p * np.cos(aoa_post_stall_plus)

                aoa_total = np.hstack((aoa_post_stall_minus, aoa_pre_stall, aoa_post_stall_plus)).flatten()
                Cl_total = np.hstack((Cl_post_stall_m, Cl_pre_stall, Cl_post_stall_p)).flatten()
                Cd_total = np.hstack((Cd_post_stall_m, Cd_pre_stall, Cd_post_stall_p)).flatten()

                if plot:
                    color = plt.cm.tab10(counter)

                    axs[0, 0].scatter(np.rad2deg(aoa_total), Cl_total, color=color, s=5)
                    axs[0, 0].set_xlabel("angle of attack (deg)")
                    axs[0, 0].set_ylabel("Cl")
                    
                    axs[0, 1].scatter(np.rad2deg(aoa_total), Cl_total, color=color, s=5)
                    axs[0, 1].set_xlabel("angle of attack (deg)")
                    axs[0, 1].set_ylabel("Cl")
                    axs[0, 1].set_xlim([-15, 20])

                    axs[1, 0].scatter(np.rad2deg(aoa_total), Cd_total, color=color, s=5)
                    axs[1, 0].set_xlabel("angle of attack (deg)")
                    axs[1, 0].set_ylabel("Cd")

                    axs[1, 1].scatter(np.rad2deg(aoa_total), Cd_total, color=color, s=5)
                    axs[1, 1].set_xlabel("angle of attack (deg)")
                    axs[1, 1].set_ylabel("Cd")
                    axs[1, 1].set_ylim([0, 0.05])
                    axs[1, 1].set_xlim([-15, 20])

                # assemble training data
                inputs = np.zeros((len(aoa_total), 3))
                outputs = np.zeros((len(aoa_total), 2))

                inputs[:, 0] = aoa_total
                inputs[:, 1] = Re_range[i]
                inputs[:, 2] = Ma_range[j]
                if True:
                    inputs_list.append(inputs)

                outputs[:, 0] = Cl_total
                outputs[:, 1] = Cd_total

                training_inputs = np.vstack((training_inputs, inputs))
                training_outputs = np.vstack((training_outputs, outputs))

                counter += 1

    xt = training_inputs[1:, :]
    yt_Cl = training_outputs[1:, 0]
    yt_Cd = training_outputs[1:, 1]

    xt_alpha_Cl_min_max = training_inputs_alpha_Cl_min_max[1:, :]
    yt_alpha_Cl_min_max = training_outputs_alpha_Cl_min_max[1:, :]

    if os.path.isfile(f"{data_directory_path}/{quantity}_model"):
        if force_retrain:
            print(f":::::::::::::::::::::::::::TRAINING {quantity} MODEL:::::::::::::::::::::::::::")
            if quantity == "Cl":
                model = get_ml_model(xt, yt_Cl, data_directory_path, "Cl", tune_hyper_parameters, num_trials)
                model.eval()
            elif quantity == "Cd":
                model = get_ml_model(xt, yt_Cd, data_directory_path, "Cd", tune_hyper_parameters, num_trials)
                model.eval()
            elif quantity == "alpha_Cl_min_max":
                model = get_ml_model(xt_alpha_Cl_min_max, yt_alpha_Cl_min_max, data_directory_path, "alpha_Cl_min_max", tune_hyper_parameters, num_trials)
                model.eval()
            else:
                raise NotImplementedError


        else:
            if os.path.isfile(f"{data_directory_path}/tuned_{quantity}_model_params.pickle"):
                with open(f"{data_directory_path}/tuned_{quantity}_model_params.pickle", "rb") as pickle_file:
                    params_dict = pickle.load(pickle_file)
            else:
                raise NotImplementedError(f"No tuned hyper parameters for model {quantity}. Set 'tune_hyper_parameters' to true to get optimal parameters.")
                # with open(f"{data_directory_path}/../general_Cl_model_params.pickle", "rb") as pickle_file:
                #     params_dict = pickle.load(pickle_file)

            model = build_model_from_parameters(params_dict, None, None, None, None, None, quantity, train=False)
            model.eval()
            model.load_state_dict(torch.load(f"{data_directory_path}/{quantity}_model", map_location=torch.device("cpu")))
            
    else:
        print(f":::::::::::::::::::::::::::TRAINING {quantity} MODEL:::::::::::::::::::::::::::")
        if quantity == "Cl":
            model = get_ml_model(xt, yt_Cl, data_directory_path, "Cl", tune_hyper_parameters, num_trials)
            model.eval()
        elif quantity == "Cd":
            model = get_ml_model(xt, yt_Cd, data_directory_path, "Cd", tune_hyper_parameters, num_trials)
            model.eval()
        elif quantity == "alpha_Cl_min_max":
            model = get_ml_model(xt_alpha_Cl_min_max, yt_alpha_Cl_min_max, data_directory_path, "alpha_Cl_min_max", tune_hyper_parameters, num_trials)
            model.eval()
        else:
            raise NotImplementedError


    X_max = np.max(training_inputs[1:, :], axis=0, keepdims=True) 
    X_min = np.min(training_inputs[1:, :], axis=0, keepdims=True) 

    if plot:
        for i, inputs in enumerate(inputs_list):
            color = plt.cm.tab10(i)
            aoa_total = inputs[:, 0]

            inputs_scaled = (inputs - X_min) / (X_max - X_min)
            inputs_torch = torch.tensor(inputs_scaled, dtype=torch.float64).to(device)

            if quantity == "Cl":
                Cl = model(inputs_torch).cpu().detach().numpy().flatten()
                axs[0, 0].plot(np.rad2deg(aoa_total), Cl, color=color)
                axs[0, 0].set_xlabel("angle of attack (deg)")
                axs[0, 0].set_ylabel("Cl")
                
                axs[0, 1].plot(np.rad2deg(aoa_total), Cl, color=color)
                axs[0, 1].set_xlabel("angle of attack (deg)")
                axs[0, 1].set_ylabel("Cl")
                axs[0, 1].set_xlim([-15, 20])

            elif quantity == "Cd":
                Cd = model(inputs_torch).cpu().detach().numpy().flatten()
                axs[1, 0].plot(np.rad2deg(aoa_total), Cd, color=color)
                axs[1, 0].set_xlabel("angle of attack (deg)")
                axs[1, 0].set_ylabel("Cd")

                axs[1, 1].plot(np.rad2deg(aoa_total), Cd, color=color)
                axs[1, 1].set_xlabel("angle of attack (deg)")
                axs[1, 1].set_ylabel("Cd")
                axs[1, 1].set_ylim([0, 0.05])
                axs[1, 1].set_xlim([-15, 20])

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{data_directory_path}/airfoil_model_plot.png')
        plt.show()

    return model, X_max, X_min
    
def get_ml_model(input_data, output_data, data_directory_path, type_="Cl", tune_hyper_parameters=False, num_trials=500):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(input_data, output_data, train_size=0.70, shuffle=True)

    if len(y_test.shape) == 1:
        output_data_dim = 1
    else:
        output_data_dim = y_test.shape[1]

    # Normalizing data
    X_max = np.max(X_train_raw, axis=0, keepdims=True) 
    X_min = np.min(X_train_raw, axis=0, keepdims=True) 

    X_train = (X_train_raw - X_min) / (X_max - X_min)
    X_test = (X_test_raw - X_min) / (X_max - X_min)

    if type_ in ["Cl", "Cd"]:
        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float64).reshape(-1, 1).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float64).reshape(-1, 1).to(device)

    else:
        # Convert to 2D PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
        y_train = torch.tensor(y_train, dtype=torch.float64).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float64).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float64).to(device)

    def get_data(batch_size):
        train_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            list(zip(X_test, y_test)),
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader, valid_loader

    if tune_hyper_parameters:
        if type_ not in ["alpha_Cl_min_max", "Cl", "Cd"]:
            in_features = 3
            out_features_data = output_data.shape[1]
            max_layers = 10
            min_units = 25
            max_units = 250
            add_drop_out = True
        elif type_ == "alpha_Cl_min_max":
            in_features = 2
            out_features_data = 2
            max_layers = 6
            min_units = 5
            max_units = 100
            add_drop_out = False
        else:
            in_features = 3
            out_features_data = 1
            max_layers = 6
            min_units = 5
            max_units = 100
            add_drop_out = False
        # define objective function for optuna hyper parameter tuning
        def objective(trial):
            model = define_model(
                trial,
                in_features_data=in_features,
                out_features_data=out_features_data,
                max_layers=max_layers,
                max_units=max_units,
                min_units=min_units,
                add_drop_out=add_drop_out,
            ).to(device)

            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW", "Adagrad"])
            lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-5, log=True)
            optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

            gamma = trial.suggest_float("gamma", 0.5, 0.99)
            step_size = trial.suggest_int("step_size", 20, 200)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            epochs = 1000

            batch_size = trial.suggest_int("batch_size", 5, 20)
            train_loader, valid_loader = get_data(batch_size)

            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.view(data.size(0), -1).to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                
                    if type_ == "alpha_Cl_min_max":
                        mask = 0
                    else:
                        Re = data[:, 1] * (10e6 - 1e5) + 1e5
                        Ma = data[:, 2] * 0.6 
                        aoa = (data[:, 0] * (np.pi/2 + np.pi/2) - np.pi/2) * 180 / np.pi

                        mask = (aoa >= -5) & (aoa <=8)
                        mask = mask.float()

                    if type_ == "Cd":
                        penalty = torch.mean(torch.relu(-output))
                    else:
                        penalty = 0

                    loss = F.mse_loss(output, target) + 10 * penalty
                    if mask == 0:
                        weighted_loss = loss
                    else:
                        weighted_loss = ((1 + (10 -1) * mask) * loss).mean() 

                    weighted_loss.backward()
                    optimizer.step()

                # Validation of the model.
                scheduler.step()
                model.eval()
                y_pred = model(X_test)
                mse_test = F.mse_loss(y_pred, y_test)
                trial.report(mse_test, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return mse_test
        
        print(":::::::::::::::::::::::::::TUNING HYPER PARAMETERS (MAY TAKE UP TO A FEW HOURS):::::::::::::::::::::::::::")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=num_trials, timeout=12* 3600)

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        with open(f"{data_directory_path}/tuned_{type_}_model_params.pickle", "wb") as pickle_file:
            pickle.dump(trial.params, pickle_file)

        model = build_model_from_parameters(trial.params, X_train, X_test, y_train, y_test, data_directory_path, type_, output_data_dim=output_data_dim)

    else:
        if os.path.isfile(f"{data_directory_path}/tuned_{type_}_model_params.pickle"):
            with open(f"{data_directory_path}/tuned_{type_}_model_params.pickle", "rb") as pickle_file:
                params_dict = pickle.load(pickle_file)
        else:
            raise NotImplementedError("Need to tune hyper parameters first.")
            # with open(f"{data_directory_path}/../general_{type_}_model_params.pickle", "rb") as pickle_file:
            #     params_dict = pickle.load(pickle_file)
        
        model = build_model_from_parameters(params_dict, X_train, X_test, y_train, y_test, data_directory_path, type_, output_data_dim=output_data_dim)
    
    return model

def build_model_from_parameters(params_dict, X_train, X_test, y_train, y_test, data_directory_path, type_, output_data_dim=None, train=True):
    if output_data_dim is not None:
        output_dim = output_data_dim
    elif type_ == "alpha_Cl_min_max":
        output_dim = 2
    else:
        output_dim = 1


    output_data_dim = y_test
    n_unints = []
    activation_fun = []
    dropout = []
    for key, value in params_dict.items():
        if "n_units" in key:
            n_unints.append(value)
        # elif "dropout" in key:
        #     dropout.append(value)
        elif "activation_fun" in key:
            activation_fun.append(value)

    n_layers = len(n_unints)
    layers = []
    if type_ == "alpha_Cl_min_max":
        in_features = 2
    else:
        in_features = 3
    for j in range(n_layers):
        out_features = n_unints[j]
        layers.append(nn.Linear(in_features, out_features))
        activation_fun_name = activation_fun[j]
        layers.append(getattr(nn, activation_fun_name)())
        if dropout:
            p = dropout[j]
            layers.append(nn.Dropout(p))
        in_features = out_features


    layers.append(nn.Linear(in_features, output_dim))

    model = nn.Sequential(*layers).to(device)

    if train:
        criterion = nn.MSELoss()

        # Other hyperparameters
        optimizer_name = params_dict["optimizer"]
        lr = params_dict["lr"]
        weight_decay = params_dict["weight_decay"]
        batch_size = params_dict["batch_size"]
        scheduler_step_size = params_dict["step_size"]
        gamma = params_dict["gamma"]
        
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

        input_seq = X_train
        output_seq = y_train

        if type_ == "alpha_Cl_min_max":
            input_batches = input_seq.unfold(1, 2, 1)
        else:
            input_batches = input_seq.unfold(1, 3, 1)

        output_batches = output_seq.unfold(1, output_dim, 1)

        num_epochs = 1000
        for epoch in range(num_epochs):
            # shuffle batches
            perm = torch.randperm(input_batches.size()[0])
            input_batches = input_batches[perm]
            output_batches = output_batches[perm]
            
            # loop over batches
            for k in range(0, input_batches.size()[0], batch_size):
                # get batch
                input_batch = input_batches[k:k+batch_size]
                output_batch = output_batches[k:k+batch_size]
                
                # zero gradients
                optimizer.zero_grad()
                
                # forward pass
                output_pred = model(input_batch)

                aoa = (input_batch[:, 0, 0] * (np.pi/2 + np.pi/2) - np.pi/2) * 180 / np.pi

                mask = (aoa >= -5) & (aoa <=8)
                mask = mask.float()
                

                if type_ == "Cd":
                    penalty = torch.mean(torch.relu(-output_pred[:, :, :]))
                else:
                    penalty = 0

                # compute loss
                loss = criterion(output_pred, output_batch) + 20 * penalty
                # loss = criterion(output_pred, output_batch) 
                
                # Penalize the region where -8 < alpha < 10 
                weighted_loss = ((1 + (10 -1) * mask) * loss).mean() 

                # backward pass and optimization
                weighted_loss.backward()
                optimizer.step()

            test_loss = criterion(model(X_test), y_test)
            scheduler.step()
            print('Epoch [{}/{}], Train Loss: {:.10e}, Test Loss: {:.10e}'.format(epoch+1, num_epochs, loss.item(), test_loss.item()))

        torch.save(model.state_dict(), f"{data_directory_path}/{type_}_model")

    return model
