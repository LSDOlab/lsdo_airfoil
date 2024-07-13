import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time

from lsdo_airfoil import UIUC_AIRFOILS

def remove_duplicates(x, y):
    # Combine x and y into a single array for easy handling
    combined = np.vstack((x, y)).T
    
    # Use np.unique to find unique rows based on the first column (x values)
    unique_combined, indices = np.unique(combined[:, 0], return_index=True)
    
    # Extract the unique rows
    unique_combined = combined[indices]
    
    # Split the unique_combined back into x and y
    unique_x = unique_combined[:, 0]
    unique_y = unique_combined[:, 1]
    
    return unique_x, unique_y

def process_and_interpolate_data(data, x_interp, filter_=True):
    # 1) Get rid of data where x > 1
    x = data[:, 0]
    y = data[:, 1]

    mask = (x >= 0) & (x <= 1)
    x_norm_ind = np.where(mask)[0]
    x_norm = x[x_norm_ind]
    y_norm = y[x_norm_ind]

    # 2) find the index of the minimum (absolute) x-value
    x_min_ind = np.argmin(abs(x_norm))
    x_max_inds = np.where(x_norm == np.max(x_norm))[0]
    
    # 3) Split the data into upper and lower based on where 
    #    thestagnation point is
    # If the stagnation point is on the upper curve
    if x_min_ind > x_max_inds[0]:
        x_upper_1 = np.flip(x_norm[x_max_inds[0]+1:x_min_ind+1])
        y_upper_1 = np.flip(y_norm[x_max_inds[0]+1:x_min_ind+1])
        x_upper_2 = x_norm[0:x_max_inds[0]+1]
        y_upper_2 = y_norm[0:x_max_inds[0]+1]

        x_upper = np.hstack((x_upper_1, x_upper_2))
        y_upper = np.hstack((y_upper_1, y_upper_2))

        x_lower = x_norm[x_min_ind:x_max_inds[1]+1]
        y_lower = y_norm[x_min_ind:x_max_inds[1]+1]


    # Else, the stagnation point is on the lower curve
    else:
        x_upper = x_norm[x_min_ind+1:x_max_inds[0]+1]
        y_upper = y_norm[x_min_ind+1:x_max_inds[0]+1]

        x_lower_1 = np.flip(x_norm[1:x_min_ind+1])
        y_lower_1 = np.flip(y_norm[1:x_min_ind+1])
        x_lower_2 = x_norm[x_max_inds[0]+1:x_max_inds[1]+1]
        y_lower_2 = y_norm[x_max_inds[0]+1:x_max_inds[1]+1]

        x_lower = np.hstack((x_lower_1, x_lower_2))
        y_lower = np.hstack((y_lower_1, y_lower_2))

    x_upper = (x_upper - np.min(x_upper)) / (np.max(x_upper) - np.min(x_upper))
    x_lower = (x_lower - np.min(x_lower)) / (np.max(x_lower) - np.min(x_lower))

    x_lower_unique, y_lower_unique = remove_duplicates(x_lower, y_lower)
    x_upper_unique, y_upper_unique = remove_duplicates(x_upper, y_upper)


    from scipy.interpolate import interp1d
    y_interp_upper_fun = interp1d(x_upper_unique, y_upper_unique, kind='quadratic', bounds_error=False, fill_value="extrapolate")
    y_interp_lower_fun = interp1d(x_lower_unique, y_lower_unique, kind='quadratic', bounds_error=False, fill_value="extrapolate")
    y_interp_upper = y_interp_upper_fun(x_interp)
    y_interp_lower = y_interp_lower_fun(x_interp)

    if filter_:
        from scipy.signal import savgol_filter
        y_interp_upper[0:60] = savgol_filter(y_interp_upper[0:60], 10, 2)
        y_interp_lower[0:60] = savgol_filter(y_interp_lower[0:60], 10, 2)

    return x_upper, x_lower, y_upper, y_lower, y_interp_upper, y_interp_lower

def run_xfoil(
    airfoil : str,
    aoa_range=None,
    reynolds_range=None,
    mach_range=None,
    num_interp=120,
    x_spacing="sin",
    power=2,
    plot_airfoil=False,
    force_regenerate_data=False,
    transition_top="free",
    transition_bottom="free",
    pane=250,
    save_data=True,
):
    mean_alpha = np.mean(aoa_range)
    mean_Re = np.mean(reynolds_range)
    mean_Ma = np.mean(mach_range)

    coefficients = np.zeros((1, 3))
    cp_data = np.zeros((1, 2 * num_interp))
    ue_data = np.zeros((1, 2 * num_interp))
    cf_data = np.zeros((1, 2 * num_interp))
    ds_data = np.zeros((1, 2 * num_interp))
    t_data = np.zeros((1, 2 * num_interp))
    shape_data = np.zeros((1, 2 * num_interp))
    
    inputs = np.zeros((1, 3))

    if x_spacing == "linear":
        x_interp = np.linspace(0., 1., num_interp)
    elif x_spacing == "power":
        x_interp = np.linspace(0., 1, num_interp)**power
    elif x_spacing == "sin":
        x_interp = 0.5 + 0.5*np.sin(np.pi*(np.linspace(0., 1., num_interp)-0.5))

    uiuc_airfoils = os.listdir(os.fsencode(UIUC_AIRFOILS))
    try:
        airfoil_coords = np.loadtxt(
            f'{UIUC_AIRFOILS}/{airfoil}/{airfoil}_raw.txt'
        )
    except:
        raise Exception(f"Uknown airfoil. Available airfoils are {uiuc_airfoils}")
    
    if plot_airfoil:
        plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    parent_dir = os.getcwd()
    os.chdir(f'{UIUC_AIRFOILS}/{Path(airfoil)}')
    generate_data = True
    if force_regenerate_data is False:
        if os.path.isdir('aero_data'):
            if os.path.exists(f"aero_data/aero_coefficients/coeffs_file_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy"):
                inputs =np.load(f"aero_data/inputs/inputs_file_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy")
                coeffs = np.load(f"aero_data/aero_coefficients/coeffs_file_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy")
                cp_data= np.load(f'aero_data/pressure_coefficient/cp_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')
                ue_data = np.load(f'aero_data/edge_velocity/edge_velocity_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')
                shape_data = np.load(f'aero_data/kinematic_shape_parameter/kinematic_shape_param_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')
                ds_data = np.load(f'aero_data/delta_star/delta_star_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')
                t_data = np.load(f'aero_data/theta/theta_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')
                cf_data = np.load(f'aero_data/skin_friction_coefficient/skin_friction_coeff_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}.npy')


                return coeffs, cp_data, ue_data, cf_data, ds_data, t_data, shape_data, inputs, parent_dir

    if generate_data:
        fig, axs = plt.subplots(2, 3)
        for alpha in aoa_range:
            for Re in reynolds_range:
                for Ma in mach_range:
                    if os.path.exists("polar_file.txt"):
                        os.remove("polar_file.txt")

                    if os.path.isdir('aero_data'):
                        pass
                    else:
                        os.makedirs('aero_data')
                        os.makedirs('aero_data/inputs')
                        os.makedirs('aero_data/aero_coefficients')
                        os.makedirs('aero_data/pressure_coefficient')
                        os.makedirs('aero_data/edge_velocity')
                        os.makedirs('aero_data/kinematic_shape_parameter')
                        # os.makedirs('aero_data/max_shear_coefficient')
                        os.makedirs('aero_data/delta_star')
                        os.makedirs('aero_data/theta')
                        os.makedirs('aero_data/skin_friction_coefficient')
                        # os.makedirs('aero_data/dissipation_coefficent')
                        # os.makedirs('aero_data/Re_theta')
                    

                    path = f'{UIUC_AIRFOILS}/{Path(airfoil)}/{airfoil}'
                    
                    input_file = open("input_file.in", 'w')
                    input_file.write(f"LOAD {airfoil}_raw.txt\n")
                    input_file.write("RDEF xfoil_parameters.def \n")
                    # input_file.write(airfoil_folder_name + '\n')
                    input_file.write(f"PANE {pane}\n")
                    input_file.write("PANE\n")
                    input_file.write("OPER\n")
                    input_file.write("Visc {0}\n".format(Re))
                    input_file.write("M {0}\n".format(Ma))
                    input_file.write("VPAR \n")
                    if transition_top != "free" or transition_bottom != "free":
                        input_file.write("xtr \n")
                        if transition_top != "free":
                            input_file.write(f"{transition_top} \n")
                        else:
                            input_file.write("\n")
                        if transition_bottom != "free":
                            input_file.write(f"{transition_bottom} \n")
                        else:
                            input_file.write("\n")

                    input_file.write("\n")
                    input_file.write("PACC\n")
                    input_file.write("polar_file.txt\n\n")
                    input_file.write("ITER {0}\n".format(1000))
                    
                    input_file.write("alfa {0}\n".format(alpha))
                    
                    # Distributions
                    input_file.write('cpwr\n')
                    input_file.write("cpx.txt\n")

                    input_file.write('vplo \n')
                    
                    input_file.write('h \n')
                    input_file.write('dump \n')
                    input_file.write('bl_kinematic_shape_param.txt \n')
                    
                    input_file.write('dt \n')
                    input_file.write('dump \n')
                    input_file.write('bl_delta_star_theta_top_surface.txt \n')
                
                    input_file.write('db \n')
                    input_file.write('dump \n')
                    input_file.write('bl_delta_star_theta_bottom_surface.txt \n')
                
                    input_file.write('ue \n')
                    input_file.write('dump \n')
                    input_file.write('bl_edge_velocity.txt \n')
                
                    input_file.write('cf \n')
                    input_file.write('dump \n')
                    input_file.write('bl_skin_friction_coeff.txt \n')
                    
                    input_file.write('cd \n')
                    input_file.write('dump \n')
                    input_file.write('bl_dissipation_coeff.txt \n')

                    input_file.write('ct \n')
                    input_file.write('dump \n')
                    input_file.write('bl_max_shear_coeff.txt \n')

                    input_file.write('rt \n')
                    input_file.write('dump \n')
                    input_file.write('bl_Re_theta.txt \n')

                    input_file.write("\n\n")
                    input_file.write("quit\n")

                    input_file.close()

                    start_time = time.monotonic()
                    startupinfo = None
                    
                    if os.name == 'nt':
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    proc = subprocess.Popen(
                        "xfoil < input_file.in", 
                        shell=True, 
                        startupinfo=startupinfo,  
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        # stdout=subprocess.PIPE, 
                        # stderr=subprocess.PIPE,
                    )

                    while proc.poll() is None:
                        # print('xfoil not working')
                        elapsed_time = time.monotonic() - start_time
                        if elapsed_time > 2:
                            proc.terminate()
                            break
                        time.sleep(0.01)

                    if proc.poll() is not None:
                        # Sub-process completed within the time limit
                        output, errors = proc.communicate()
                    
                        # load the data
                        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
                        
                        if os.path.exists("cpx.txt"):
                            cp_x = np.loadtxt('cpx.txt')
                            x = cp_x[:, 0]
                            y = cp_x[:, 1]

                            cp_x_upper, cp_x_lower, cp_y_upper, cp_y_lower, \
                                cp_y_interp_upper, cp_y_interp_lower = process_and_interpolate_data(cp_x, x_interp, filter_=False)


                        # Edge velocity
                        if os.path.exists("bl_edge_velocity.txt"):
                            bl_ue_x = np.loadtxt('bl_edge_velocity.txt')
                            x = bl_ue_x[:, 0]
                            y = bl_ue_x[:, 1]
                            
                            ue_x_upper, ue_x_lower, ue_y_upper, ue_y_lower, \
                                ue_y_interp_upper, ue_y_interp_lower = process_and_interpolate_data(bl_ue_x, x_interp, filter_=False)

                            # bl_edge_vs_x[i, :, :] = bl_ue_x

                        # Shape parameter
                        if os.path.exists("bl_kinematic_shape_param.txt"):
                            bl_kin_param_x = np.loadtxt('bl_kinematic_shape_param.txt')
                            # bl_kin_param_vs_x[i, :, :] = bl_kin_param_x
                            x_shape = bl_kin_param_x[:, 0]
                            y_shape = bl_kin_param_x[:, 1]

                            shape_x_upper, shape_x_lower, shape_y_upper, shape_y_lower, \
                                shape_y_interp_upper, shape_y_interp_lower = process_and_interpolate_data(bl_kin_param_x, x_interp, filter_=False)


                        # Momentum & displacement thickness top
                        if os.path.exists("bl_delta_star_theta_top_surface.txt"):
                            bl_ds_t_top_x = np.loadtxt('bl_delta_star_theta_top_surface.txt')
                            x_ds_t_top = bl_ds_t_top_x[:, 0]
                            y_ds_t_top = bl_ds_t_top_x[:, 1]

                            num_points = len(x_ds_t_top)

                            # displacement thicknes top
                            x_ds_top = x_ds_t_top[0:num_points//2]
                            y_ds_top = y_ds_t_top[0:num_points//2]

                            # momentum thickness top
                            x_t_top = x_ds_t_top[num_points//2:]
                            y_t_top = y_ds_t_top[num_points//2:]

                        # Momentum & displacement thickness bottom
                        if os.path.exists("bl_delta_star_theta_bottom_surface.txt"):
                            bl_ds_t_bottom_x = np.loadtxt('bl_delta_star_theta_bottom_surface.txt')
                            # bl_delta_theta_bottom_vs_x[i, :, :] = bl_ds_t_bottom_x
                            x_ds_t_bottom = bl_ds_t_bottom_x[:, 0]
                            y_ds_t_bottom = bl_ds_t_bottom_x[:, 1]

                            num_points = len(x_ds_t_bottom)

                            # displacement thicknes bottom
                            x_ds_bottom = x_ds_t_bottom[0:num_points//2]
                            y_ds_bottom = y_ds_t_bottom[0:num_points//2]

                            # momentum thickness bottom
                            x_t_bottom = x_ds_t_bottom[num_points//2:]
                            y_t_bottom = y_ds_t_bottom[num_points//2:]

                            x_t = np.hstack((x_t_top, x_t_bottom))
                            y_t = np.hstack((y_t_top, y_t_bottom))

                            t = np.vstack((x_t, y_t)).T

                            x_ds = np.hstack((x_ds_top, x_ds_bottom))
                            y_ds = np.hstack((y_ds_top, y_ds_bottom))

                            ds = np.vstack((x_ds, y_ds)).T

                            ds_x_upper, ds_x_lower, ds_y_upper, ds_y_lower, \
                                ds_y_interp_upper, ds_y_interp_lower = process_and_interpolate_data(ds, x_interp, filter_=False)
                            
                            t_x_upper, t_x_lower, t_y_upper, t_y_lower, \
                                t_y_interp_upper, t_y_interp_lower = process_and_interpolate_data(t, x_interp, filter_=False)

                        
                        if os.path.exists("bl_skin_friction_coeff.txt"):
                            bl_cf_x = np.loadtxt('bl_skin_friction_coeff.txt')
                            x = bl_cf_x[:, 0]
                            y = bl_cf_x[:, 1]

                            cf_x_upper, cf_x_lower, cf_y_upper, cf_y_lower, \
                                cf_y_interp_upper, cf_y_interp_lower = process_and_interpolate_data(bl_cf_x, x_interp, filter_=False)


                        # if os.path.exists("bl_dissipation_coeff.txt"):
                        #     bl_dissip_x = np.loadtxt('bl_dissipation_coeff.txt')
                        #     # bl_dissip_coeff_vs_x[i, :, :] = bl_dissip_x

                        # if os.path.exists("bl_max_shear_coeff.txt"):
                        #     bl_max_shear_x = np.loadtxt('bl_max_shear_coeff.txt')
                        #     # bl_max_shear_coeff_vs_x[i, :, :] = bl_max_shear_x

                        # if os.path.exists("bl_Re_theta.txt"):
                        #     bl_Re_theta = np.loadtxt('bl_Re_theta.txt')
                        #     # bl_Re_theta_vs_x[i, :, :] = bl_Re_theta

                        # if save_data:
                        #     np.save(f"aero_data/aero_coefficients/polar_file_aoa_{alpha}_Re_{Re}_M_{Ma}", polar_data)
                        #     np.save(f'aero_data/pressure_coefficient/cp_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', cp_x)
                        #     np.save(f'aero_data/edge_velocity/edge_velocity_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ue_x)
                        #     np.save(f'aero_data/kinematic_shape_parameter/kinematic_shape_param_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_kin_param_x)
                        #     np.save(f'aero_data/delta_star_and_theta_top/delta_star_theta_top_surface_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ds_t_top_x)
                        #     np.save(f'aero_data/delta_star_and_theta_bottom/delta_star_theta_bottom_surface_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ds_t_bottom_x)
                        #     np.save(f'aero_data/skin_friction_coefficient/skin_friction_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_cf_x)
                            # np.save(f'aero_data/dissipation_coefficent/dissipation_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_dissip_x)
                            # np.save(f'aero_data/max_shear_coefficient/max_shear_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_max_shear_x)
                            # np.save(f'aero_data//Re_theta/Reyn_theta_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_Re_theta)


                        axs[0, 0].scatter(cp_x_upper, cp_y_upper, s=4, color='k')
                        axs[0, 0].plot(x_interp, cp_y_interp_upper, color='k')
                        axs[0, 0].scatter(cp_x_lower, cp_y_lower, s=4, color='b')
                        axs[0, 0].plot(x_interp, cp_y_interp_lower, color='b')

                        axs[0, 1].scatter(ue_x_upper, ue_y_upper, s=4, color='k')
                        axs[0, 1].plot(x_interp, ue_y_interp_upper, color='k')
                        axs[0, 1].scatter(ue_x_lower, ue_y_lower, s=4, color='b')
                        axs[0, 1].plot(x_interp, ue_y_interp_lower, color='b')

                        axs[0, 2].scatter(shape_x_upper, shape_y_upper, s=4, color='k')
                        axs[0, 2].plot(x_interp, shape_y_interp_upper, color='k')
                        axs[0, 2].scatter(shape_x_lower, shape_y_lower, s=4, color='b')
                        axs[0, 2].plot(x_interp, shape_y_interp_lower, color='b')

                        axs[1, 0].scatter(ds_x_upper, ds_y_upper, s=4, color='k')
                        axs[1, 0].plot(x_interp, ds_y_interp_upper, color='k')
                        axs[1, 0].scatter(ds_x_lower, ds_y_lower, s=4, color='b')
                        axs[1, 0].plot(x_interp, ds_y_interp_lower, color='b')
                        
                        axs[1, 1].scatter(t_x_upper, t_y_upper, s=4, color='k')
                        axs[1, 1].plot(x_interp, t_y_interp_upper, color='k')
                        axs[1, 1].scatter(t_x_lower, t_y_lower, s=4, color='b')
                        axs[1, 1].plot(x_interp, t_y_interp_lower, color='b')

                        axs[1, 2].scatter(cf_x_upper, cf_y_upper, s=4, color='k')
                        axs[1, 2].plot(x_interp, cf_y_interp_upper, color='k')
                        axs[1, 2].scatter(cf_x_lower, cf_y_lower, s=4, color='b')
                        axs[1, 2].plot(x_interp, cf_y_interp_lower, color='b')
                        
                        plt.tight_layout()

                        # exit()

                        # Saving data
                        # Coefficients
                        if polar_data.size != 0:
                            print(Re, Ma, alpha)
                            polar_data_array = np.zeros((1, 3))
                            polar_data_array[0, 0] = polar_data[1]
                            polar_data_array[0, 1] = polar_data[2]
                            polar_data_array[0, 2] = polar_data[4]
                            coefficients = np.vstack((coefficients, polar_data_array))

                            # Cp
                            cp_data_array = np.zeros((1, 2 * num_interp))
                            cp_data_array[0, 0:num_interp] = cp_y_interp_upper
                            cp_data_array[0, num_interp:] = cp_y_interp_lower
                            cp_data = np.vstack((cp_data, cp_data_array))


                            # Ue
                            ue_data_array = np.zeros((1, 2 * num_interp))
                            ue_data_array[0, 0:num_interp] = ue_y_interp_upper
                            ue_data_array[0, num_interp: ] = ue_y_interp_lower
                            ue_data = np.vstack((ue_data, ue_data_array))

                            # Cf 
                            cf_data_array = np.zeros((1, 2 * num_interp))
                            cf_data_array[0, 0:num_interp] = cf_y_interp_upper
                            cf_data_array[0, num_interp: ] = cf_y_interp_lower
                            cf_data = np.vstack((cf_data, cf_data_array))

                            # ds
                            ds_data_array = np.zeros((1, 2 * num_interp))
                            ds_data_array[0, 0:num_interp] = ds_y_interp_upper
                            ds_data_array[0, num_interp: ] = ds_y_interp_lower
                            ds_data = np.vstack((ds_data, ds_data_array))

                            # t
                            t_data_array = np.zeros((1, 2 * num_interp))
                            t_data_array[0, 0:num_interp] = t_y_interp_upper
                            t_data_array[0, num_interp: ] = t_y_interp_lower
                            t_data = np.vstack((t_data, t_data_array))

                            # shape
                            shape_data_array = np.zeros((1, 2 * num_interp))
                            shape_data_array[0, 0:num_interp] = shape_y_interp_upper
                            shape_data_array[0, num_interp: ] = shape_y_interp_lower
                            shape_data = np.vstack((shape_data, shape_data_array))

                            # inputs
                            input_data_array = np.zeros((1, 3))
                            input_data_array[0, 0] = alpha 
                            input_data_array[0, 1] = Re 
                            input_data_array[0, 2] = Ma
                            inputs = np.vstack((inputs, input_data_array))
        
        if save_data:
            np.save(f"aero_data/inputs/inputs_file_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}", inputs[1:, :])
            np.save(f"aero_data/aero_coefficients/coeffs_file_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}", coefficients[1:, :])
            np.save(f'aero_data/pressure_coefficient/cp_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', cp_data[1:, :])
            np.save(f'aero_data/edge_velocity/edge_velocity_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', ue_data[1:, :])
            np.save(f'aero_data/kinematic_shape_parameter/kinematic_shape_param_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', shape_data[1:, :])
            np.save(f'aero_data/delta_star/delta_star_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', ds_data)
            np.save(f'aero_data/theta/theta_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', t_data)
            np.save(f'aero_data/skin_friction_coefficient/skin_friction_coeff_vs_x_aoa_{mean_alpha}_Re_{mean_Re}_M_{mean_Ma}', cf_data[1:, :])

        if False:
            plt.show()

        return coefficients[1:, :], cp_data[1:, :], ue_data[1:, :], cf_data[1:, :], ds_data[1:, :], t_data[1:, :], shape_data[1:, :], inputs[1:, :], parent_dir


if __name__ == "__main__":
    coefficients, cp_data, ue_data, cf_data, ds_data, t_data, shape_data, inputs, parent_dir = run_xfoil(
        airfoil="ls417",
        aoa_range=[-4], #np.linspace(-8, 10, 19),
        mach_range=[0.2],
        reynolds_range=[2e6],
    )

    print(coefficients)
