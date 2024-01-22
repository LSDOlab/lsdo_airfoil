from lsdo_airfoil import UIUC_AIRFOILS_2
import os
import numpy as np
from pathlib import Path
import subprocess
import time


uiuc_airfoil_dir = os.fsencode(UIUC_AIRFOILS_2)
parent_dir = os.getcwd()

alfa_range = np.arange(-10, 18, 0.25)
ma_range = [0, 0.3, 0.4, 0.5, 0.6]
re_range = [1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 8e6]


counter = 0
for file in os.listdir(uiuc_airfoil_dir):
    airfoil_folder_name = os.fsdecode(file)
    if airfoil_folder_name in ['nplx', 'rg14', 'goe368', 'e668']:
        pass
    else:
        print(airfoil_folder_name)
        airfoil_filename = f"{airfoil_folder_name}_raw.txt"
        os.chdir(parent_dir)
        os.chdir(f"{UIUC_AIRFOILS_2}/{Path(airfoil_folder_name)}")
        for alpha in alfa_range:
            for Re in re_range:
                for Ma in ma_range:
                    if os.path.exists("polar_file.txt"):
                        os.remove("polar_file.txt")

                    if os.path.isdir('aero_data'):
                        pass
                    else:
                        os.makedirs('aero_data')
                        os.makedirs('aero_data/aero_coefficients')
                        os.makedirs('aero_data/pressure_coefficient')
                        os.makedirs('aero_data/edge_velocity')
                        os.makedirs('aero_data/kinematic_shape_parameter')
                        os.makedirs('aero_data/max_shear_coefficient')
                        os.makedirs('aero_data/delta_star_and_theta_top')
                        os.makedirs('aero_data/delta_star_and_theta_bottom')
                        os.makedirs('aero_data/skin_friction_coefficient')
                        os.makedirs('aero_data/dissipation_coefficent')
                        os.makedirs('aero_data/Re_theta')
                    

                    path = f'{UIUC_AIRFOILS_2}/{Path(airfoil_folder_name)}/{airfoil_filename}'
                    
                    input_file = open("input_file.in", 'w')
                    input_file.write(f"LOAD {airfoil_filename}\n")
                    input_file.write(airfoil_folder_name + '\n')
                    input_file.write("PANE\n")
                    input_file.write("OPER\n")
                    input_file.write("Visc {0}\n".format(Re))
                    input_file.write("M {0}\n".format(Ma))
                    input_file.write("PACC\n")
                    input_file.write("polar_file.txt\n\n")
                    input_file.write("ITER {0}\n".format(200))
                    
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
                        "xfoil < input_file.in ", 
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
                        if elapsed_time > 0.5:
                            proc.terminate()
                            break
                        time.sleep(0.01)

                    if proc.poll() is not None:
                        # Sub-process completed within the time limit
                        output, errors = proc.communicate()
                    
                        # save the data
                        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
                        # print(polar_data)
                        np.save(f"aero_data/aero_coefficients/polar_file_aoa_{alpha}_Re_{Re}_M_{Ma}", polar_data)

                        if os.path.exists("cpx.txt"):
                            cp_x = np.loadtxt('cpx.txt')
                            print(cp_x)
                            np.save(f'aero_data/pressure_coefficient/cp_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', cp_x)

                        

                        if os.path.exists("bl_edge_velocity.txt"):
                            bl_ue_x = np.loadtxt('bl_edge_velocity.txt')
                            np.save(f'aero_data/edge_velocity/edge_velocity_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ue_x)
                            # bl_edge_vs_x[i, :, :] = bl_ue_x

                        if os.path.exists("bl_kinematic_shape_param.txt"):
                            bl_kin_param_x = np.loadtxt('bl_kinematic_shape_param.txt')
                            np.save(f'aero_data/kinematic_shape_parameter/kinematic_shape_param_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_kin_param_x)
                            # bl_kin_param_vs_x[i, :, :] = bl_kin_param_x

                        if os.path.exists("bl_delta_star_theta_top_surface.txt"):
                            bl_ds_t_top_x = np.loadtxt('bl_delta_star_theta_top_surface.txt')
                            np.save(f'aero_data/delta_star_and_theta_top/delta_star_theta_top_surface_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ds_t_top_x)
                            # bl_delta_theta_top_vs_x[i, :, :] = bl_ds_t_top_x

                        if os.path.exists("bl_delta_star_theta_bottom_surface.txt"):
                            bl_ds_t_bottom_x = np.loadtxt('bl_delta_star_theta_bottom_surface.txt')
                            np.save(f'aero_data/delta_star_and_theta_bottom/delta_star_theta_bottom_surface_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_ds_t_bottom_x)
                            # bl_delta_theta_bottom_vs_x[i, :, :] = bl_ds_t_bottom_x
                        
                        if os.path.exists("bl_skin_friction_coeff.txt"):
                            bl_cf_x = np.loadtxt('bl_skin_friction_coeff.txt')
                            np.save(f'aero_data/skin_friction_coefficient/skin_friction_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_cf_x)
                            # bl_skin_frict_vs_x[i, :, :] = bl_cf_x

                        if os.path.exists("bl_dissipation_coeff.txt"):
                            bl_dissip_x = np.loadtxt('bl_dissipation_coeff.txt')
                            np.save(f'aero_data/dissipation_coefficent/dissipation_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_dissip_x)
                            # bl_dissip_coeff_vs_x[i, :, :] = bl_dissip_x

                        if os.path.exists("bl_max_shear_coeff.txt"):
                            bl_max_shear_x = np.loadtxt('bl_max_shear_coeff.txt')
                            np.save(f'aero_data/max_shear_coefficient/max_shear_coeff_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_max_shear_x)
                            # bl_max_shear_coeff_vs_x[i, :, :] = bl_max_shear_x

                        if os.path.exists("bl_Re_theta.txt"):
                            bl_Re_theta = np.loadtxt('bl_Re_theta.txt')
                            np.save(f'aero_data//Re_theta/Reyn_theta_vs_x_aoa_{alpha}_Re_{Re}_M_{Ma}', bl_Re_theta)
                            # bl_Re_theta_vs_x[i, :, :] = bl_Re_theta


    # counter += 1
    # if counter == 1:
    #     exit()
        # np.save(f'aero_data/polar_data_aoa_{alpha_range[i]}_Re_{Re[j]}_M_{Ma}', polar_data)
