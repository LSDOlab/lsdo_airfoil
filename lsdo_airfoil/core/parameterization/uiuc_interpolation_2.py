import matplotlib.pyplot as plt
import numpy as np
from lsdo_airfoil.utils.compute_b_spline_mat import get_bspline_mtx, fit_b_spline
import os
from lsdo_airfoil import UIUC_AIRFOILS
import scipy
from lsdo_airfoil.core.parameterization.test_script import find_parametric, construct_bspline_matrix


def find_nearest(array, value):
    """
    function to find the element closest to a number
    """
    idx = np.where(array == (np.abs(array - value)).argmin())[0]
    return array[idx], idx

# Plot the skewed distribution of points between 0 and 1
plot_skewed_distribution = False

# get current working directory
cwd = os.getcwd()

# Define total number of B-spline control points
# num_cpts = 60

# # Half cosine distribution
# N = num_cpts / 2
# i_vec = np.arange(0, N)
# den = N * np.pi/ np.arccos(0.5)
# x_interp_1 = 1 - np.cos(np.pi/den * i_vec)

# # Half sine distribution
# den2 = N / ((np.arcsin(1) - np.arcsin(0.5)) / np.pi)
# n_start = np.arcsin(0.5) * den2 / np.pi
# n_stop = np.arcsin(1) * den2 / np.pi
# i_vec_2 = np.arange(n_start, n_stop+1)
# x_interp_2 = np.sin(np.pi/den2 * i_vec_2)

# # Total skewed distribution of points between 0 and 1
# control_points = np.hstack((x_interp_1, x_interp_2))
# if plot_skewed_distribution:
#     plt.scatter(control_points, control_points, s=3)
#     plt.show()
# print(control_points.shape)
# # exit()

# # Define total number of B-spline control points
# num_pts = 300

# # Half cosine distribution
# N = num_pts / 2
# i_vec = np.arange(0, N)
# den = N * np.pi/ np.arccos(0.5)
# x_interp_1 = 1 - np.cos(np.pi/den * i_vec)

# # Half sine distribution
# den2 = N / ((np.arcsin(1) - np.arcsin(0.5)) / np.pi)
# n_start = np.arcsin(0.5) * den2 / np.pi
# n_stop = np.arcsin(1) * den2 / np.pi
# i_vec_2 = np.arange(n_start, n_stop+1)
# x_interp_2 = np.sin(np.pi/den2 * i_vec_2)

# # Total skewed distribution of points between 0 and 1
# x_interp = np.hstack((x_interp_1, x_interp_2))

# Logistics

# x_interp = np.linspace(0, 1, num_cpts+1)
num_cpts = 99 # Real number is + 1
x_range = np.linspace(0, 1, num_cpts)
i_vec = np.arange(0, len(x_range)+1)
# control_points = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)
control_points = np.linspace(0, 1, num_cpts+1)
x = np.linspace(0, 1, 14)
k = 1.5
control_points = x**k
# control_points = 1 / (1 + np.exp(-k * (x-0.5)))
# plt.scatter(control_points, control_points, s=4)
# print(np.flip(control_points))
# plt.show()
# exit()
# control_points[0] = 0
# control_points[-1] = 1

# num_pts = 399
# x_range = np.linspace(0, 1, num_pts)
# i_vec = np.arange(0, len(x_range)+1)
# x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)
x_interp = np.linspace(0, 1, 100)
# x_interp = 1 / (1 + np.exp(-k * (x_interp-0.5)))
# x_interp = 1 - (np.exp(-k * x_interp))
x_interp = (x_interp**k)

print(x_interp)
# exit()
u_x_dense = find_parametric(x_interp, control_points, order=4, opt_iter=5)
np.save('u_x_dense_points.npy', u_x_dense)
# u_x_dense = np.load('u_x_dense_points.npy')
B_dense = construct_bspline_matrix(len(control_points), len(x_interp), order=4, u=u_x_dense).toarray()
np.save('dense_b_spline_mat.npy', B_dense, allow_pickle=True)

# B_dense = np.load('dense_b_spline_mat.npy', allow_pickle=True)
# print(B_dense.shape)
# exit()

# exit()

# Iterate over UIUC airfoils
uiuc_airfoil_dir = os.fsencode(UIUC_AIRFOILS)

bad_airfoils = [
    # Airfoils with extreme camber or airfoils with non-normalized coordinates
    'as6099.dat',
    's4096.dat',
    'as6092.dat',
    'trainer60.dat',
    'as6095.dat',
    'ui1720.dat',
    'as6096.dat',
    'nlf414f.dat',
    'mi-vawt1.dat',
    'ah88k136.dat',
    'vr8b.dat',
    'hor04.dat',
    's1221-4deg-flap.dat',
    'naca1.dat',
    's4095.dat',
    'e664ex.dat',
    'fx77w343.dat',
    'fx77w258.dat',
    's4094.dat',
    's1221.dat',
    'esa40.dat',
    'as6094.dat',
    'ua79sfm.dat',
    
    # Open trailing edge (criteria is that deviation in y at the trailing edge is greater than 1% of the chord)
    'ah93w480b.dat',
    'e863.dat',
    'arad10.dat',
    'hs1430.dat',
    'fx79w660a.dat',
    'ah93w300.dat',
    'davissm.dat',
    'ah94w301.dat',
    'fx79w470a.dat',
    'fx77w270.dat',
    'goe711.dat',
    'ht21.dat',
    'goe257.dat',
    'raf6prop_sm.dat',
    'e864.dat',
    'arad6.dat',
    'oaf102.dat',
    'e862.dat',
    'arad20.dat',
    'arad13.dat',
    'fx77w270s.dat',
    'dsma523b.dat',
    'hs1620.dat',
    'oaf128.dat',
    'goe199.dat',
    'hs1712.dat',

    # Airfoils whose trailing edge y-coordinate is not 0 (criteria is a deviation of greater than 1% of the chord)
    'k2.dat',
    'sc20606.dat',
    'ebambino7.dat',
    'sc20710.dat',
    'ag45c03.dat',
    'clarkyh.dat',
    'sc21010.dat',
    'sc20706.dat',
    'bambino6.dat',
    'n24.dat',
    'ls421mod.dat',
    'sc20610.dat',
    'k3.dat',
    'nacam18.dat',
    'usa35b.dat',
    'sc20712.dat',
    'nacacyh.dat',
    'sc20612.dat',
    'sc20714.dat',
    'eiffel371.dat',
    'sc21006.dat',
    'glennmartin3.dat',
    'ag47c03.dat',
    'ag46c03.dat',
    'glennmartin4.dat',
    'prandtl-d-centerline.dat',
    'sc20614.dat',
    'nasasc2-0714.dat',

    # Extremely cambered and thin airfoils
    
    'as6098.dat',
    'as6093.dat',
    'as6091.dat',

    
]

counter = 0
counter_2 = 0

# fig, axs = plt.subplots(1, 2, figsize=(14, 10))
ax = plt.gca()

for file in os.listdir(uiuc_airfoil_dir):
    filename = os.fsdecode(file)
    # remove any Identifier files
    if 'Identifier' in filename:
        os.remove(UIUC_AIRFOILS/filename)
    elif filename in bad_airfoils:
        pass
    else:
        counter += 1
        airfoil_coords = np.genfromtxt(UIUC_AIRFOILS/filename, skip_header=1)


        x = airfoil_coords[:, 0]
        y = airfoil_coords[:, 1]
        
        # Find index of leading edge
        zero_indices = np.where(x == 0)[0]

        # If there are x-coordinates that are exactly zero
        if len(zero_indices) == 0: 
            min_x_idx = np.where(x == np.min(x))[0]
            if len(min_x_idx) == 2:
                x_upper = x[0:min_x_idx[1]]
                x_lower = x[min_x_idx[1]:]

                y_upper = y[0:min_x_idx[1]]
                y_lower = y[min_x_idx[1]:]

               

        
        
        elif len(zero_indices) >= 1: 
            zero_index = zero_indices[0]
            x_upper = x[0:zero_index+1]
            x_lower = x[zero_index:]

            y_upper = y[0:zero_index+1]
            y_lower = y[zero_index:]

        else:
            print(zero_indices)
            raise NotImplementedError
        
        
        x_upper = np.flip(x_upper)
        y_upper = np.flip(y_upper)

        u_x_upper = find_parametric(x_upper, control_points, order=4, opt_iter=10)
        u_x_lower = find_parametric(x_lower, control_points, order=4, opt_iter=10)
        # B = get_bspline_mtx(len(control_points), len(x_upper), order=4, u=u_x)
        c_upper, B_upper = fit_b_spline(y_upper, len(control_points), b_spline_order=4, u=u_x_upper, alpha=1e-2)
        c_lower, B_lower = fit_b_spline(y_lower, len(control_points), b_spline_order=4, u=u_x_lower, alpha=1e-2)
        print(c_upper)
        print(c_lower)

        y_upper_b_sp = B_upper @ c_upper
        y_upper_b_sp_dense = B_dense @ c_upper
        y_lower_b_sp = B_lower @ c_lower
        y_lower_b_sp_dense = B_dense @ c_lower
        # print(u_x)
        # print(B.toarray())

        print('FITTING_ERROR UPPER---', np.mean(abs(y_upper_b_sp - y_upper) / y_upper * 100))
        print('FITTING_ERROR UPPER---', np.mean(abs(y_lower_b_sp - y_lower) / y_lower * 100))

        color = next(ax._get_lines.prop_cycler)['color']
        color2 = next(ax._get_lines.prop_cycler)['color']
        plt.scatter(x_upper, y_upper, s=5 ,color=color)
        plt.plot(x_upper, y_upper_b_sp, color=color)
        plt.plot(x_interp, y_upper_b_sp_dense, color=color2)

        plt.scatter(x_lower, y_lower, s=5,color=color)
        plt.plot(x_lower, y_lower_b_sp, color=color)
        plt.plot(x_interp, y_lower_b_sp_dense, color=color2)
        plt.savefig('test_fig.png')
        if counter == 2:
            plt.show()
            exit()

        # c_upper, B_upper = fit_b_spline(y_upper, 100, 3, 1e-11)
        # print(c_upper)
        # c_lower, B_lower = fit_b_spline(y_lower, 100, 3, 1e-11)
        # print(c_lower)

        # y_upper_bsp = B_upper @ c_upper
        # y_lower_bsp = B_lower @ c_lower


        # color = next(ax._get_lines.prop_cycler)['color']
        # color2 = next(ax._get_lines.prop_cycler)['color']
        # plt.scatter(x_upper, y_upper, color=color, s=3, label=filename)
        # plt.scatter(x_lower, y_lower, color=color, s=3)
        # plt.plot(x_upper, y_upper_bsp, color=color, label='b-spline fit')
        # plt.plot(x_lower, y_lower_bsp, color=color)
        # plt.axis('equal')
        # plt.legend()
        # plt.show()
        # exit()
        # plt.pause(0.5)
        # plt.clf()

            # if zero_indices[0] == 0:
        #         if len(y) % 2 == 0:
        #             num_upper = int(len(y)/2)
        #         else:
        #             num_upper = int(round(len(y))/2 + 1)

        #         x_upper = x[0:num_upper]
        #         x_lower = x[num_upper:]

        #         y_upper = y[0:num_upper]
        #         y_lower = y[num_upper:]
        #         color = next(ax._get_lines.prop_cycler)['color']
        #         plt.plot(x_upper, y_upper, color=color, label=filename)
        #         plt.plot(x_lower, y_lower, color=color)
        #         plt.axis('equal')
        #         plt.legend()
        #         plt.pause(5)
        #         plt.clf()



        # if len(x) %2 !=0:
        #     zeros, idx = find_nearest(y, 0)
        #     if len(zeros) == 0:
        #         print(filename)
        #         print(counter)
        #         print(y[np.abs(y).argmin()])
        #         # plt.plot(x, y, label=filename)
        #         # plt.legend()
        #         # plt.axis('equal')
        #     pass
        #     # print(np.where(y==0)[0])
        #     # print(filename)
        #     # print(counter_2)
        #     # counter_2+=1

        # else:
        #     num_points = int(len(x) / 2)
        #     x_upper = np.flip(x[0:num_points+1])
        #     x_lower = x[num_points+1:]
        #     # print(x_upper)
        #     # print(x_lower)
        #     # print(filename)
        #     # print('\n')
        
        # if len(np.where(y==0)[0]) > 1:
        # # if x[0] != 1 or x[-1] !=1:
        # # if abs(y[0] - y[-1])>0.01:
        # # if abs(y[0]) > 0.01 or abs(y[-1]) > 0.01:
        #     # if len(np.where(y==0)[0]) > 1:
        #     # print(max(abs(y[0]), abs(y[-1])))
        #         # print(find_nearest(y, 0))
        #         # print(np.where(y==0)[0])
        #         print(np.where(y==0)[0])
        #         print(filename)
        #         print(counter_2)
        #         counter_2 += 1
        #         plt.plot(x, y, label=filename)
        #         plt.legend()
        #         plt.axis('equal')
        #         plt.pause(0.05)
            # plt.clf()

            # print(airfoil_coords)

plt.show()