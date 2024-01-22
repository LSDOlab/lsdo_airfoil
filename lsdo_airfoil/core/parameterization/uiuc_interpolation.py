import matplotlib.pyplot as plt
import numpy as np
from lsdo_airfoil.utils.compute_b_spline_mat import get_bspline_mtx, fit_b_spline
import os
from lsdo_airfoil import UIUC_AIRFOILS, UIUC_AIRFOILS_2
import scipy
from lsdo_airfoil.core.parameterization.test_script import find_parametric, construct_bspline_matrix
import pickle
from smt.surrogate_models import RMTB
from scipy.interpolate import splev, splrep


# Plot the skewed distribution of points between 0 and 1
plot_skewed_distribution = False

def find_nearest(array, value):
    """
    function to find the element closest to a number
    """
    idx = np.where(array == (np.abs(array - value)).argmin())[0]
    return array[idx], idx

def make_skewed_normalized_distribution(num_points, half_cos=False, power=False):
    if half_cos:
        # Half cosine distribution
        N = num_points 
        i_vec = np.arange(0, N)
        # print(2 * len(i_vec))
        # print(i_vec)
        half_cos = 1 - np.cos(np.pi /(2 * len(i_vec) -2) * i_vec)
        # Total skewed distribution of points between 0 and 1
        return half_cos
    elif power:
        return np.linspace(0, 1, num_points)**power
    else:
        # Half cosine distribution
        N = num_points / 2
        i_vec = np.arange(0, N-1)
        den = (N-1) * np.pi/ np.arccos(0.5)
        half_cos = 1 - np.cos(np.pi/den * i_vec)

        # Half sine distribution
        den2 = N / ((np.arcsin(1) - np.arcsin(0.5)) / np.pi)
        n_start = np.arcsin(0.5) * den2 / np.pi
        n_stop = np.arcsin(1) * den2 / np.pi
        i_vec_2 = np.arange(n_start, n_stop+1)
        half_sin = np.sin(np.pi/den2 * i_vec_2)

        # Total skewed distribution of points between 0 and 1
        return np.hstack((half_cos, half_sin))

def find_duplicate_indices(vector):
    # Create an empty dictionary to store elements and their indices
    element_indices = {}
    duplicates = []

    for index, element in enumerate(vector):
        # If the element is already in the dictionary, it's a duplicate
        if element in element_indices:
            duplicates.append(index)
            duplicates.append(element_indices[element])
        else:
            # Store the index of the first occurrence of the element
            element_indices[element] = index

    return duplicates


# get current working directory
cwd = os.getcwd()

num_pts = 150
num_ctr_pts = 80
x_interp =    make_skewed_normalized_distribution(num_pts, half_cos=True, power=False) #  np.linspace(0, 1, num_pts)**1.2 #  
control_points = make_skewed_normalized_distribution(num_ctr_pts, half_cos=True, power=False) #    np.linspace(0, 1, num_ctr_pts) # 
ux_dense = None# find_parametric(x_interp, control_points, 4, 5)

# B_dense = get_bspline_mtx(num_ctr_pts, num_pts, 4 , u=ux_dense)

if plot_skewed_distribution:
    plt.scatter(x_interp, x_interp, s=4)
    plt.scatter(control_points, control_points, s=4)
    plt.show()

# Iterate over UIUC airfoils
uiuc_airfoil_dir = os.fsencode(UIUC_AIRFOILS)

bad_airfoils = [
    'as6099.dat',                 # check
    's4096.dat',                  # check
    'as6092.dat',                 # check
    'trainer60.dat',              # check
    'as6095.dat',                 # check
    'as6096.dat',                 # check
    'ah88k136.dat',               # check
    'vr8b.dat',                   # check
    's1221-4deg-flap.dat',        # check
    'naca1.dat',                  # check
    
    'e664ex.dat',                 # check
    's1221.dat',                  # check
    'as6094.dat',                 # check
    'ua79sfm.dat',                # check
    'ah93w480b.dat',              # check
    'hs1430.dat',                 # check
    'fx79w660a.dat',              # check
    'fx79w470a.dat',              # check
    'raf6prop_sm.dat',            # check
    'dsma523b.dat',               # check

    'goe199.dat',                 # check
    'ebambino7.dat',              # check
    'bambino6.dat',               # check
    'eiffel371.dat',              # check
    'as6098.dat',                 # check
    'as6093.dat',                 # check
    'as6091.dat',                 # check
    'e377m.dat',                  # check
    'goe462.dat',                 # check
    'ste87151.dat',               # check

    'tsagi_r3a.dat',              # check
    'm8.dat',                     # check
    'eiffel430.dat',              # check
    'eiffel10.dat',               # check
    'n6h20.dat',                  # check
    'goe440.dat',                 # check
    'usa31.dat',                  # check
    'goe802b.dat',                # check
    'hh02.dat',                   # check
    'saratov.dat',                # check
   
    'e376.dat',                   # check
    'e377.dat',                   # check
    'e378.dat',                   # check
    'goe388.dat',                 # check
    'naca63a210.dat',             # check
    'stcyr171.dat',               # check
    'naca63206.dat',              # check
    'goe802a.dat',                # check
    'goe451.dat',                 # check
    'nacam12.dat',                # check
    
    'goe229.dat',                 # check
    'e49.dat',                    # check
    'naca23024.dat',              # check
    'goe559.dat',                 # check
    'goe13k.dat',                 # check
    'm9.dat',                     # check
    'as6097.dat',                 # check
    'strand.dat',                 # check
    'l1003.dat',                  # check
    'goe481.dat',                 # check
    
    'dga1182.dat',                # check
    'griffith30SymSuction.dat',   # check
    'dbln526.dat',                # check
    'e379.dat',                   # check
    'du861372.dat',               # check
    'isa961.dat',                 # check
    'ag38.data',                  # check
    'fg1.dat',                    # check
    'm7.dat',                     # check
    'eiffel428.dat',              # check
    
    'ma409.dat',                  # check
    'ah81k144.dat',               # check
    'stf86361.dat',               # check
    'rc0864c.dat',                # check
    'isa960.dat',                 # check
    'n9.dat',                     # check
    'rc1064c.dat', 
    'rc1064.dat',
    'e863.dat' 
    'ah81k144wfKlappe.dat',
    'kenmar.dat',
    'ht05.dat',
    'ea61009.dat',
    'vr9.dat',
    'fx63145.dat',
    'fx38153.dat',
    'vr8.dat',
    'b707b.dat',

    

]

airfoils_that_need_high_smoothness = [
'n63210.dat', # overlapping trailing edge --> increase s in splrep
'hor20.dat', # overlapping trailing edge --> increase s in splrep
'goe780.dat', # overlapping trailing edge --> increase s in splrep
'goe301.dat', # overlapping trailing edge --> increase s in splrep
'rg1496.dat',  # inf error --> don't include in interpolation error statistic
'goe15k.dat', # runge oscillations --> increase s in splrep

'npl9627.dat', # fit blows up
'fx63158.dat', # fit blows up
'sc20403.dat', # fit blows up
'sc20610.dat', # fit blows up

'naca2424.dat', # some runge oscillations --> increase s in splrep
'naca4424.dat', # some runge oscillations --> increase s in splrep
'naca643218.dat', # some runge oscillations --> increase s in splrep
'fx711530.dat', # some runge oscillations --> increase s in splrep
'fx711525.dat', # some runge oscillations --> increase s in splrep
'fx63145.dat', # some runge oscillations --> increase s in splrep
'vr1.dat', # some runge oscillations --> increase s in splrep
'usa35a.dat', # some runge oscillations --> increase s in splrep
'goe284.dat', # some runge oscillations --> increase s in splrep
'pt40.dat', # some runge oscillations --> increase s in splrep
'hq2090sm.dat', # some runge oscillations --> increase s in splrep
'e1214.dat', # some runge oscillations --> increase s in splrep
'goe438.dat', # some runge oscillations --> increase s in splrep
'n63412.dat', # some runge oscillations --> increase s in splrep
'e1212mod.dat', # some runge oscillations --> increase s in splrep
'fx38153.dat', # some runge oscillations --> increase s in splrep
'fx63137.dat', # some runge oscillations --> increase s in splrep
'fx082512.dat', # some runge oscillations --> increase s in splrep
'naca661212.dat', # some runge oscillations --> increase s in splrep
'rhodesg30.dat', # some runge oscillations --> increase s in splrep
'du84132v.dat', # some runge oscillations --> increase s in splrep
'n64212ma.dat', # some runge oscillations --> increase s in splrep
'goe711.dat', # some runge oscillations --> increase s in splrep
'npl9660.dat', # some runge oscillations --> increase s in splrep
'hq1510.dat', # some runge oscillations --> increase s in splrep
'hq159b.dat', # some runge oscillations --> increase s in splrep
'naca66209.dat', # some runge oscillations --> increase s in splrep
'gu255118.dat', # some runge oscillations --> increase s in splrep
'naca66206.dat', # some runge oscillations --> increase s in splrep
'e654.dat', 
'p51hroot.dat',
'mh64.dat',
's4061.dat',
's4110.dat',
]

airfoil_list_2 = [
    'dae31.dat',
    'e224.dat',
    'mh64.dat',
    'e334.dat',
    'e479.dat',
    's4053.dat',
    'cast102.dat',
    'n63210.dat',
    'e654.dat',
    's8064.dat',
    'p51hroot.dat',
    'mh62.dat',
    'ah79100a.dat',
    'goe681.dat',
    'e656.dat',
    'e178.dat',
    'fx6617a2.dat',
    'eh1590.dat',
    'prandtl-d-wingtip.dat',
    'e863.dat',
    'rae2822.dat',
    'e431.dat',
    'lds2.dat',
    'sd7084.dat',
    'fx61184.dat',
    'e332.dat',
    'sm701.dat',
    'fx66h80.dat',
    's4110.dat',
    'sd6060.dat',
    'marske7.dat',
    'naca2424.dat',
    'mh20.dat',
    'fx711530.dat',
    'mh24.dat',
    's2027.dat',
    'vr1.dat',
    'fx63158.dat',
    'mh113.dat',
    'ma409sm.dat',
    'sc1095.dat',
    'naca4424.dat',

]

airfoils_smoothing = {
    'e334.dat': (1e-9, 0),
    'p51hroot.dat': (1e-7, 0, 3),
    'e479.dat' : (1e-9, 1e-9),
    'e61.dat' : (1e-12, 1e-9),
    'e476.dat' : (1e-9, 1e-9),
    'cast102.dat' : (1e-9, 0),
    'marske1.dat' : (1e-10, 1e-8),
    'mh62.dat' : (1e-12, 1e-9),
    'goe681.dat' : (1e-9, 1e-11),
    'goe244.dat' : (0, 1e-9),
    'goe144.dat' : (0, 1e-10),
    'ah79100a.dat' : (1e-8, 1e-8, 2),
    'mh201.dat' : (1e-11, 1e-11),
    'dae31.dat' : (1e-8, 1e-8), 
    'e224.dat' : (1e-9, 0),
    'mh64.dat' : (1e-13, 1e-9),
    'e656.dat' : (1e-8, 1e-13),
    's4053.dat' : (1e-7, 1e-8), 
    'e654.dat' : (1e-7, 1e-13),
    'e178.dat' : (1e-12, 1e-11),
    's8064.dat' : (1e-7, 0),
    'fx6617a2.dat' : (1e-6, 1e-7),
    'eh1590.dat' : (1e-9, 1e-9),
    'prandtl-d-wingtip.dat' : (1e-8, 1e-8),
    'e683.dat' : (1e-10, 1e-10),
    'rae2822.dat' : (1e-9, 1e-9),
    'e431.dat' : (1e-9, 0),
    'lds2.dat' : (1e-9, 0),
    'sd7084.dat' : (1e-9, 1e-8),
    'fx61184.dat' : (1e-8, 0),
    'e332.dat' : (1e-7, 0),
    'sm701.dat' : (1e-9, 1e-9),
    'naca642415.dat' : (1e-9, 0),
    'goe423.dat' : (1e-9, 0),
    'sc20412.dat' : (1e-11, 1e-10),
    'ag18.dat' : (1e-11, 1e-10),
    'mh120.dat' : (0, 1e-8),
    'fx63100.dat' : (1e-8, 0),
    'vr15.dat' : (1e-9, 1e-8),
    'fx66h80.dat' : (1e-9, 1e-9),
    's8052.dat' : (1e-11, 1e-7, 2),
    'arad10.dat' : (1e-9, 1e-7, 2),
    's4061.dat' : (0, 1e-8),
    'e339.dat' : (1e-9, 0),
    'ah93k130.dat' : (1e-8, 1e-10),
    's4110.dat' : (1e-7, 1e-7),
    'sd6060.dat' : (1e-8, 0),
    'e471.dat' : (0, 1e-9),
    'ah63k127.dat' : (1e-10, 5e-7),
    'us1000root.dat' : (0, 1e-7, 2),
    'marske7.dat' : (1e-8, 1e-8),
    'naca2424.dat' : (5e-7, 0, 2),
    'mh20.dat': (0., 1e-8),
    'ah83159.dat' : (1e-10, 1e-8),
    'mue139.dat' : (1e-10, 1e-8),
    'rc510.dat' : (1e-8, 1e-8),
    'fx711530.dat' : (1e-6, 1e-6, 2),
    'mh24.dat' : (0, 1e-8),
    's2027.dat' : (1e-8, 1e-8),
    'e71.dat' : (0., 1e-10),
    'vr1.dat' : (1e-7, 1e-7, 2),
    'fx63158.dat' : (5e-6, 5e-6, 2),
    'mh113.dat' : (1e-7, 1e-8),
    'ma409sm.dat' : (1e-8, 1e-7),
    'eh2070.dat' : (1e-8, 1e-8, 2),
    'sc1095.dat' : (1e-9, 1e-9),
    'naca4424.dat' : (1e-6, 0., 2),
    'la203a.dat' : (0., 1e-9),
    'fx63137sm.dat' : (1e-11, 1e-8, 2),   
    'hor20dat': (0., 1e-10),
    'ah79100c.dat' : (1e-8, 1e-8, 2),
    's1016.dat' : (1e-9, 1e-9),
    's5020.dat' : (1e-8, 1e-8, 2),
    'mh44.dat' : (1e-8, 0.),
    'goe795.dt' : (1e-8, 0.),
    'sg6040.dat' : (1e-8, 1e-11),
    'rc10n1.dat' : (1e-7, 0),
    'goe235.dat' : (1e-9, 0, 2),
    'geo795.dat' : (1e-9, 1e-11, 2),
    'fx711525.dat' : (1e-7, 1e-7, 2),
    's3010.dat' : (0., 1e-7),
    'fx74130wp1.dat' : (1e-9, 1e-8),
    'dae51.dat' : (1e-9, 1e-8),
    'fx61163.dat' : (1e-9, 1e-11, 2),
    'e393.dat' : (1e-9, 1e-8),
    'fx66196v.dat' : (1e-8, 1e-10, 2),
    'naca747a415.dat' : (1e-9, 0, 2),
    'c141b.dat' : (1e-8, 1e-8, 2),
    'mh38.dat' : (1e-8, 0.),
    'sc1012r8.dat' : (0, 1e-8),
    'fx74cl6140.dat' : (1e-10, 1e-9, 2),
    'pt40.dat' : (1e-8, 1e-8, 2),
    'sd7037.dat' : (0., 1e-8),
    'rg12a.dat' : (1e-8, 0),
    'ah79100b.dat' : (1e-7, 1e-8, 2),
    'e598.dat' : (1e-8, 0, 2),
    'fx63147.dat' : (1e-6, 1e-6, 2),
    'mh102.dat' : (1e-8, 0),
    'e584.dat' : (1e-9, 0, 2),
    'ah81131.dat' : (1e-11, 1e-9),
    'sa7024.dat' : (0, 1e-9, 2), 
    'c141e.dat' : (1e-8, 1e-8, 2),
    'as5045.dat' : (0., 1e-8, 2),
    'bacnlf.dat' : (1e-11, 1e-8, 2),
    'ste87391.dat' : (1e-9, 1e-8, 2),
    's7012.dat' : (0., 1e-7),
    'davissm.dat' : (1e-8, 1e-8, 2),
    'ag455ct02r.dat' : (1e-9, 1e-11, 2),
    'e214.dat' : (1e-10, 1e-10, 2),
    'ht23.dat' : (1e-9, 1e-10, 2),
    'e211.dat' : (0., 1e-8, 3),
    'uag8814320.dat' : (1e-8, 0, 2),
    'fx05h126.dat' : (1e-9, 1e-9, 2),
    'sd8000.dat' : (1e-8, 0, 2),
    'ag45c03.dat' : (1e-8, 0, 2),
    'e420.dat' : (1e-8, 0, 2),
    'e1212mod.dat' : (0, 1e-8, 2),
    'e857.dat' : (1e-8, 0),
    'rg14a147.dat' : (1e-9, 0, 2),
    'mh126.dat' : (1e-8, 1e-9, 2),
    'nlf0115.dat' : (1e-9, 1e-8, 2),
    'e193.dat' : (0, 1e-8, 2),
    'fx75141.dat' : (1e-9, 1e-9, 2),
    'fx84w097.dat' : (1e-9, 1e-9, 2),
    'du8608418.dat' : (1e-8, 1e-11, 2),
    'rg8.dat' : (1e-11, 1e-8, 2),
    'e559.dat' : (1e-9, 1e-10, 2),
    'fx63137.dat' : (10e-7, 1e-6, 2),
    'naca643618.dat' : (1e-8, 0, 2),
    'e561.dat' : (0., 1e-8, 2),
    'mh82.dat' : (1e-8, 0),
    'nlf0215f.dat' : (1e-10, 1e-8, 2),
    'mh115.dat' : (1e-9, 0),
    'mh114.dat' : (1e-9, 0),
    'e850.dat' : (1e-8, 1e-10, 2),
    'mh104.dat' : (1e-8, 0),
    'du84132v.dat' : (1e-6, 1e-6, 3),
    'fx67k170.dat' : (1e-8, 0, 2),
    'ah80136.dat' : (1e-9, 1e-9, 2),
    'goe290.dat' : (1e-7, 0, 3),
    'hq2590sm.dat' : (1e-8, 0, 2),
    'mh122.dat' : (0, 1e-9),
    'ui1720.dat' : (1e-11, 1e-8, 2),
    'ys930.dat' : (1e-9, 0),
    'fx77w270.dat' : (2e-6, 1e-5, 3),
    'jx-gs-10.dat' : (1e-8, 1e-8),
    'e221.dat' : (1e-8, 0, 2),
    'fx61147.dat' : (1e-8, 0, 2),
    's8066.dat' : (1e-8, 0),
    'e603.dat' : (1e-8, 0),
    'naca634421.dat' : (1e-8, 0),
    'e475.dat' : (1e-9, 1e-9, 2),
    'naca23018.dat' : (1e-6, 1e-8, 3),
    'ah82150f.dat' : (1e-8, 1e-8, 2),
    'geminism.dat' : (1e-7, 1e-7, 2),
    'ag09.dat' : (0., 1e-8),
    'e715.dat' : (1e-9, 1e-9),
    'mh110.dat' : (0, 1e-8),
    'nlf1015.dat' : (0, 1e-7, 2),
    'naca654421.dat' : (1e-8, 0),
    'tempest2.dat' : (1e-8, 0),
    'goe207.dat' : (1e-7,0, 2),
    'n64212ma.dat' : (1e-6, 1e-7, 3),
    'mh30.dat' : (0, 1e-8),
    'naca632a015.dat' : (1e-7, 1e-7, 2),
    'n2414.dat' : (1e-7, 1e-10, 2),
    'mh27.dat' : (0, 1e-9),
    'fx60100sm.dat' : (1e-8, 1e-8, 2),
    'goe434.dat' : (1e-7, 0, 2),
    'rg15a213.dat' : (1e-7, 1e-10, 2),
    'ag44ct02r.dat' : (1e-7, 1e-11),
    'e853.dat' : (0., 1e-8),
    'rg14.dat' : (1e-8, 1e-7, 2),
    'ag17.dat' : (1e-9, 1e-9, 3),
    'e174.dat' : (0., 1e-7, 3),
    'ag10.dat' : (0, 1e-9, 3),
    'nlf414f.dat' : (1e-7, 1e-7),
    'e851.dat' : (1e-8, 0, 2),
    'mh121.dat' : (1e-8, 0), 
    'hq159b.dat' : (1e-8, 0, 2), 
    'fx66h60.dat' : (1e-8, 1e-8, 2), 
    'kc135winglet.dat': (1e-8, 1e-7, 2),
    'jx-gs-04.dat': (0, 1e-9),
    'waspsm.dat': (1e-8, 1e-8, 2),
    'ah88k130.dat': (1e-6, 1e-7, 2),
    'ag13.dat' : (1e-9, 1e-9, 3),
    'naca663418.dat' : (1e-8, 0, 2),
    's5010.dat' : (1e-7, 1e-10, 2),
    'gu255118.dat' : (1e-8, 1e-7, 2),
    'sc20414.dat' : (1e-8, 1e-8, 2),
    'dae21.dat' : (1e-8, 1e-9, 3),
    'e207.dat' : (1e-8, 1e-8, 2),
    'b707d.dat' : (1e-7, 1e-11, 3),
    'fx78k150.dat' : (1e-8, 1e-8, 2),
    'e560.dat' : (1e-10, 1e-8, 2),
    'naca633618.dat' : (1e-8, 0, 2),
    'fx84w127.dat' : (1e-8, 1e-8, 2),
    'e337.dat' : (1e-8, 0, 2),
    'defcnd2.dat' : (0, 1e-8, 2),
    'fx78pk188.dat' : (1e-8, 1e-8, 2),
    'e793.dat' : (1e-8, 1e-9, 2),
    'goe383.dat' : (1e-8, 0, 2),
    'fx84w140.dat' : (1e-8, 1e-8, 2),
    'fx67k150.dat' : (1e-8, 1e-8, 2),
    'eh2510.dat' : (1e-6, 0, 2),
    'mh45.dat' : (1e-8, 0),
    'e593.dat' : (1e-8, 0, 2),
    'e545.dat' : (1e-8, 0, 2),
    'fx76mp160.dat' : (1e-5, 1e-5, 2),
    'naca16009.dat' : (1e-7, 1e-7),
    'as5048.dat' : (0., 1e-8),
    'nlf416.dat' : (0., 1e-8),
    'ah94156.dat' : ( 1e-8, 0),
    'vr14.dat' : ( 1e-8, 1e-9),
    'ah93k132.dat' : (1e-9, 1e-7),
    'fxs02196.dat' : (1e-5, 1e-5, 2),
    'fx66182.dat' : (1e-7, 1e-7, 2),
    'p51droot.dat' : (1e-8, 1e-8, 2),
    'ag04.dat' : (0., 1e-8),
    'e260.dat' : (1e-7, 0),
    'august160.dat' : (1e-7, 0, 2),
    'naca747a315.dat' : (1e-8, 0, 2),
    'e197.dat' : (1e-9, 1e-7, 2),
    'e186.dat' : (1e-9, 1e-7, 2),
    'fxs03182.dat' : (1e-5, 1e-6, 2),
    'miley.dat' : (0, 1e-8),
    'goe505.dat' : (1e-8, 0),
    'goe184.dat' : (1e-8, 0, 2),
    'goe435.dat' : (1e-8, 0),
    'ah93w145.dat' : (1e-9, 1e-9),
    'sd7062.dat' : (1e-8, 1e-9),
    'fx66s196.dat' : (5e-7, 0, 2),
    'e662.dat' : (1e-8, 1e-8, 2),
    'ag03.dat' : (1e-10, 1e-8), 
    'fx05188.dat' : (1e-6, 1e-8, 2),
    's8038.dat' : (1e-8, 1e-8, 2),
    'e1210.dat' : (1e-8, 1e-7, 2),
    'fx75193.dat' : (1e-6, 1e-6, 2),
    'e360.dat' : (1e-8, 0.),
    'lwk79100.dat' : (1e-7, 1e-8, 2),
    'fx73170a.dat' : (1e-8, 1e-6, 2),
    'e184.dat' : (1e-8, 1e-7, 2),
    'eh2010.dat' : (1e-8, 1e-7, 2),
    'fx74130wp2mod.dat' : (1e-7, 1e-7, 2),
    'e222.dat' : (1e-8, 1e-7, 2),
    'df101.dat' : (1e-8, 1e-7),
    'davis_corrected.dat' : (1e-8, 0.),
    'e342.dat' : (1e-7, 0, 2), 
    'rae69ck.dat' : (1e-8, 1e-8, 2),
    'ag47ct02r.dat' : (1e-8, 1e-9, 2),
    's1091.dat' : (0., 1e-7),
    'fx76mp140.dat' : (1e-7, 1e-7, 2),
    'fx84w218.dat' : (1e-7, 1e-7, 2),
    'fx62k131.dat' : (1e-8, 0, 2),
    'e168.dat' : (1e-8, 1e-8, 2),
    'hq259b.dat' : (1e-8, 0, 2),
    'hq3514.dat' : (1e-8, 0, 2),
    'fx79l100.dat' : (1e-8, 1e-8, 2),
    'psu-90-125wl.dat' : (1e-9, 1e-8, 2),
    's6063.dat' : (1e-8, 0),
    'e1213.dat' : (1e-8, 1e-9),
    'be50sm.dat' : (1e-8, 1e-7, 2),
    'ls413mod.dat' : (1e-8, 0, 2),
    'mh60.dat' : (1e-9, 0, 2),
    'sa7036.dat' : (1e-8, 0, 2),
    's8055.dat' : (0., 1e-8, 2),
    'ah21-9.dat' : (1e-8, 1e-8, 2),
    'v43015.dat' : (1e-8, 0, 2),
    'sp4721bs.dat' : (0, 1e-10),
    'avistar.dat' : (1e-8, 1e-8, 2),
    'e361.dat' : (1e-8, 0),
    'e1230.dat' : (1e-10, 1e-8),
    'sd7090.dat' : (1e-8, 1e-10, 2),
    'e604.dat' : (1e-8, 0),
    'e216.dat' : (0., 1e-8),
    'naca644421.dat' : (1e-8, 0, 2), 
    'mh49.dat' : (1e-8, 1e-9, 2), 
    'fx73170.dat' : (1e-8, 1e-7),
    'e212.dat' : (0, 1e-8),
    'sa7026.dat' : (0, 1e-8, 2),
    'fx60126.dat' : (1e-7, 1e-7, 2),
    'ea61012.dat' : (1e-6, 1e-6, 2), 
    'mh116.dat' : (1e-9, 0),
    'naca632615.dat' : (1e-7, 0, 2),
    's1020.dat' : (0, 1e-8,  2),
    'fx83w227.dat' : (1e-7, 1e-8, 2),
    'e220.dat' : (1e-8, 0, 2),
    'fx71l150.dat' : (3e-6, 3e-6, 2),
    'marske5.dat' : (1e-8, 0, 2),
    'rg12a189.dat' : (1e-8, 0, 2),
    'fx84w150.dat' : (1e-8, 1e-8),
    's2046.dat' : (1e-7, 1e-8),
    'mh80.dat' : (1e-8, 0., 2),
    'dh4009sm.dat' : (1e-7, 1e-7, 2),
    'rc08b3.dat' : (1e-8, 1e-8, 2),
    'sc20612.dat' : (1e-9, 1e-7, 2),
    'fx69pr281.dat' : (1e-9, 1e-9),
    'e817.dat' :  (1e-9, 1e-9),
    'sa7025.dat' :  (0, 1e-8, 2),
    'ah79k143.dat' :  (1e-8, 1e-8, 2),
    'e664.dat' :  (0, 1e-8),
    'e182.dat' : (0., 1e-8),
    'fx61140.dat' : (1e-8, 0, 2),
    'e657.dat' : (1e-9, 0),
    'e558.dat' : (1e-8, 0),
    'fx74130wp2.dat' : (1e-9, 1e-8, 2),
    'e864.dat' : (1e-9, 1e-9, 2),
    'goe701.dat' : (1e-8, 0, 2),
    'ultimate.dat' : (1e-7, 1e-7, 2),
    'mh26.dat' : (1e-8, 1e-8),
    's4062.dat' : (1e-8, 0.),
    'naca6412.dat' : (1e-8, 1e-7),
    's4022.dat' : (1e-8, 1e-9, 2),
    'goe770.dat' : (1e-8, 0.),
    'goe646.dat' : (1e-8, 0., 2),
    's4233.dat' : (0., 1e-8), 
    's8025.dat' : (1e-7, 0), 
    'n0012.dat' : (1e-9, 1e-9), 
    'fauvel.dat' : (1e-7, 1e-7, 2), 
    'e201.dat' : (1e-11, 1e-8, 2), 
    'vr5.dat' : (1e-8, 0., 2), 
    'e343.dat' : (1e-8, 0.),
    'sg6050.dat' : (1e-8, 0.),
    'fxl142k.dat' : (1e-8, 1e-8),
    'fx78k140.dat' : (1e-7, 1e-8, 2),
    'ht12.dat' : (1e-8, 1e-8, 2),
    'cr001sm.dat' : (1e-8, 1e-8, 2),
    'ls413.dat' : (1e-8, 1e-8, 2),
    'mh23.dat' : (0., 1e-9),
    'e422.dat' : (1e-8, 0., 2),
    'sc20712.dat' : (1e-9, 1e-9, 2),
    'e206.dat' : (1e-8, 1e-7, 2),
    'e1212.dat' : (1e-7, 1e-8, 2),
    'rg12.dat' : (1e-8, 1e-7, 2),
    'fx62k153.dat' : (1e-8, 0, 2),
    'mh61.dat' : (0.,1e-8),
    'as5046.dat' : (1e-8, 1e-8, 2),
    'e266.dat': ( 0., 1e-8, 2),
    'fxm2.dat' : (4e-6, 1e-7, 3),
    'goe234.dat' : (1e-8, 0., 2),
    'e838.dat' : (1e-8, 1e-8, 2), 
    's2055.dat' : (1e-8, 1e-9, 2),
    'sd5060.dat' : (1e-8, 1e-8, 2),
    'e327.dat' : (1e-8, 0, 2),
    'e554.dat' : (1e-9, 0),
    'sg6041.dat' : (1e-10, 1e-7),
    'sc20714.dat' : (1e-10, 1e-9),
    'e63.dat' : (1e-7, 0),
    'mh43.dat' : (0., 1e-9),
    'fx60157.dat' : (1e-6, 1e-6, 2),
    'fx75vg166.dat' : (1e-8, 1e-8, 2),
    'arad6.dat' : (1e-9, 1e-9),
    'e1211.dat' : (1e-10, 1e-8),
    'mh46.dat' : (0, 1e-8),
    'fx711520.dat' : (1e-6, 1e-6, 2),
    'e1233.dat' : (1e-8, 1e-8, 2),
    'sc20406.dat' : (1e-7, 1e-7),
    'e542.dat' : (1e-8, 0), 
    'sc1095r8.dat' : (1e-9, 1e-8), 
    'e325.dat' : (1e-8, 0), 
    'oaf102.dat' : (1e-7, 1e-8, 2), 
    'c141c.dat' : (1e-8, 1e-8), 
    's1210.dat' : (0, 1e-8),
    'ag14.dat' : (0, 1e-7, 2),
    'mh150.dat' : (0, 1e-8),
    'fx72150b.dat' : (1e-7, 5e-7, 2),
    'ec863914.dat' : (1e-7, 1e-8, 2),
    's1012.dat' : (1e-7, 1e-7),
    's4180.dat' : (1e-7, 1e-8),
    'goe650.dat' : (1e-8, 0),
    's102s.dat' : (1e-7, 1e-8, 2),
    'fx83w108.dat' : (1e-7, 1e-7, 2),
    'rc12n1.dat' : (1e-8, 1e-8),
    'naca4418.dat' : (5e-7, 1e-9, 2),
    'ah80140.dat' : (1e-8, 1e-8), 
    'sd8040.dat' : (1e-8, 1e-7, 2),
    'bw3.dat' : (1e-7, 1e-7, 2),
    'b707a.dat' : (1e-6, 1e-6, 3),
    'goe122.dat' : (1e-8, 1e-8),
    'e874.dat' : (0., 1e-8),
    'e68.dat' : (1e-7, 1e-8, 2),
    'a18sm.dat' : (1e-8, 1e-8, 2),
    'goe410.dat' : (5e-8, 5e-8, 2),
    'naca2421.dat' : (1e-8, 1e-9, 2),
    'b707c.dat' : (1e-5, 1e-8, 2),
    'e432.dat'  : (1e-7, 0),
    'rc1264c.dat' : (1e-8, 1e-15, 2),
    'goe765.dat' : (1e-8, 0, 2), 
    'e854.dat'  : (0, 1e-8),
    'fx77080.dat' : (1e-7, 1e-7, 2), 
    'sg6051.dat' : (5e-8, 0),
    'sa7035.dat' : (5e-8, 0),
    'naca4415.dat' : (1e-8, 1e-8, 2),
    'e862.dat' : (1e-8, 1e-8, 2),
    'fx72150a.dat' : (1e-7, 1e-8, 2),
    'l188root.dat' : (1e-8, 1e-8),
    'ah93156.dat' : (1e-8, 0),
    'pmc19sm.dat' : (1e-8, 1e-8, 2),
    'fx63120.dat' : (1e-8, 1e-8, 2),
    'sd2083.dat' : (1e-10, 1e-8, 2),
    'mh22.dat' : (1e-8, 0.),
    'hq195.dat' : (1e-8, 0, 2),
    'rc410.dat' : (1e-8, 1e-8, 2),
    'sc21006.dat' :(1e-8, 1e-9, 3),
    'fx78k140a20.dat' : (1e-8, 0, 2),
    's7075.dat' : (1e-8, 1e-8, 2),
    'sg6042.dat' : (1e-9, 1e-7, 2),
    'lrn1007.dat' : (1e-8, 0, 2),
    'e837.dat' : (1e-8, 1e-8, 2),
    's4158.dat' : (1e-8, 1e-8, 2),
    's9037.dat' : (0, 1e-8, 2),
    'c141d.dat' : (1e-7, 1e-7, 2), 
    's3024.dat' : (0., 1e-8),
    'psu94097.dat' : (1e-8, 0),
    'naca653618.dat' : (1e-8, 0),
    's8023.dat' : (1e-8, 1e-8),
    's8036.dat' : (0., 1e-8, 2),
    's4310.dat' : (1e-9, 1e-8, 2),
    'fxlv152.dat' : (1e-6, 1e-6, 2),
    'sd2030.dat' : (1e-9, 1e-8, 2),
    's9000.dat' : (0., 1e-8, 2),
    'dormoy.dat' : (1e-9, 0., 2),
    'e328.dat' : (1e-8, 0., 2),
    'fx84w175.dat' : (1e-8, 1e-8, 2),
    'eh0009.dat' : (1e-9, 1e-8, 2),
    'th25816.dat' : (1e-9, 1e-8, 2), 
    'ah79k132.dat' : (1e-8, 1e-8, 2),
    'fx76100.dat' : (1e-8, 1e-8, 2),
    'ag46ct02r.dat' : (1e-8, 1e-8, 2),
    'e635.dat' : (1e-8, 1e-9, 2),
    'fx6310.dat' : (1e-8, 0, 2),
    'e387.dat' : (0., 1e-8, 2),
    'e682.dat' : (0., 1e-8, 2),
    'fx79l120.dat' : (1e-7, 1e-7, 2),
    'naca654421a05.dat' : (1e-8, 0., 2),
    'df102.dat' : (1e-8, 1e-9, 2), 
    'jx-gs-15.dat' : (0., 1e-9),
    'arad20.dat' : (1e-9, 1e-9, 2),
    'joukowsk.dat' : (1e-8, 1e-8, 2),
    'mh112.dat' : (1e-8, 0, 2),
    'fx66s171.dat' : (1e-8, 1e-9, 2),
    'fx601261.dat' : (1e-6, 1e-6, 2),
    'fx73cl2152.dat' : (1e-8, 1e-8, 2),
    'k3311sm.dat' : (1e-8, 1e-8, 2),
    'goe777.dat' : (1e-8, 0, 2),
    'e587.dat' : (1e-8, 0),
    'defcnd3.dat' : (1e-8, 1e-9, 2),
    'e62.dat' : (0., 1e-8, 2), 
    'e392.dat' : (1e-8, 0), 
    's2050.dat' : (1e-8, 0, 2),
    'fx83w160.dat' : (1e-8, 1e-8, 2), 
    'fx74080.dat' : (1e-8, 1e-8, 2), 
    'e639.dat' : (1e-8, 1e-8, 2), 
    'e64.dat' : (0., 1e-8, 2), 
    'e678.dat' : (0., 1e-8, 2), 
    'e908.dat' : (0, 1e-8),
    's2091.dat' : (1e-8, 0),
    'e544.dat' : (1e-8, 0),
    'ag47c03.dat' : (1e-8, 0),
    'mh32.dat' : (0., 1e-8),
    's3014.dat' : (1e-8, 1e-7, 2),
    'naca23012.dat' : (1e-8, 0., 2),
    'zv15_35.dat' : (0., 1e-8),
    'ag26.dat' : (1e-8, 1e-8),
    'sg6043.dat' : (1e-9, 1e-8),
    'n0009sm.dat' : (1e-9, 1e-9, 2),
    'mh95.dat' : (1e-8, 1e-8, 2),
    's3021.dat' : (1e-10, 1e-8, 2),
    'e210.dat' : (1e-8, 1e-9, 2),
    'fx74cl5140.dat' : (1e-7, 0, 2),
    's4083.dat' : (0., 1e-8, 2),
    'sd8020.dat' : (1e-8, 1e-8, 2),
    's1014.dat' : (1e-8, 1e-8, 2),
    'e231.dat' : (1e-7, 1e-9),
    'n2415.dat' : (1e-7, 1e-9, 2),
    's3002.dat' : (0., 1e-8),
    'sa7038.dat' : (1e-7, 1e-7, 2),
    'e852.dat' : (1e-8, 0),
    's3016.dat' : (1e-8, 1e-8),
    'hq2511.dat' : (1e-8, 1e-8, 2),
    'mh78.dat' : (1e-8, 0., 2),
    'mh81.dat' : (1e-8, 0., 2),
    'fx76120.dat' : (1e-8, 1e-8, 2),
    'npl9615.dat' : (1e-10, 1e-7, 2),
    's1048.dat' : (1e-8, 1e-8, 2),
    'e331.dat' : (1e-8, 0.),
    'ua2-180sm.dat' : (1e-8, 1e-11),
    'ch10sm.dat' : (1e-8, 1e-8, 2),
    's2048.dat' : (1e-7, 1e-7, 2),
    'e203.dat' : (1e-8, 1e-8, 2),
    'n11h9.dat' : (3e-8, 0., 3),
    'mh84.dat' : (1e-8, 0., 2),
    'fx77w121.dat' : (1e-8, 1e-9, 2),
    'ag46c03.dat' : (1e-8, 0),
    'e335.dat' : (1e-8, 0., 2),
    's4095.dat' : (1e-9, 1e-7, 2),
    'sd7034.dat' : (1e-7, 0, 2),
    'ua2-180.dat' : (1e-10, 1e-6, 2),
    'e580.dat' : (1e-9, 1e-8, 2),
    'arad13.dat' : (1e-7, 1e-7, 2),
    's1010.dat' : (1e-7, 1e-7, 2),
    'fx77w270s.dat' : (1e-7, 1e-7, 2),
    'usnps4.dat' : (1e-7, 1e-7, 2),
    'goe675.dat' : (1e-7, 0, 2),
    'naca66-018.dat' : (1e-8, 1e-8, 2),
    'la2573a.dat' : (1e-8, 1e-8, 2),
    's2060.dat' : (1e-8, 0, 2),
    'ht13.dat' : (1e-8, 1e-8), 
    'lwk80150k25.dat' : (1e-8, 1e-8, 2), 
    'hq1511.dat' : (1e-8, 1e-8, 2), 
    'fx63143.dat' : (5e-6, 5e-6, 2),
    'e423.dat' : (1e-7, 0, 2),
    'lrn1015.dat' : (1e-7, 0, 2),
    'e230.dat' : (1e-7, 0, 2),
    'sc1094r8.dat' : (1e-10, 1e-8, 2),
    'dfvlrr4.dat' : (1e-8, 0), 
    's6062.dat' : (1e-8, 0, 2), 
    'mh70.dat' : (1e-8, 0), 
    'mh83.dat' : (1e-8, 1e-8, 2), 
    'rg15.dat' : (1e-8, 1e-7, 2), 
    'naca643418.dat' : (1e-8, 0, 2), 
    'fx77w343.dat' : (3e-6, 3e-6, 2),
    'fx77w258.dat' : (3e-6, 3e-6, 2),
    'e591.dat' : (1e-8, 1e-8, 2),
    'naca633418.dat' : (1e-8, 0, 2),
    'e638.dat' : (1e-8, 1e-10, 2),
    'mh200.dat' : (1e-8, 0),
    'chen.dat' : (0, 1e-8),
    'vr7b.dat' : (1e-8, 0, 2),
    'fx66a175.dat' : (3e-7, 0, 2),
    'fx72ls160.dat' : (0, 3e-7, 2),
    'sd7032.dat' : (1e-8, 1e-9, 2),
    'e385.dat' : (0, 1e-7, 2),
    'e434.dat' : (1e-8, 1e-9, 2),
    'e642.dat' : (1e-11, 1e-8, 2),
    's7055.dat' : (1e-8, 1e-8, 2),
    'e374.dat' : (1e-8, 1e-8, 2),
    'fx05191.dat' : (3e-7, 3e-7, 2),
    'fx73k170.dat' : (1e-8, 1e-8, 2),
    'rc12b3.dat' : (1e-7, 1e-7, 2),
    'ah21-7.dat' : (1e-7, 1e-7, 2),
    's4094.dat' : (1e-8, 1e-7, 2),
    's3025.dat' : (0, 1e-7),
    'e546.dat' : (1e-8, 0),
    'e472.dat' : (1e-7, 1e-7, 2),
    'rc08n1.dat' : (1e-8, 5e-6, 2),
    'l188tip.dat' : (1e-8, 1e-8, 2),
    'naca2411.dat' : (1e-7, 1e-8, 2),
    'goe571.dat' : (0., 1e-8, 2),
    'e228.dat' : (1e-9, 1e-8, 2),
    'e336.dat' : (1e-8, 0),
    'oaf095.dat' : (1e-8, 1e-8, 2),
    'fx60100.dat': (1e-6, 1e-6, 3),
    'mh108.dat': (0, 1e-9),
    'ah82150a.dat': (1e-8, 1e-8, 2),
    's8037.dat': (0, 1e-8),
    'ag08.dat' : (0., 1e-9),
    'clarkysm.dat' : (1e-8, 1e-8, 2),
    'sd6080.dat' : (0., 1e-8), 
    's4320.dat' : (1e-9, 1e-7, 2), 
    'j5012.dat' : (1e-8, 1e-8, 2), 
    'fx73cl1152.dat' : (1e-8, 1e-8, 2),
    'fx79w151a.dat' : (1e-8, 1e-8, 2),
    'mh117.dat' : (0., 1e-9),
    'mrc-20.dat' : (1e-8, 1e-9),
    'n0011sc.dat' : (1e-8, 1e-8, 2),
    'nl31t.dat' : (1e-8, 1e-7, 2),
    'ht22.dat' : (1e-8, 1e-9, 2),
    'fx69274.dat' : (1e-8, 1e-8, 2),
    'mh94.dat' : (1e-8, 0), 
    'jx-gs-06.dat' : (1e-9, 1e-8),
    's2062.dat' : (1e-8, 0., 2),
    'm665.dat' : (1e-10, 1e-8, 2),
    'e67.dat' : (1e-8, 1e-8),
    'usa40.dat' : (3e-6, 0, 2),
    'e341.dat' : (1e-8, 0),
    'ag11.dat' : (0., 1e-8), 
    'c141f.dat' : (1e-7, 1e-7, 2),
    'rc10b3.dat' : (1e-8, 1e-8, 2),
    'e428.dat' : (1e-8, 1e-7, 2),
    'b737c.dat' : (1e-8, 1e-6, 3),
    'goe498.dat' : (1e-7, 0, 2),
    'e417.dat' : (0., 1e-8, 2),
    'ah83150q.dat' : (1e-8, 1e-7, 2),
    'gm15sm.dat' : (1e-7, 1e-7, 2),
    'sd7003.dat' : (1e-7, 0, 2),
    'lg10sc.dat' : (1e-8, 1e-7, 2),
    'fx78k161.dat' : (1e-6, 1e-6, 2),
    'naca633018.dat' : (1e-8, 1e-8, 2),
    'prandtl-d-centerline.dat' : (1e-7, 1e-7, 2),
    'mh18.dat' : (1e-11, 1e-8),
    'fx66s161.dat' : (1e-8, 1e-9, 2),
    'e474.dat' : (1e-8, 1e-9, 2),
    'e426.dat' : (1e-12, 1e-8, 3),
    'bqm34.dat' : (1e-8, 1e-8, 2),
    'mrc-16.dat' : (1e-9, 0, 3),
    'e694.dat' : (0., 1e-8, 2),
    'hq300gd2.dat' : (1e-8, 1e-8, 2),
    'e171.dat' : (1e-8, 1e-8, 2),
    'esa40.dat' : (1e-8, 1e-7, 2),
    'eh2012.dat' : (1e-8, 1e-8, 2),
    'fx61168.dat' : (1e-6, 0, 3),
    'naca64209.dat' : (0., 1e-6, 2), 
    's9029.dat' : (1e-8, 1e-8, 2),
    'vr7.dat' : (1e-7, 0, 2),
    'usa35.dat' : (1e-7, 0, 2),
    's9032.dat' : (1e-8, 1e-8, 2),
    's9026.dat' : (1e-8, 1e-8, 2),
    'nl722343.dat' : (1e-8, 0, 2),
    'fx71089a.dat' : (1e-7, 1e-7, 2),
    'e856.dat' : (1e-10, 1e-8),
    'nl722362.dat' : (1e-8, 0, 2),
    'sd7080.dat' : (1e-9, 1e-7, 2),
    'dea11.dat' : (1e-8, 1e-9, 2),
    'e562.dat' : (1e-7, 1e-9, 2),
    'e1098.dat' : (1e-9, 1e-7, 2),
    'e403.dat' : (0., 1e-8, 2),
    'eh3012.dat'  : (1e-9, 0),
    'oaf139.dat' : (1e-8, 1e-8, 2),
    'n6409.dat' : (1e-8, 0, 2),
    'e333.dat' : (1e-8, 0, 2),
    'ag27.dat' : (0., 1e-8, 2),
    'ah93w257.dat' : (1e-9, 1e-9, 2),
    'defcnd1.dat' : (1e-9, 1e-8, 2),
    'ah79k135.dat' : (1e-8, 1e-8, 2),
    'e344.dat' : (1e-8, 0, 2),
    'oaf128.dat' : (1e-8, 1e-8, 2),
    'e399.dat' : (1e-8, 0, 2),
    'ah93w215.dat' : (1e-8, 1e-8, 2),
    'ag45ct02r.dat' : (1e-8, 1e-9, 2),
    'oaf117.dat' : (1e-8, 1e-8, 2), 
    'sd7043.dat' : (1e-8, 0), 
    'bw050209.dat' : (1e-6, 1e-7, 2),
    'e66.dat' : (1e-7, 0, 2),
    'e340.dat' : (1e-7, 0, 2),
    'e748.dat' : (1e-8, 1e-7, 2),
    'apex16.dat' : (1e-6, 1e-6, 2),
    's1223rtl.dat' : (0., 1e-8, 2),
    'mh33.dat' : (0., 1e-8),
    'fx71120.dat' : (1e-6, 1e-6, 2),
    'goe522.dat' : (1e-7, 1e-7, 2),
    'nn7mk20.dat' : (1e-8, 1e-8, 2),
    'fx74modsm.dat' : (1e-8, 1e-6, 2),
    'rg1410.dat' : (1e-8, 1e-8, 2),
    'e226.dat' : (0., 1e-8, 2),
    'e636.dat' : (1e-9, 1e-8, 2),
    'lwk80120k25.dat' : (1e-8, 1e-8, 2),
    'lwk80100.dat' : (1e-8, 1e-8, 2),
    'e195.dat' : (0., 1e-8, 2),
    'e543.dat' : (1e-8, 0., 2),
    'e209.dat' : (1e-8, 1e-8, 2),
    'e1200.dat' : (1e-8, 0, 2),
    's6061.dat' : (1e-7, 0, 2),
    'e637.dat' : (0, 1e-7, 2),
    'rg15a111.dat' : (1e-8, 1e-9, 2),
    'e818.dat' : (1e-8, 1e-9),
    's1046.dat' : (1e-7, 1e-7, 2),
    'e180.dat' : (1e-7, 0, 2),
    'fx60177.dat' : (1e-7, 0, 2),
    's8065.dat' : (1e-7, 0, 2),
    'goe795sm.dat' : (1e-8, 1e-8, 2),
    'atr72sm.dat' : (1e-8, 1e-8, 2),
    'goe115.dat' : (0, 1e-6, 2),
    's102b.dat' : (1e-7, 1e-8, 2), 
    'eh1090.dat' : (1e-8, 1e-8, 2), 
    'b707e.dat' : (1e-6, 0, 2), 
    'e176.dat' : (0., 1e-7, 2),
    'fx73cl3152.dat' : (1e-7, 1e-7, 2),
    's1223.dat' : (0., 1e-8, 2),
    'eh1070.dat' : (1e-8, 1e-8, 2),
    'ah80129.dat' : (1e-8, 1e-8, 2),
    'lwk80080.dat' : (1e-8, 1e-8, 2),
    'fx79k144.dat' : (1e-7, 1e-7, 2),
    'falcon.dat' : (1e-7, 1e-7, 2),
    'n8h12.dat' : (1e-7, 0, 2), 
    'fx80080.dat' : (1e-8, 1e-8, 2),
    'c141a.dat' : (1e-8, 1e-8, 2),
    'sp4721la.dat' : (0., 1e-8),
    'ag12.dat' : (1e-11, 1e-9, 2), 
    'mh42.dat' : (1e-11, 1e-8, 2), 
    'fg4.dat' : (4e-6, 1e-8, 2),
    'mh25.dat' : (0, 1e-8),
    'fx68h120.dat' : (1e-8, 1e-8, 2),
    'lnv109a.dat' : (1e-8, 1e-8, 2),
    'jn153.dat' : (0., 1e-8, 2), 
    'e205.dat' : (0., 1e-7, 2), 
}

counter = 0
counter_2 = 0

smt = False
fig, axs = plt.subplots(1, 2, figsize=(14, 10))
# plt.figure(figsize=(12, 12))
ax = plt.gca()

b_sp_interp_error_upper = []
b_sp_interp_error_lower = []

airfoil_dict = {}

for file in os.listdir(uiuc_airfoil_dir):
    filename = os.fsdecode(file)
    # remove any Identifier files
    if 'Identifier' in filename:
        os.remove(UIUC_AIRFOILS/filename)
    
    elif filename in bad_airfoils:
        pass
        # airfoil_coords = np.genfromtxt(UIUC_AIRFOILS/filename, skip_header=1)
        # x = airfoil_coords[:, 0]
        # y = airfoil_coords[:, 1]

        # zero_indices = np.where(x == 0)[0]

        # # If there are x-coordinates that are exactly zero
        # if len(zero_indices) == 0: 
        #     min_x_idx = np.where(x == np.min(x))[0]
        #     if len(min_x_idx) == 2:
        #         x_upper = x[0:min_x_idx[1]]
        #         x_lower = x[min_x_idx[1]:]

        #         y_upper = y[0:min_x_idx[1]]
        #         y_lower = y[min_x_idx[1]:]

        #         y_LE = y[min_x_idx[1]]
        #         x_LE = x[min_x_idx[1]]


        #     elif len(min_x_idx) == 1:
        #         x_upper = x[0:min_x_idx[0]+1]
        #         x_lower = x[min_x_idx[0]:]

        #         y_upper = y[0:min_x_idx[0]+1]
        #         y_lower = y[min_x_idx[0]:]

        #         y_LE = y[min_x_idx[0]]
        #         x_LE = x[min_x_idx[0]]

            
        #     else:
        #         print(x)
        #         raise NotImplementedError
        
        # elif len(zero_indices) >= 1: 
        #     zero_index = zero_indices[0]
        #     x_upper = x[0:zero_index+1]
        #     x_lower = x[zero_index:]

        #     y_upper = y[0:zero_index+1]
        #     y_lower = y[zero_index:]

        #     y_LE = y[zero_index]
        #     x_LE = x[zero_index]

        # else:
        #     print(zero_indices)
        #     raise NotImplementedError

        # x_max_upper = np.max(x_upper)
        # x_min_upper = np.min(x_upper)

        # x_max_lower = np.max(x_lower)
        # x_min_lower = np.min(x_lower)

        # scaled_x_upper = (x_upper - x_min_upper) / (x_max_upper - x_min_upper)
        # scaled_x_lower = (x_lower - x_min_lower) / (x_max_lower - x_min_lower)

        
        # scaled_y_upper = (y_upper- y_LE - x_min_upper) / (x_max_upper - x_min_upper)
        # scaled_y_lower = (y_lower -y_LE - x_min_lower) / (x_max_lower - x_min_lower)
        
        # # print(scaled_x_upper)
        # # print(scaled_y_upper)
        # # exit()
        # # scaled_y_upper = (y_upper - x_min_upper) / (x_max_upper - x_min_upper)
        # # scaled_y_lower = (y_lower - x_min_lower) / (x_max_lower - x_min_lower)

        # if smt: 
        #     xlimits = np.array([[0., 1.]])
        #     sm_upper = RMTB(
        #         print_global=False,
        #         xlimits=xlimits,
        #         order=4,
        #         num_ctrl_pts=int(10 * len(scaled_x_upper.flatten())), 
        #         energy_weight=1e-16,
        #         regularization_weight=1e-16,
        #     )
        #     sm_upper.set_training_values((scaled_x_upper.flatten()), (scaled_y_upper.flatten()))
        #     sm_upper.train()

        #     sm_lower = RMTB(
        #         print_global=False,
        #         xlimits=xlimits,
        #         order=4,
        #         num_ctrl_pts=int(10 * len(scaled_x_lower.flatten())),  
        #         energy_weight=1e-16,
        #         regularization_weight=1e-16,
        #     )
        #     sm_lower.set_training_values(scaled_x_lower.flatten(), scaled_y_lower.flatten())
        #     sm_lower.train()

        #     y_interp_upper = sm_upper.predict_values(scaled_x_upper)
        #     y_interp_lower = sm_lower.predict_values(scaled_x_lower)

        #     y_dense_upper = sm_upper.predict_values(x_interp)
        #     y_dense_lower = sm_lower.predict_values(x_interp)

        # else:
        #     # y_interp_1d_upper_fun = scipy.interpolate.interp1d(scaled_x_upper, scaled_y_upper, kind='cubic')
        #     # y_interp_1d_lower_fun = scipy.interpolate.interp1d(scaled_x_lower, scaled_y_lower, kind='cubic')

        #     # y_interp_upper = y_interp_1d_upper_fun(scaled_x_upper)
        #     # y_interp_lower = y_interp_1d_lower_fun(scaled_x_lower)

        #     # y_dense_upper = y_interp_1d_upper_fun(x_interp)
        #     # y_dense_lower = y_interp_1d_lower_fun(x_interp)


        #     spl_upper = splrep(x=np.flip(scaled_x_upper), y=np.flip(scaled_y_upper), k=3, s=1e-10)
        #     spl_lower = splrep(x=scaled_x_lower, y=scaled_y_lower, k=3, s=1e-10)

        #     y_interp_upper = splev(scaled_x_upper, spl_upper)
        #     y_interp_lower = splev(scaled_x_lower, spl_lower)

        #     y_dense_upper = splev(x_interp, spl_upper)
        #     y_dense_lower = splev(x_interp, spl_lower)



            

        

        # # print('x_scaled_lower', scaled_x_lower)
        # print('scaled_x_upper', scaled_x_upper)

        # print('FITTING_ERROR UPPER---', np.mean(((y_interp_upper[1:-1] - scaled_y_upper[1:-1]) / scaled_y_upper[1:-1]) * 100))
        # print('FITTING_ERROR LOWER---', np.mean(((y_interp_lower[1:-1] - scaled_y_lower[1:-1]) / scaled_y_lower[1:-1]) * 100))
        
        

        
        # print(x_LE, y_LE)
        # print('\n')

        # color = next(ax._get_lines.prop_cycler)['color']
        # # plt.plot(x, y ,color=color, label=filename)
        # # plt.plot(x_lower-x_LE, y_lower-y_LE ,color=color, label=filename)
        # # plt.plot(x_upper-x_LE, y_upper-y_LE ,color=color)

        # plt.scatter(scaled_x_upper, scaled_y_upper, color=color, label=filename, s=4)
        # plt.scatter(scaled_x_lower, scaled_y_lower, color=color, s=4)

        # # plt.plot(scaled_x_upper, y_interp_upper, color=color)
        # # plt.plot(scaled_x_lower, y_interp_lower, color=color)

        # plt.plot(x_interp, y_dense_upper, color=color)
        # plt.plot(x_interp, y_dense_lower, color=color)

        # plt.axis('equal')
        # plt.grid()
        # plt.legend()


    else:
        counter += 1
        airfoil_coords = np.genfromtxt(UIUC_AIRFOILS/filename, skip_header=1)

        if counter < 0:
            pass

        else:
            if counter <1:
                pass
            elif counter >= 1:
                x = airfoil_coords[:, 0]
                y = airfoil_coords[:, 1]

                if os.path.isdir(UIUC_AIRFOILS_2/filename.removesuffix('.dat')):
                    pass
                else:
                    os.makedirs(UIUC_AIRFOILS_2/filename.removesuffix('.dat'))
                    np.savetxt(UIUC_AIRFOILS_2/filename.removesuffix('.dat')/f"{filename.removesuffix('.dat')}_raw.txt", airfoil_coords)

                zero_indices = np.where(x == 0)[0]

                # If there are x-coordinates that are exactly zero
                if len(zero_indices) == 0: 
                    min_x_idx = np.where(x == np.min(x))[0]
                    if len(min_x_idx) == 2:
                        x_upper = x[0:min_x_idx[1]]
                        x_lower = x[min_x_idx[1]:]

                        y_upper = y[0:min_x_idx[1]]
                        y_lower = y[min_x_idx[1]:]

                        y_LE = y[min_x_idx[1]]
                        x_LE = x[min_x_idx[1]]


                    elif len(min_x_idx) == 1:
                        x_upper = x[0:min_x_idx[0]+1]
                        x_lower = x[min_x_idx[0]:]

                        y_upper = y[0:min_x_idx[0]+1]
                        y_lower = y[min_x_idx[0]:]

                        y_LE = y[min_x_idx[0]]
                        x_LE = x[min_x_idx[0]]

                    
                    else:
                        print(x)
                        raise NotImplementedError
                
                elif len(zero_indices) >= 1: 
                    zero_index = zero_indices[0]
                    x_upper = x[0:zero_index+1]
                    x_lower = x[zero_index:]

                    y_upper = y[0:zero_index+1]
                    y_lower = y[zero_index:]

                    y_LE = y[zero_index]
                    x_LE = x[zero_index]

                else:
                    print(zero_indices)
                    raise NotImplementedError
                
                lower_dup = find_duplicate_indices(x_lower)
                if lower_dup:
                    print('DUPLICATES LOWER', filename, lower_dup)
                    print(lower_dup[0])
                    if len(lower_dup) > 2:
                        x_lower = np.delete(x_lower, lower_dup[0:2], 0)
                        y_lower = np.delete(y_lower, lower_dup[0:2], 0)
                    else:
                        x_lower = np.delete(x_lower, lower_dup[0], 0)
                        y_lower = np.delete(y_lower, lower_dup[0], 0)
                upper_dup = find_duplicate_indices(x_upper)
                if upper_dup:
                    print('DUPLICATES UPPER', filename, upper_dup)
                    x_upper = np.delete(x_upper, upper_dup[0], 0)
                    y_upper = np.delete(y_upper, upper_dup[0], 0)

                x_max_upper = np.max(x_upper)
                x_min_upper = np.min(x_upper)

                x_max_lower = np.max(x_lower)
                x_min_lower = np.min(x_lower)

                if filename == 'ua79sff.dat':
                    scaled_x_upper = (x_upper - x_min_upper) / (x_max_upper - x_min_upper)
                    scaled_x_lower = (x_lower - x_min_lower) / (x_max_lower - x_min_lower) 

                    scaled_y_upper = (y_upper -y_LE) / (x_max_upper - x_min_upper)
                    scaled_y_lower = (y_lower - y_LE) / (x_max_lower - x_min_lower)

                elif filename == 'b707b.dat':
                    scaled_x_lower = x_lower
                    scaled_x_upper = np.flip(x_upper)
                    scaled_y_lower = y_lower
                    scaled_y_upper = np.flip(y_upper)
                    # print(filename)
                    print(x)
                    print(y)
                    print(x_upper)
                    print(y_upper)
                    # print(y_lower)
                
                else:
                    scaled_x_upper = (x_upper - x_min_upper) / (x_max_upper - x_min_upper)
                    scaled_x_lower = (x_lower - x_min_lower) / (x_max_lower - x_min_lower)
                    
                    scaled_y_upper = (y_upper- y_LE - x_min_upper) / (x_max_upper - x_min_upper)
                    scaled_y_lower = (y_lower -y_LE - x_min_lower) / (x_max_lower - x_min_lower)
                
                # print(scaled_x_upper)
                # print(scaled_y_upper)
                # exit()
                # scaled_y_upper = (y_upper - x_min_upper) / (x_max_upper - x_min_upper)
                # scaled_y_lower = (y_lower - x_min_lower) / (x_max_lower - x_min_lower)

                if smt: 
                    xlimits = np.array([[0., 1.]])
                    sm_upper = RMTB(
                        print_global=False,
                        xlimits=xlimits,
                        order=3,
                        num_ctrl_pts=int(10 * len(scaled_x_upper.flatten())), 
                        energy_weight=1e-16,
                        regularization_weight=1e-16,
                    )
                    sm_upper.set_training_values((scaled_x_upper.flatten()), (scaled_y_upper.flatten()))
                    sm_upper.train()

                    sm_lower = RMTB(
                        print_global=False,
                        xlimits=xlimits,
                        order=3,
                        num_ctrl_pts=int(10 * len(scaled_x_lower.flatten())),  
                        energy_weight=1e-16,
                        regularization_weight=1e-16,
                    )
                    sm_lower.set_training_values(scaled_x_lower.flatten(), scaled_y_lower.flatten())
                    sm_lower.train()

                    y_interp_upper = sm_upper.predict_values(scaled_x_upper)
                    y_interp_lower = sm_lower.predict_values(scaled_x_lower)

                    y_dense_upper = sm_upper.predict_values(x_interp)
                    y_dense_lower = sm_lower.predict_values(x_interp)

                else:
                    # # if filename == 'n63210.dat':
                    if filename in airfoils_smoothing.keys():
                        print(f'---------------------------------{filename}')
                        s_upper = airfoils_smoothing[filename][0]
                        s_lower = airfoils_smoothing[filename][1]
                        if len(airfoils_smoothing[filename]) == 3:
                            k=airfoils_smoothing[filename][2]
                            w_upper = np.linspace(1, 1, len(scaled_x_upper))
                            w_lower = np.linspace(1, 1, len(scaled_x_lower))
                        else:
                            k=3
                            w_upper = np.linspace(1., 0.5, len(scaled_x_upper))
                            w_lower = np.linspace(1., 0.5, len(scaled_x_lower))
                    # #         if filename in ['goe15k.dat', 'n63210.dat', 'ah79100a.dat', 'p51hroot.dat']:
                    # #             s_upper = 1e-6
                    # #             s_lower = 1e-8
                    # #             k=2
                    # #         else:
                    # #             s_upper = s_lower = 1e-8
                    # #             k=2
                    # #     else:
                    # #         s_upper = s_lower = 1e-8
                    # #         k=3
                    #         s_upper = 1e-8
                    #         s_lower = 1e-8
                    #         k=3
                    #         w_upper = np.linspace(1, 0.7, len(scaled_x_upper))
                    #         w_lower = np.linspace(1, 0.7, len(scaled_x_lower))
                    else:    
                        s_upper = 0# 1e-10
                        s_lower = 0 # 1e-10
                        k=3
                        w_upper = np.linspace(1.5, 0.5, len(scaled_x_upper))
                        w_lower = np.linspace(1.5, 0.5, len(scaled_x_lower))
            

                    if scaled_x_upper[0] != 0:
                        spl_upper = splrep(x=np.flip(scaled_x_upper), y=np.flip(scaled_y_upper), k=k, w=w_upper, s=s_upper)
                    else:
                        spl_upper = splrep(x=scaled_x_upper, y=scaled_y_upper, k=k, w=w_upper, s=s_upper)
                    if scaled_x_lower[0] != 0:
                        
                        # print(scaled_y_lower)
                        spl_lower = splrep(x=np.flip(scaled_x_lower), y=np.flip(scaled_y_lower), k=k, w=w_lower, s=s_lower)
                    else:
                        spl_lower = splrep(x=scaled_x_lower, y=scaled_y_lower, k=k, w=w_lower, s=s_lower)


                    y_interp_upper = splev(scaled_x_upper, spl_upper)
                    y_interp_lower = splev(scaled_x_lower, spl_lower)

                    y_dense_upper = splev(x_interp, spl_upper)
                    y_dense_lower = splev(x_interp, spl_lower)
                    # else:

                    #     y_interp_1d_upper_fun = scipy.interpolate.interp1d(scaled_x_upper, scaled_y_upper, kind='cubic')
                    #     y_interp_1d_lower_fun = scipy.interpolate.interp1d(scaled_x_lower, scaled_y_lower, kind='cubic')

                    #     y_interp_upper = y_interp_1d_upper_fun(scaled_x_upper)
                    #     y_interp_lower = y_interp_1d_lower_fun(scaled_x_lower)

                    #     y_dense_upper = y_interp_1d_upper_fun(x_interp)
                    #     y_dense_lower = y_interp_1d_lower_fun(x_interp)
                    regularization = [0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
                    errors_upper = [100]
                    errors_lower = [100]
                    c_upper_list = [0]
                    c_lower_list = [0]
                    for reg in regularization:
                    # Upper/Lower
                        c_upper, B_upper = fit_b_spline(y_dense_upper, len(control_points), b_spline_order=4, alpha=0, u=ux_dense)
                        c_lower, B_lower = fit_b_spline(y_dense_lower, len(control_points), b_spline_order=4, alpha=0, u=ux_dense)

                        y_upper_b_sp = B_upper @ c_upper
                        y_lower_b_sp = B_lower @ c_lower

                        b_spline_error_upper = np.mean(abs(y_upper_b_sp[1:-1] - y_dense_upper[1:-1]) / abs(y_dense_upper[1:-1]) * 100)
                        b_spline_error_lower = np.mean(abs(y_lower_b_sp[1:-1] - y_dense_lower[1:-1]) / abs(y_dense_lower[1:-1]) * 100)

                        if b_spline_error_upper < errors_upper[0]:
                            errors_upper[0] = b_spline_error_upper
                            c_upper_list[0] = c_upper
                        
                        if b_spline_error_lower < errors_lower[0]:
                            errors_lower[0] = b_spline_error_lower
                            c_lower_list[0] = c_lower

                        

                    
                    # pass
                        
                c_upper = c_upper_list[0]
                c_lower = c_lower_list[0]
                y_upper_b_sp = B_upper @ c_upper
                y_lower_b_sp = B_lower @ c_lower

                interpolated_coords = np.zeros((300, 2))
                interpolated_coords[0:150, 0] = np.flip(x_interp)
                interpolated_coords[0:150, 1] = np.flip(y_upper_b_sp)
                interpolated_coords[150:, 0] = x_interp
                interpolated_coords[150:, 1] = y_lower_b_sp
                # print(filename)
                np.savetxt(UIUC_AIRFOILS_2/filename.removesuffix('.dat')/f"{filename.removesuffix('.dat')}_interpolated.txt", interpolated_coords)
                # plt.plot(interpolated_coords[:, 0], interpolated_coords[:, 1])
                # plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1])
                # plt.axis('equal')
                # print(interpolated_coords)
                # print(interpolated_coords.shape)
                # plt.show()
                # exit()

                interpolation_error_upper = np.mean(((y_interp_upper[1:-1] - scaled_y_upper[1:-1]) / scaled_y_upper[1:-1]) * 100)
                interpolation_error_lower = np.mean(((y_interp_lower[1:-1] - scaled_y_lower[1:-1]) / scaled_y_lower[1:-1]) * 100)

                print('B-spline error UPPER---', errors_upper[0])
                print('B-spline error LOWER---', errors_lower[0])
                print('FITTING_ERROR UPPER---', interpolation_error_upper)
                print('FITTING_ERROR LOWER---', interpolation_error_lower)

                

            
                # color = next(ax._get_lines.prop_cycler)['color']

                # axs[0].clear()
                # axs[0].scatter(scaled_x_upper, scaled_y_upper, color=color, label=filename, s=4)
                # axs[0].scatter(scaled_x_lower, scaled_y_lower, color=color, s=4)

                

                # axs[0].plot(x_interp, y_dense_upper, color=color)
                # axs[0].plot(x_interp, y_dense_lower, color=color)

                # axs[0].plot(x_interp, y_upper_b_sp, color=color)
                # axs[0].plot(x_interp, y_lower_b_sp, color=color)

                # axs[0].axis('equal')
                # axs[0].set_xlim([-0.01, 0.5])
                # axs[0].grid()
                # axs[0].legend()

                # axs[1].clear()
                # axs[1].scatter(scaled_x_upper, scaled_y_upper, color=color, label=filename, s=4)
                # axs[1].scatter(scaled_x_lower, scaled_y_lower, color=color, s=4)

                # axs[1].plot(x_interp, y_dense_upper, color=color)
                # axs[1].plot(x_interp, y_dense_lower, color=color)

                # # axs[1].plot(x_interp, y_upper_b_sp, color=color)
                # # axs[1].plot(x_interp, y_lower_b_sp, color=color)

                # axs[1].plot(scaled_x_upper, scaled_y_upper, color=color)
                # axs[1].plot(scaled_x_lower, scaled_y_lower, color=color)

                # # axs[1].plot(scaled_x_upper, y_interp_upper, color='r')
                # # axs[1].plot(scaled_x_lower, y_interp_lower, color='r')
                
                # # axs[1].plot(x, y, color='r')

                # axs[1].axis('equal')
                # axs[1].set_xlim([0.5, 1.01])
                # axs[1].grid()
                # axs[1].legend()

                # plt.suptitle(f'# {counter} ' + f'Interpolation error: {np.mean((abs(interpolation_error_upper), abs(interpolation_error_lower)))}' + '\n' + 
                #         f'Interpolation error b_spline: {np.mean((abs(errors_upper[0]), abs(errors_lower[0])))}')
                # # plt.pause(0.1)    
                # plt.waitforbuttonpress()    


            # # Find index of leading edge
            # zero_indices = np.where(x == 0)[0]

            # # If there are x-coordinates that are exactly zero
            # if len(zero_indices) == 0: 
            #     min_x_idx = np.where(x == np.min(x))[0]
            #     if len(min_x_idx) == 2:
            #         x_upper = x[0:min_x_idx[1]]
            #         x_lower = x[min_x_idx[1]:]

            #         y_upper = y[0:min_x_idx[1]]
            #         y_lower = y[min_x_idx[1]:]

            #     elif len(min_x_idx) == 1:
            #         x_upper = x[0:min_x_idx[0]+1]
            #         x_lower = x[min_x_idx[0]:]

            #         y_upper = y[0:min_x_idx[0]+1]
            #         y_lower = y[min_x_idx[0]:]
                
            #     else:
            #         print(x)
            #         raise NotImplementedError
            
            # elif len(zero_indices) >= 1: 
            #     zero_index = zero_indices[0]
            #     x_upper = x[0:zero_index+1]
            #     x_lower = x[zero_index:]

            #     y_upper = y[0:zero_index+1]
            #     y_lower = y[zero_index:]

            # else:
            #     print(zero_indices)
            #     raise NotImplementedError
            
            
            # x_upper = np.flip(x_upper)
            # y_upper = np.flip(y_upper)
            # x_upper[0] = 0
            # x_upper[-1] = 1
            # # y_upper[0] = 0
            # y_upper[-1] = 0

            # x_lower[0] = 0
            # x_lower[-1] = 1
            # # y_lower[0] = 0
            # y_lower[-1] = 0

            # if y_lower[0] > 0 and y_lower[1] > 0:
            #     print(counter, filename)
            #     color = next(ax._get_lines.prop_cycler)['color']
            #     plt.plot(x_upper, y_upper ,color=color, label=filename)
            #     plt.plot(x_lower, y_lower ,color=color)
            #     plt.axis('equal')
            #     plt.legend()

            # if y_lower[0] > 0.01:
            #     print(counter, filename)
            #     color = next(ax._get_lines.prop_cycler)['color']
            #     plt.plot(x_upper, y_upper ,color=color, label=filename)
            #     plt.plot(x_lower, y_lower ,color=color)
            #     plt.axis('equal')
            #     plt.legend()

            # if y_lower[-1] > 0.01 or y_upper[-1] < -0.01:
            #     print(counter, filename)
            #     color = next(ax._get_lines.prop_cycler)['color']
            #     plt.plot(x_upper, y_upper ,color=color, label=filename)
            #     plt.plot(x_lower, y_lower ,color=color)
            #     plt.axis('equal')
            #     plt.legend()

            # if y_upper[0] > 0.005:
            #     print(counter, filename, x_upper)
            #     color = next(ax._get_lines.prop_cycler)['color']
            #     plt.plot(x_upper, y_upper ,color=color, label=filename)
            #     plt.plot(x_lower, y_lower ,color=color)
            #     plt.axis('equal')
            #     plt.legend()

            # # else:
            # #     print(counter, filename, x_upper)

            

            # lower_dup = find_duplicate_indices(x_lower)
            # if lower_dup:
            #     print(lower_dup[0])
            #     if len(lower_dup) > 2:
            #         x_lower = np.delete(x_lower, lower_dup[0:2], 0)
            #         y_lower = np.delete(y_lower, lower_dup[0:2], 0)
            #     else:
            #         x_lower = np.delete(x_lower, lower_dup[0], 0)
            #         y_lower = np.delete(y_lower, lower_dup[0], 0)
            # upper_dup = find_duplicate_indices(x_upper)
            # if upper_dup:
            #     x_upper = np.delete(x_upper, upper_dup[0], 0)
            #     y_upper = np.delete(y_upper, upper_dup[0], 0)
            # # print(lower_dup)
            # # print(upper_dup)
            # y_interp_1d_upper_fun = scipy.interpolate.interp1d(x_upper, y_upper, kind='cubic')
            # y_interp_1d_lower_fun = scipy.interpolate.interp1d(x_lower, y_lower, kind='cubic')

            

            # y_interp_upper = y_interp_1d_upper_fun(x_upper)
            # y_interp_lower = y_interp_1d_lower_fun(x_lower)

            # # print('FITTING_ERROR UPPER---', np.mean(abs(y_interp_upper[1:-1] - y_upper[1:-1]) / y_upper[1:-1] * 100))
            # # print('FITTING_ERROR LOWER---', np.mean(abs(y_interp_lower[1:-1] - y_lower[1:-1]) / y_lower[1:-1] * 100))
            
            # y_dense_upper = y_interp_1d_upper_fun(x_interp)
            # y_dense_lower = y_interp_1d_lower_fun(x_interp)

            # thickness = y_dense_upper - y_dense_lower
            # camber = (y_dense_upper + y_dense_lower) / 2

            # regularization = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 0]
            # # regularization = [0]

            # error_upper = []
            # error_lower = []
            # cp_lower_list = []
            # cp_upper_list = []


            # error_camber = []
            # error_thickness = []
            # cp_camber_list = []
            # cp_thickness_list = []
            # for reg in regularization:
            #     # # Upper/Lower
            #     # c_upper, B_upper = fit_b_spline(y_dense_upper, len(control_points), b_spline_order=4, alpha=reg, u=ux_dense)
            #     # c_lower, B_lower = fit_b_spline(y_dense_lower, len(control_points), b_spline_order=4, alpha=reg, u=ux_dense)

            #     # y_upper_b_sp = B_upper @ c_upper
            #     # y_lower_b_sp = B_lower @ c_lower

            #     # error_upper.append(np.max(abs(y_upper_b_sp[1:-1] - y_dense_upper[1:-1]) / abs(y_dense_upper[1:-1]) * 100))
            #     # error_lower.append(np.max(abs(y_lower_b_sp[1:-1] - y_dense_lower[1:-1]) / abs(y_dense_lower[1:-1]) * 100))

            #     # cp_lower_list.append(c_lower)
            #     # cp_upper_list.append(c_upper)

            #     # Camber/ thickness
            #     c_camber, B_camber = fit_b_spline(camber, len(control_points), b_spline_order=4, alpha=reg)#, u=ux_dense)
            #     c_thickness, B_thickness = fit_b_spline(thickness, len(control_points), b_spline_order=4, alpha=reg,)# u=ux_dense)

            #     y_camber_b_sp = B_camber @ c_camber
            #     y_thickness_b_sp = B_thickness @ c_thickness

            #     if np.mean(camber) >0:
            #         error_camber.append(np.max(abs(y_camber_b_sp[1:-1] - camber[1:-1]) / abs(camber[1:-1]) * 100))
            #     error_thickness.append(np.max(abs(y_thickness_b_sp[1:-1] - thickness[1:-1]) / abs(thickness[1:-1]) * 100))

            #     cp_camber_list.append(c_camber)
            #     cp_thickness_list.append(c_thickness)


            # # print(filename, x_upper, y_upper)
            # # print(filename, x_lower, y_lower)
            # # min_error_idx_upper = np.where(error_upper == min(error_upper))[0][0]
            # # min_error_idx_lower = np.where(error_lower == min(error_lower))[0][0]
            # if error_camber:
            #     min_error_idx_upper = np.where(error_camber == min(error_camber))[0][0]
            #     b_sp_error_upper = error_camber[min_error_idx_upper]
            #     cp_camber = cp_camber_list[min_error_idx_upper]
            #     b_sp_interp_error_upper.append(b_sp_error_upper)

            # else:
            #     cp_camber = c_camber
            
            # # min_error_idx_upper = np.where(error_upper == min(error_upper))[0][0]
            
            
            # # min_error_idx_upper = np.where(error_upper == min(error_upper))[0][0]
            # min_error_idx_lower = np.where(error_thickness == min(error_thickness))[0][0]
            
            # # b_sp_error_upper = error_upper[min_error_idx_upper]            
            # b_sp_error_lower = error_thickness[min_error_idx_lower]

            # # b_sp_error_upper = error_upper[0]
            # # b_sp_error_lower = error_lower[0]
            
            # if b_sp_error_lower > 1:#  or b_sp_error_upper > 1:
            #     print(filename, counter)
            #     # print('BSP FITTING ERROR CAMBER---', b_sp_error_upper)
            #     print('BSP FITTING ERROR THICKNESS---', b_sp_error_lower)

            # # b_sp_interp_error_upper.append(b_sp_error_upper)
            # b_sp_interp_error_lower.append(b_sp_error_lower)

            # # cp_upper = cp_upper_list[min_error_idx_upper]
            # # cp_lower = cp_lower_list[min_error_idx_lower]

            # cp_thickness = cp_thickness_list[min_error_idx_lower]


            # # y_upper_b_sp = B_upper @ c_upper
            # # y_lower_b_sp = B_lower @ c_lower

            # # y_camber_b_sp = B_camber @ c_camber
            # # y_thickness_b_sp = B_thickness @ c_thickness


            # airfoil_dict[filename] = {
            #     'x_upper_raw' : x_upper,
            #     'x_lower_raw' : x_lower,
            #     'y_upper_raw' : y_upper,
            #     'y_lower_raw' : y_lower,
                
            #     'upper_ctr_pts' : cp_camber,
            #     'lower_ctr_pts' : cp_thickness,
            #     'max_b_spline_error_upper' : b_sp_error_upper,
            #     'max_b_spline_error_lower' : b_sp_error_lower,
            #     # 'max_b_spline_error_upper' : error_camber[0],
            #     # 'max_b_spline_error_lower' : error_thickness[0],
            # }


            # color = next(ax._get_lines.prop_cycler)['color']
            # plt.scatter(x_upper, y_upper, s=5 ,color=color)
            # # plt.plot(x_upper, y_interp_upper, color=color)
            # plt.plot(x_interp, y_upper_b_sp, color=color,label=f'{counter} : {filename}')

            # plt.scatter(x_lower, y_lower, s=5,color=color)
            # # plt.plot(x_lower, y_interp_lower, color=color)
            # plt.plot(x_interp, y_lower_b_sp, color=color)
            # plt.axis('equal')
            # plt.legend()
            # if counter % 5 == 0:
            #     plt.savefig('/home/marius_ruh/packages/lsdo_lab/lsdo_airfoil/lsdo_airfoil/core/parameterization/airfoil_images_downselect/' + f'{counter}.png')
            #     plt.clf()

plt.show()
# exit()

# with open('airfoil_interpolation_7.pickle', 'wb') as handle:
#     pickle.dump(airfoil_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print(np.mean(b_sp_interp_error_upper))
# print(np.mean(b_sp_interp_error_lower))


# plt.show()