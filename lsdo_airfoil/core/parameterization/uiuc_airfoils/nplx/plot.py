import matplotlib.pyplot as plt 
import numpy as np
import os
from lsdo_airfoil import UIUC_AIRFOILS_2


os.chdir(UIUC_AIRFOILS_2)
polar_data_2 = np.loadtxt("nplx/polar_file_2.txt", skiprows=12)
polar_data = np.loadtxt("nplx/polar_file.txt", skiprows=12)

plt.plot(polar_data[:, 0], polar_data[:, 1])
plt.plot(polar_data_2[:, 0], polar_data_2[:, 1])
plt.show()