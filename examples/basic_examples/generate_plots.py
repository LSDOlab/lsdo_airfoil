import matplotlib.pyplot as plt
import csdl_alpha as csdl
import numpy as np
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from lsdo_airfoil import UIUC_AIRFOILS
import random


recorder = csdl.Recorder(inline=True)
recorder.start()

nasa_langley_airfoil_maker = ThreeDAirfoilMLModelMaker(
    airfoil_name="ls417",
        aoa_range=np.linspace(-12, 16, 50), 
        reynolds_range=[1e5, 2e5, 5e5, 1e6, 2e6, 4e6, 7e6, 10e6], 
        mach_range=[0., 0.2, 0.3, 0.4, 0.5, 0.6],
)
x_interp = nasa_langley_airfoil_maker.x_interp


alpha_Cl_min_max_model = nasa_langley_airfoil_maker.get_airfoil_model(
    quantities=["alpha_Cl_min_max"],
)

coefficient_model = nasa_langley_airfoil_maker.get_airfoil_model(
    quantities=["Cl", "Cd"]
)

cp_model = nasa_langley_airfoil_maker.get_airfoil_model(
    quantities=["Cp"], force_retrain=False
)

num_points = 1

# alpha = csdl.Variable(shape=(num_points, ), value=np.deg2rad(np.linspace(-5, 13, num_points)))
alpha = csdl.Variable(shape=(num_points, ), value=np.deg2rad(2))
Re = csdl.Variable(shape=(num_points, ), value=5e6)
Ma = csdl.Variable(shape=(num_points, ), value=0.18)


Cl, Cd = coefficient_model.evaluate(alpha, Re, Ma)
Cp = cp_model.evaluate(alpha, Re, Ma)
alpha_min_max = alpha_Cl_min_max_model.evaluate(alpha, Re, Ma)


plt.figure(1)
plt.plot(np.rad2deg(alpha.value), Cl.value)
plt.xlabel("alpha")
plt.ylabel("Cl")
plt.grid()

plt.figure(2)
plt.plot(np.rad2deg(alpha.value), Cd.value)
plt.xlabel("alpha")
plt.ylabel("Cd")
plt.grid()


plt.figure(3)
plt.xlabel("Cd")
plt.ylabel("Cl")
plt.plot(Cd.value, Cl.value)
plt.grid()


plt.figure(4)
plt.xlabel("chord")
plt.ylabel("Cp")
# for i in range(num_points):
color = (random.random(), random.random(), random.random())
# plt.plot(x_interp, Cp.value[i, 0:len(x_interp)], color=color)
# plt.plot(x_interp, Cp.value[i, len(x_interp):], color=color)

plt.plot(x_interp, Cp.value[0, 0:len(x_interp)], color=color)
plt.plot(x_interp, Cp.value[0, len(x_interp):], color=color)




alpha_1 = csdl.Variable(shape=(1, ), value=np.deg2rad(-1.5))
Re_1 = csdl.Variable(shape=(1, ), value=1.8e6)
Ma_1 = csdl.Variable(shape=(1, ), value=0.18)
test_cp_data_1 = np.loadtxt(f"{UIUC_AIRFOILS}/ls417/test_cp_data_aoa_m15_Re_18e6_ma_023.txt")
test_cp_pred_1 = cp_model.evaluate(alpha_1, Re_1, Ma_1)

alpha_2 = csdl.Variable(shape=(1, ), value=np.deg2rad(4.5))
Re_2 = csdl.Variable(shape=(1, ), value=6.8e6)
Ma_2 = csdl.Variable(shape=(1, ), value=0.45)
test_cp_data_2 = np.loadtxt(f"{UIUC_AIRFOILS}/ls417/test_cp_data_aoa_p45_Re_86e6_ma_045.txt")
test_cp_pred_2 = cp_model.evaluate(alpha_2, Re_2, Ma_2)

alpha_3 = csdl.Variable(shape=(1, ), value=np.deg2rad(8))
Re_3 = csdl.Variable(shape=(1, ), value=5e5)
Ma_3 = csdl.Variable(shape=(1, ), value=0.33)
test_cp_data_3 = np.loadtxt(f"{UIUC_AIRFOILS}/ls417/test_cp_data_aoa_p8_Re_5e5_ma_033.txt")
test_cp_pred_3 = cp_model.evaluate(alpha_3, Re_3, Ma_3)

plt.figure(5)
plt.plot(x_interp, test_cp_data_1[0:len(x_interp)], linestyle=":", color='k', label="test data")
plt.plot(x_interp, test_cp_data_1[len(x_interp):], linestyle=":", color='k')
plt.plot(x_interp, test_cp_pred_1.value[0, 0:len(x_interp)], color='r', label=f"AoA={np.rad2deg(alpha_1.value)}, Re={Re_1.value}, Ma={Ma_1.value}")
plt.plot(x_interp, test_cp_pred_1.value[0, len(x_interp):], color='r')


plt.plot(x_interp, test_cp_pred_2.value[0, 0:len(x_interp)], color='b', label=f"AoA={np.rad2deg(alpha_2.value)}, Re={Re_2.value}, Ma={Ma_2.value}")
plt.plot(x_interp, test_cp_pred_2.value[0, len(x_interp):], color='b')
plt.plot(x_interp, test_cp_data_2[0:len(x_interp)], linestyle=":", color='k')
plt.plot(x_interp, test_cp_data_2[len(x_interp):], linestyle=":", color='k')

plt.plot(x_interp, test_cp_pred_3.value[0, 0:len(x_interp)], color='g', label=f"AoA={np.rad2deg(alpha_3.value)}, Re={Re_3.value}, Ma={Ma_3.value}")
plt.plot(x_interp, test_cp_pred_3.value[0, len(x_interp):], color='g')
plt.plot(x_interp, test_cp_data_3[0:len(x_interp)], linestyle=":", color='k')
plt.plot(x_interp, test_cp_data_3[len(x_interp):], linestyle=":", color='k')

plt.ylabel("Cp")
plt.xlabel("Normalized chord")
plt.legend()

plt.show()

