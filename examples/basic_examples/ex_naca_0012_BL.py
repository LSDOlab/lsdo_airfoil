import matplotlib.pyplot as plt
import csdl_alpha as csdl
import numpy as np
from lsdo_airfoil.core.three_d_airfoil_aero_model import ThreeDAirfoilMLModelMaker
from lsdo_airfoil import UIUC_AIRFOILS
import random


recorder = csdl.Recorder(inline=True)
recorder.start()

naca_0012_airfoil_maker = ThreeDAirfoilMLModelMaker(
    airfoil_name="naca0012",
        aoa_range=np.linspace(-6, 8, 15), 
        reynolds_range=[1e5, 2e5, 5e5, 7e5, 1e6, 2e6], 
        mach_range=[0., 0.1, 0.2, 0.3, 0.4, 0.5],
        use_x_from_data=True,
)

x = naca_0012_airfoil_maker.x_interp


alpha = csdl.Variable(value=np.deg2rad(0))
Re = csdl.Variable(value=1.5e6)
Ma = csdl.Variable(value=0.166431)

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))

Cp_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["Cp"],
)
Cp_pred = Cp_model.evaluate(alpha, Re, Ma)
Cp_data = np.loadtxt("Cp.dat", skiprows=2)
axs[0, 0].plot(x, Cp_pred.flatten().value, label="ML prediction")
axs[0, 0].plot(x, Cp_data[:, 2], label="XFOIL data")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("Cp")
axs[0, 0].legend()


Ue_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["Ue"],
)

Ue_pred = Ue_model.evaluate(alpha, Re, Ma)
Ue_data = np.loadtxt("BL.dat", skiprows=1)
axs[0, 1].plot(x, Ue_pred.flatten().value, label="ML prediction")
axs[0, 1].plot(x, np.abs(Ue_data[0:250, 3]), label="XFOIL data")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("Ue")
axs[0, 1].legend()


Cf_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["Cf"], tune_hyper_parameters=True,
)

Cf_pred = Cf_model.evaluate(alpha, Re, Ma)
Cf_data = np.loadtxt("BL.dat", skiprows=1)
axs[0, 2].plot(x, Cf_pred.flatten().value, label="ML prediction")
axs[0, 2].plot(x, Cf_data[0:250, 6],  label="XFOIL data")
axs[0, 2].set_xlabel("x")
axs[0, 2].set_ylabel("Cf")
axs[0, 2].legend()



theta_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["theta"], tune_hyper_parameters=True, 
)

theta_pred = theta_model.evaluate(alpha, Re, Ma)
theta_data = np.loadtxt("BL.dat", skiprows=1)
axs[1, 0].plot(x, theta_pred.flatten().value, label="ML prediction")
axs[1, 0].plot(x, theta_data[0:250, 5], label="XFOIL data")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("theta")
axs[1, 0].legend()


delta_star_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["delta_star"], tune_hyper_parameters=True,
)

delta_star_pred = delta_star_model.evaluate(alpha, Re, Ma)
delta_star_data = np.loadtxt("BL.dat", skiprows=1)
axs[1, 1].plot(x, delta_star_pred.flatten().value, label="ML prediction")
axs[1, 1].plot(x, delta_star_data[0:250, 4], label="XFOIL data")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("delta star")
axs[1, 1].legend()

shape_parameter_model = naca_0012_airfoil_maker.get_airfoil_model(
    quantities=["shape_parameter"], tune_hyper_parameters=True,
)

shape_parameter_pred = shape_parameter_model.evaluate(alpha, Re, Ma)
shape_parameter_data = np.loadtxt("BL.dat", skiprows=1)
axs[1, 2].plot(x, shape_parameter_pred.flatten().value, label="ML prediction")
axs[1, 2].plot(x, shape_parameter_data[0:250, -1],  label="XFOIL data")
axs[1, 2].set_xlabel("x")
axs[1, 2].set_ylabel("H")
axs[1, 2].legend()

plt.tight_layout()
plt.show()

