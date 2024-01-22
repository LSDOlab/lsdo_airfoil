
from python_csdl_backend import Simulator
from lsdo_airfoil.core.airfoil_model_csdl import AirfoilModelCSDL, ClModelCSDL, CdModelCSDL, LiftDragCoefficientsCSDL
import csdl
import numpy as np
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem


num_nodes = 1
M = 0.3
AoA = 3 # np.linspace(-60, 60, num_nodes) 
Re =  1e6

run_model = csdl.Model()
run_model.create_input('mach_number', M * np.ones((num_nodes, )))
run_model.create_input('angle_of_attack', AoA) #  * np.ones((num_nodes, )))
run_model.create_input('reynolds_number', Re * np.ones((num_nodes, )))

# run_model.add_design_variable('mach_number', lower=0., upper=0.6)
# run_model.add_design_variable('angle_of_attack', scaler=3e-2, lower=2, upper=60)
# run_model.add_design_variable('reynolds_number', scaler=0.5e-6, lower=1e5, upper=1e6)



# airfoil_model_1 = AirfoilModelCSDL(
#     num_nodes=num_nodes,
#     compute_control_points=False,
#     airfoil_name='NACA_4412',
# )

# run_model.add(airfoil_model_1, 'airfoil_model_1')


lift_drag = LiftDragCoefficientsCSDL(
    num_nodes=num_nodes,
    airfoil_name='NACA_4412',
    compute_control_points=False,

)

run_model.add(lift_drag, 'lift_drag')


Cl = run_model.declare_variable('Cl', shape=(num_nodes, ))
Cd = run_model.declare_variable('Cd', shape=(num_nodes, ))

# run_model.register_output('LoD', Cl/Cd)

# run_model.add_objective('LoD', scaler=-1e-2)
# run_model.add_design_variable('control_points', lower=-0.5, upper=0.5)
run_model.add_constraint('Cl', equals=0.5)
run_model.add_objective('Cd', scaler=50)

sim = Simulator(run_model)
sim.run()



import matplotlib.pyplot as plt

# cl = sim['Cl'].flatten()
# cd = sim['Cd'].flatten()

# fig, axs = plt.subplots(2, 2, layout='constrained', figsize=(10, 8))
# axs[0, 0].plot(AoA, cl)
# axs[0, 0].set_xlabel('angle of attack')
# axs[0, 0].set_ylabel('Cl')
# axs[0, 0].grid()

# axs[0, 1].plot(AoA, cd)
# axs[0, 1].set_xlabel('angle of attack')
# axs[0, 1].set_ylabel('Cd')
# axs[0, 1].grid()

# axs[1, 0].plot(AoA, cl/cd)
# axs[1, 0].set_xlabel('angle of attack')
# axs[1, 0].set_ylabel('Cl/Cd')
# axs[1, 0].grid()

# axs[1, 1].plot(cd, cl)
# axs[1, 1].set_xlabel('Cd')
# axs[1, 1].set_ylabel('Cl')
# axs[1, 1].grid()

# fig.tight_layout()
# plt.show()

# sim.check_totals(of='LoD', wrt='angle_of_attack', step=1e-3)

prob = CSDLProblem(problem_name='airfoil_ml_test', simulator=sim)
optimizer = SLSQP(
    prob,
    maxiter=100, 
    ftol=1e-12,
)
optimizer.solve()
optimizer.print_results()

print(sim['Cd'].flatten())
print(sim['Cl'].flatten())
print(sim['mach_number'])
print(sim['angle_of_attack'])
print(sim['reynolds_number'])

# print(sim['LoD'])

cpts = sim['control_points'].flatten()
print(cpts)

camber_cpts = np.zeros(18, )
thickness_cpts = np.zeros(18, )
camber_cpts[1:-1] = cpts[0:16]
thickness_cpts[1:-1] = cpts[16: ]

from lsdo_airfoil.utils.compute_b_spline_mat import compute_b_spline_mat

num_pts = 300
x_range = np.linspace(0, 1, num_pts)
i_vec = np.arange(0, len(x_range))
x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)

B = compute_b_spline_mat(num_pts=num_pts)[0]
camber = np.matmul(B, camber_cpts)
thickness = np.matmul(B, thickness_cpts)

upper = camber + 0.5 * thickness
lower = camber - 0.5 * thickness

plt.plot(x_interp, upper)
plt.plot(x_interp, lower)

plt.show()
exit()

# print('\n')
# control_points = sim['control_points'].flatten()
# cpts_camber = np.hstack(([0], control_points[0:16], [0]))
# cpts_thickness = np.hstack(([0], control_points[16:], [0]))


# B, B_star = compute_b_spline_mat(301)
# full_camber = np.matmul(B, cpts_camber)
# full_thickness = np.matmul(B, cpts_thickness)
# full_upper = full_camber + 0.5 * full_thickness
# full_lower = full_camber - 0.5 * full_thickness

# plt.plot(x_interp, full_upper, color='blue')
# plt.plot(x_interp, full_lower, color='blue')


# print(x_lower.shape)
# print(upper.shape)
# plt.scatter(x_upper, upper, color='red')
# plt.scatter(x_lower, lower, color='red')

# plt.show()
# exit()

# np.save('sample_airfoils/control_points/boeing_vertol_vr_12', sim['control_points'])
