import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from smt.surrogate_models import RMTB


t = np.arange(0, 1.1, .1)
x = np.sin(2*np.pi*t)
y = np.cos(2*np.pi*t)
tck, u = interpolate.splprep([x, y], s=0)
unew = np.arange(0, 1.01, 0.01)
out = interpolate.splev(unew, tck)
print(out[0]-unew)
plt.figure()
plt.plot(x, y, 'x', out[0], out[1], np.sin(2*np.pi*unew), np.cos(2*np.pi*unew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-1.05, 1.05, -1.05, 1.05])
plt.title('Spline of parametrically-defined curve')
plt.show()
exit()

# xt = np.array([0.0, 0.5/4, 1.0/4, 1.8/4, 4.0/4])
xt = np.array([1., 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 
               0.1, 0.075, 0.05, 0.025, 0.0125, 0.]) 

xt = xt * (2) - 1

# yt = np.array([0., 1.0, 1.5, 0.9, 1.0])
yt = np.array([-0.0093,  0.0052,  0.014,   0.0318,  0.046,   0.0561,  0.0628,  0.0673,  0.0684,
  0.0642,  0.0592,  0.0518,  0.0457, 0.0377,  0.0265,  0.0182,  0.    ]) 

yt = yt * (2) - 1

xlimits = np.array([[-1., 1.]])

sm = RMTB(
    xlimits=xlimits,
    order=4,
    num_ctrl_pts=int(5 * len(xt)),
    energy_weight=1e-15,
    regularization_weight=0,
)
sm.set_training_values(xt, yt)
sm.train()

num = 200
# x1 = np.linspace(0.0, 1.0, num)**0.5
# x2 = np.linspace(0.0, 1.0, num)**2
i_vec = np.arange(0, num)
x3 = 1 - 2 * np.cos(np.pi /(2 * len(i_vec) -2) * i_vec)
# x3 = np.linspace(-1, 1.0, num)

# y1 = sm.predict_values(x1)
# y2 = sm.predict_values(x2)
y3 = sm.predict_values(x3)

y_interp = sm.predict_values(xt).flatten()

# print('yt', yt)
# print('y_interp', y_interp)

print('interpolation error', (y_interp - yt) / yt * 100)

plt.plot(xt, yt, "o")
plt.plot(xt, y_interp, "*")
# plt.plot(x1, y1)
# plt.plot(x2, y2)
plt.plot(x3, y3)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Training data", "Prediction"])
plt.show()

exit()

x_coordinates = np.linspace(0, 9, 100)
y_coordinates = np.cos(x_coordinates)

min_val = np.min(np.concatenate([x_coordinates, y_coordinates]))
max_val = np.max(np.concatenate([x_coordinates, y_coordinates]))

scaled_x = (x_coordinates - min_val) / (max_val - min_val)
scaled_y = (y_coordinates - min_val) / (max_val - min_val)


plt.plot(x_coordinates, y_coordinates)
plt.plot(scaled_x, scaled_y)

plt.show()
exit()
def basis_fun_and_derivatives(x, i, p, knots, counter=0):
    counter = counter
    if p == 0:
        if knots[i] <= x < knots[i+1]:
            return 1, 0, 0  # Basis function value, first derivative, and second derivative
        else:
            return 0, 0, 0
    else:
        val1, dval1, ddval1 = 0, 0, 0
        val2, dval2, ddval2 = 0, 0, 0
        
        if knots[i+p] - knots[i] != 0:
            val1, dval1, ddval1 = basis_fun_and_derivatives(x, i, p-1, knots=knots)
            val1 *= (x - knots[i]) / (knots[i+p] - knots[i])
            dval1 *= 1 / (knots[i+p] - knots[i])
            ddval1 *= 0  # Second derivative of a zero degree basis function is always 0
        
        if knots[i+p+1] - knots[i+1] != 0:
            val2, dval2, ddval2 = basis_fun_and_derivatives(x, i+1, p-1, knots=knots)
            val2 *= (knots[i+p+1] - x) / (knots[i+p+1]- knots[i+1])
            dval2 *= -1 / (knots[i+p+1] - knots[i+1])
            ddval2 *= 0  # Second derivative of a zero degree basis function is always 0
        
        val = val1 + val2
        dval = dval1 + dval2
        ddval = ddval1 + ddval2
        
        return val, dval, ddval

knots = np.array([0., 0., 0., 0.5, 1., 1., 1.])
x = np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.])

for xi in x:
    result, first_derivative, second_derivative = basis_fun_and_derivatives(xi, 1, 3, knots=knots)
    print(f"basis_fun({xi}) = {result}, first_derivative = {first_derivative}, second_derivative = {second_derivative}")

import numpy as np

def basis_fun_and_derivatives(x, i, p, knots, counter=0):
    counter = counter
    if p == 0:
        if knots[i] <= x < knots[i+1]:
            return 1, 0, 0  # Basis function value, first derivative, and second derivative
        else:
            return 0, 0, 0
    else:
        val1, dval1, ddval1 = basis_fun_and_derivatives(x, i, p-1, knots=knots)
        val2, dval2, ddval2 = basis_fun_and_derivatives(x, i+1, p-1, knots=knots)
        
        delta1 = knots[i+p] - knots[i]
        delta2 = knots[i+p+1] - knots[i+1]
        
        alpha1 = 0 if delta1 == 0 else p / delta1
        alpha2 = 0 if delta2 == 0 else p / delta2
        
        val = alpha1 * val1 * (x - knots[i]) + alpha2 * val2 * (knots[i+p+1] - x)
        dval = alpha1 * (val1 * (p / delta1) + dval1 * (x - knots[i])) + \
                alpha2 * (val2 * (-p / delta2) + dval2 * (knots[i+p+1] - x))
        ddval = alpha1 * (dval1 * (p / delta1) + ddval1 * (x - knots[i])) + \
                alpha2 * (dval2 * (-p / delta2) + ddval2 * (knots[i+p+1] - x))
        
        return val, dval, ddval

knots = np.array([0., 0., 0., 0.5, 1., 1., 1.])
x = np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.])

for xi in x:
    result, first_derivative, second_derivative = basis_fun_and_derivatives(xi, 1, 2, knots=knots)
    print(f"basis_fun({xi}) = {result}, first_derivative = {first_derivative}, second_derivative = {second_derivative}")
