from lsdo_airfoil.utils.compute_b_spline_mat import get_bspline_mtx, fit_b_spline
import numpy as np
import scipy


def eval_basis_fun(x, i, p, knots):
    if p == 0:
        # if  knots[i+1] == x  and x == knots[-1]:
        #     return 1
        if (knots[i] <= x) and (x < knots[i+1]):
            return 1
        else:
            return 0
    else:
        val1 = 0
        val2 = 0
        
        if knots[i+p] - knots[i] != 0:
            val1 = (x - knots[i]) / (knots[i+p] - knots[i]) * eval_basis_fun(x, i, p-1, knots=knots)
        
        if knots[i+p+1] - knots[i+1] != 0:
            val2 = (knots[i+p+1] - x) / (knots[i+p+1]- knots[i+1]) * eval_basis_fun(x, i+1, p-1, knots=knots)
        
        return val1 + val2


def eval_der_basis_fun(x, i, p, knots):
    # step = 1e-8
    # eval = eval_basis_fun(x, i=i, p=p, knots=knots)
    
    # if x == 1:
    #     eval_minus = eval_basis_fun(x - step, i=i, p=p, knots=knots)
    #     return (eval - eval_minus) / step

    # else:
    #     eval_plus = eval_basis_fun(x + step, i=i, p=p, knots=knots)
    #     return (eval_plus - eval) / step
    if p == 0:
        return 0
    else:
        val1 = 0
        val2 = 0

        if knots[i+p] - knots[i] != 0:
            val1 = p / (knots[i+p] - knots[i]) * eval_basis_fun(x, i, p-1, knots)
        # if knots[i+p] - knots[i] != 0:
        #     val1 = 1 / (knots[i+p] - knots[i]) * (eval_basis_fun(x, i, p, knots) + (x-knots[i])*eval_der_basis_fun(x, i, p-1, knots))

        # if knots[i+p+1] - knots[i+1] != 0:
        #     val2 = 1 / (knots[i+p+1]- knots[i+1]) * (eval_der_basis_fun(x, i+1, p-1, knots) * (knots[i+p+1] - x) - eval_basis_fun(x, i+1, p-1, knots))
        if knots[i+p+1] - knots[i+1] != 0:
            val2 = p / (knots[i+p+1]- knots[i+1]) * eval_basis_fun(x, i+1, p-1, knots)

        return val1 - val2


def eval_2nd_der_basis_fun(x, i, p, knots):
    # step = 1e-8
    # eval = eval_basis_fun(x, i=i, p=p, knots=knots)
    # if x == 1:
    #     eval_minus = eval_basis_fun(x - step, i=i, p=p, knots=knots)
    #     eval_minus_minus = eval_basis_fun(x - 2 * step, i=i, p=p, knots=knots)
    #     return (eval - 2*eval_minus + eval_minus_minus) / step**2
    # else:
    #     eval_plus = eval_basis_fun(x + step, i=i, p=p, knots=knots)
    #     eval_plus_plus = eval_basis_fun(x + 2 * step, i=i, p=p, knots=knots)
    #     return (eval_plus_plus - 2*eval_plus + eval) / step**2
    
    if p == 0:
        return 0
    else:
        val1 = 0
        val2 = 0

        if knots[i+p] - knots[i] != 0:
            val1 = 1 / (knots[i+p] - knots[i]) * (eval_der_basis_fun(x, i, p-1, knots) * (2 - knots[i]) + x * eval_2nd_der_basis_fun(x, i, p-1, knots))

        if knots[i+p+1] - knots[i+1] != 0:
            val2 = 1 / (knots[i+p+1]- knots[i+1]) * (eval_2nd_der_basis_fun(x, i+1, p-1, knots) * (knots[i+p+1] -x) - 2 * eval_der_basis_fun(x, i+1, p-1, knots))

        return val1 + val2



# knots = np.array([0., 0., 0., 0.5, 1., 1., 1.])
# x = np.array([0., 0.14285714, 0.28571429, 0.42857143, 0.57142857, 0.71428571, 0.85714286, 1.])
x=np.linspace(0, 1, 20)
if False:
    order = 4
    num_ctr_points = 10
    p = order - 1
    knots = np.zeros(num_ctr_points + order)
    knots[order-1:num_ctr_points+1] = np.linspace(0, 1, num_ctr_points - order + 2)
    knots[num_ctr_points+1:] = 1.0
    i=2
    for xi in x:
        result = eval_basis_fun(xi, i=i, p=p, knots=knots)
        step = 1e-5
        result_perturbed = eval_basis_fun(xi + step, i=i, p=p, knots=knots)
        resul_perturbed_negative = eval_basis_fun(xi - step, i=i, p=p, knots=knots)


        fd = (result_perturbed -result) / step
        fd_2 = (resul_perturbed_negative - 2*result + result_perturbed) / (step**2)
        
        d_result = eval_der_basis_fun(xi, i=i, p=p, knots=knots)
        d2_result = eval_2nd_der_basis_fun(xi, i=i, p=p, knots=knots)
        print(f"basis_fun({xi}) = {result} | basis_fun_der({xi}) = {d_result} | basis_fun_der_fd({xi}) = {fd}")
        print(f"basis_fun({xi}) = {result} | basis_fun 2nd_der({xi}) = {d2_result} | basis_fun_2nd_der_fd({xi}) = {fd_2}")
        print('\n')
# exit()
def construct_bspline_matrix(num_ctr_points, num_pts, order, u=None):
    b_spline_matrix = np.zeros((num_pts, num_ctr_points))

    p = order - 1
    knots = np.zeros(num_ctr_points + order)
    knots[order-1:num_ctr_points+1] = np.linspace(0, 1, num_ctr_points - order + 2)
    knots[num_ctr_points+1:] = 1.0

    if u is not None:
        x = u
    else:
        x = np.linspace(0, 1, num_pts)

    for i in range(num_pts):
        for j in range(num_ctr_points):
            b_spline_matrix[i, j] = eval_basis_fun(x[i], j, p=p, knots=knots)

    return scipy.sparse.csr_matrix(b_spline_matrix)
    # return b_spline_matrix

    B = get_bspline_mtx(num_ctr_points, 20, order=4)#, u=np.linspace(0.1, 1, 20))
    B2 = construct_bspline_matrix(num_ctr_points, 20, order=4)

# print(B.toarray()-B2.toarray())

# exit()
def eval_resdiual(u, p, knots, control_points, point):
    num_ctr_points = len(control_points)
    row = np.zeros((num_ctr_points, ))
    for j in range(num_ctr_points):
        row[j] = eval_basis_fun(u, i=j, p=p, knots=knots)
    res =  point - row @ control_points
    return res

def eval_2nd_der_resdiual(u, p, knots, control_points, point):
    # step = 1e-7
    # eval = eval_resdiual(x, p, knots, control_points, point)
    # if x == 1:
    #     eval_minus = eval_resdiual(x - step, p, knots, control_points, point)
    #     eval_minus_minus = eval_resdiual(x - 2 *  step, p, knots, control_points, point)
    #     return (eval - 2*eval_minus + eval_minus_minus) / step**2
    # else:
    #     eval_plus = eval_resdiual(x +step, p, knots, control_points, point)
    #     eval_plus_plus = eval_resdiual(x + 2 * step, p, knots, control_points, point)
    #     return (eval_plus_plus - 2*eval_plus + eval) / step**2
    num_ctr_points = len(control_points)
    row = np.zeros((num_ctr_points, ))
    for j in range(num_ctr_points):
        row[j] = eval_2nd_der_basis_fun(u, i=j, p=p, knots=knots)

    return -row @ control_points


def eval_der_basis_residual(u, p, knots, control_points, point):
    # step = 1e-7
    # eval = eval_resdiual(x, p, knots, control_points, point)
    
    # if x == 1:
    #     eval_minus = eval_resdiual(x - step, p, knots, control_points, point)
    #     return (eval - eval_minus) / step

    # else:
    #     eval_plus = eval_resdiual(x +step, p, knots, control_points, point)
    #     return (eval_plus - eval) / step
    num_ctr_points = len(control_points)
    row = np.zeros((num_ctr_points, ))
    for j in range(num_ctr_points):
        row[j] = eval_der_basis_fun(u, i=j, p=p, knots=knots)

    return -row @ control_points

# order = 4
# control_points = np.linspace(0, 1, 5)
# num_ctr_points = len(control_points)
# knots = np.zeros(num_ctr_points + order)
# knots[order-1:num_ctr_points+1] = np.linspace(0, 1, num_ctr_points - order + 2)
# knots[num_ctr_points+1:] = 1.0

# res = eval_resdiual(0.49, order-1, knots, control_points, 0.5)
# d_res_du = eval_der_basis_residual(0.49, order-1, knots, control_points, 0.5)
# d2_res_du2 = eval_2nd_der_resdiual(0.49, order-1, knots, control_points, 0.5)

# print(d_res_du)
# print(d2_res_du2)

# print(0.49 - 0.01* d_res_du / d2_res_du2)

# print('\n')
# u_0 = 0.099
# point = 0.1
# for i in range(100):
#     d_du = eval_der_basis_residual(u_0, order-1, knots, control_points, point)
#     d2_du2 = eval_2nd_der_resdiual(u_0, order-1, knots, control_points, point)
#     res = eval_resdiual(u_0, p, knots, control_points, point)
#     if abs(res) < 1e-12:
#         print(res)
#         break
#     u_k = u_0 - 0.01 * d_du / (d2_du2)
#     print(i,u_k, u_k - u_0, res)
#     u_0 = u_k
# exit()
# objective_function = lambda u :eval_resdiual(u, order-1, knots, control_points, 0.5)
# initial_guess = 0.49
# result = scipy.optimize.minimize(objective_function, initial_guess, bounds=(0.,0.6), method='BFGS', tol=1e-5, options={'maxiter' : 1000})

# optimized_u = result.x[0]

# Print the result
# print(result)
# print("Optimized u:", optimized_u)
# print("Optimized residual:", result.fun)

# exit()

def find_parametric(points, control_points, order, opt_iter=10):
    p = order-1
    num_ctr_points = len(control_points)
    knots = np.zeros(num_ctr_points + order)
    knots[order-1:num_ctr_points+1] = np.linspace(0, 1, num_ctr_points - order + 2)
    knots[num_ctr_points+1:] = 1.0

    u_vec = np.zeros((len(points)))
    # print(len(points))
    for ind in range(len(points)):
    # for ind in range(3):
        point = points[ind]
        for num_opt in range(opt_iter):
            if num_opt == 0:
                u = np.linspace(0, 1, 10)
            else:
                if ind == len(points) -1:
                # window = 10 * opt_iter
                    u = np.linspace(u1, u2, 100)
                else:
                    u = np.linspace(u1, u2, 10)
                    
            res_list = []
            for i in range(len(u)):
                row = np.zeros((num_ctr_points, ))

                for j in range(num_ctr_points):
                    row[j] = eval_basis_fun(u[i], i=j, p=p, knots=knots)
                
                res = abs(point - row @ control_points)
                res_list.append(res)
                # print(res)
            
            # if res < 1e-8:
            #     best_idx = np.array(res_list).argmin()
            #     break
            best_idx = np.array(res_list).argmin()
            if best_idx + 1 == len(res_list):
                u1 = u[best_idx-1]
                u2 = u[best_idx]
            else:
                u1 = u[best_idx]
                u2 = u[best_idx+1]
            # print(res_list)
        print('\n')
        print(ind, best_idx, u[best_idx], res_list[best_idx])
        u_0 = u[best_idx]

        if res_list[best_idx] > 1e-8:

            for i in range(30):
                d_du = eval_der_basis_residual(u_0, order-1, knots, control_points, point)
                d2_du2 = eval_2nd_der_resdiual(u_0, order-1, knots, control_points, point)
                res = eval_resdiual(u_0, p, knots, control_points, point)
                if abs(res) < 1e-12:
                    # print(res)
                    break
                
                if d_du == 0:
                    break
                
                u_k = u_0 - 1 * res / d_du # d_du / (d2_du2) # 
                print(i,u_k, u_k - u_0, res)
                u_0 = u_k
        # r0 = res
        # for iter in range(15):
        #     # dr_du
        #     dr_du_row_list = []
        #     d_2r_du2_row_list = []
        #     # for col in range(num_ctr_points):
        #     #     dr_du_row_list.append(eval_der_basis_fun(u_0, i=1, p=p, knots=knots))
        #     #     d_2r_du2_row_list.append(eval_2nd_der_basis_fun(u_0, i=1, p=p, knots=knots))
            
        #     # dr_du =  np.array(dr_du_row_list) @ control_points
        #     # d2r_du2 =  np.array(d_2r_du2_row_list) @ control_points

        #     dr_du = 0 
        #     d2r_du2 =  0
        #     for col in range(num_ctr_points):
        #         dr_du -= eval_der_basis_fun(u_0, i=col, p=p, knots=knots) * control_points[col]
        #         d2r_du2 -= eval_2nd_der_basis_fun(u_0, i=col, p=p, knots=knots) * control_points[col]**2

        #     if d2r_du2 == 0:
        #         d2r_du2 = 1
        #         dr_du = 0.0001
                

        #     u_k = u_0 - 1 * (dr_du/ d2r_du2)

        #     if abs(u_k -u_0) < 1e-8:
        #         break
        #     # print(ind, iter, u_k - u_0, dr_du, d2r_du2)
        #     u_0 = u_k
                # break

            # u_0 = u_k
            # else:
            #     print(ind, iter, u_k - u_0, dr_du, d2r_du2)
            #     # print(ind, iter, u_k - u_0)
            #     u_0 = u_k

            # print(eval_der_basis_fun(u_0, i=1, p=p, knots=knots))
            # print(eval_2nd_der_basis_fun(u_0, i=1, p=p, knots=knots))
            # u_k = u_0 - eval_der_basis_fun(u_0, i=1, p=p, knots=knots) / eval_2nd_der_basis_fun(u_0, i=1, p=p, knots=knots)
            # print(iter, u_0, u_k - u_0, dr_du, d2r_du2)
            

        u_vec[ind] = u_0
    
    return u_vec
    
# num_cpts = 100
# num_pts = 10
# x_range = np.linspace(0, 1, num_cpts)
# i_vec = np.arange(0, len(x_range))
# x_interp = 1 - np.cos(np.pi/(2 * len(x_range)) * i_vec)

# # control_points = np.array([1.01624313e-09, 1.57073419e-06, 4.93323152e-03, 1.54932703e+01, 3.09766773e+01])

# control_points = np.linspace(0, 1, num_cpts)

# x = np.linspace(0, np.pi-1e-3, 20)
# data = np.sin(x) #x**3
# data = np.linspace(0, 1, 400)
# u_vec = find_parametric(data, control_points, 4, 5)
# # if u_vec[-1] != 1:
# #     u_vec[-1] = 1
# # u_vec[-1] = 1-1e-7


# # B = get_bspline_mtx(num_cpts, 20, order=4, u=u_vec)#, u=np.linspace(0.1, 1, 20))

# B = construct_bspline_matrix(num_cpts, 400, order=4, u=u_vec)
# # print(B)

# # print(B2.toarray())

# points = B.toarray() @ control_points


# print((data-points)/data *100)

# print(points)
# print(data)
# print(u_vec)
# exit()


def bisection(f, a, b, tol=1e-6, max_iterations=100):
    """
    Find the root of a function using the bisection method.

    Parameters:
    - f: The function for which to find the root.
    - a: The left endpoint of the interval.
    - b: The right endpoint of the interval.
    - tol: The tolerance (stop when |f(x)| < tol).
    - max_iterations: Maximum number of iterations.

    Returns:
    - The approximate root of the function.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("The function values at 'a' and 'b' must have opposite signs.")

    for iteration in range(max_iterations):
        # Compute the midpoint of the interval
        c = (a + b) / 2

        # Check if the root has been found within the tolerance
        if abs(f(c)) < tol:
            return c

        # Update the interval
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    raise ValueError("Bisection method did not converge within the maximum number of iterations.")

# Example usage:
def quadratic_function(x):
    return x**2 - 4

# root = bisection(quadratic_function, 0, 3)
# print("Approximate root:", root)
