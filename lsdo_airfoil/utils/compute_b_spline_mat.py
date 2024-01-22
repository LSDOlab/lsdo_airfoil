import numpy as np
import scipy
import time


def fit_b_spline(points, num_control_points, b_spline_order, u=None, control_points=None, alpha = 0):
    """
    Function to solve the B-spline fitting problem given. 
    L-2 regularization can be added if alpha is greater than 0

    Parameters
    ----------
    points : np.ndarray
        the data to be fitted 

    num_control_points : int
        number of b-spline control points

    b_spline_order : int
        order of the b-spline

    alpha : float (default = 0)
        regularization parameter for L-2 regularization

    """

    if u is not None and control_points is not None:
        raise ValueError("Cannot specify 'u' and 'control_points at the same time. If 'u' is provided a least squares problem is solved to find control points. If 'control_points' is specified, and optimization problem is solved to find parametric coordinates 'u'.")

    points_shape = points.shape
    if not isinstance(points, np.ndarray):
        raise ValueError('points needs to be numpy array')
    elif len(points_shape) > 2:
        raise ValueError(f'points has to be 1-D vector, received {len(points_shape)}-dimensional array')
    elif (len(points_shape) == 2) and (not any(x == 1 for x in points_shape)):
        raise ValueError(f'points has to be 1-D vector, received {len(points_shape)}-dimensional array')
    else:
        dim = max(points_shape)
    
    if control_points is not None:
        if len(control_points) != num_control_points:
            raise ValueError("Length of control point vector and num_control_points not equal")
        from caddee.core.primitives.bsplines.bspline_curve import BSplineCurve
        test_spline = BSplineCurve(
            name='test',
            control_points=control_points,
            order_u=b_spline_order,
        )
        u = test_spline.project(points, grid_search_n=50,  return_parametric_coordinates=True, max_iter=0)[0]
        B_sparse = get_bspline_mtx(num_control_points, dim, order=b_spline_order, u=u)
        return u, B_sparse


    else:
        sparse_points = scipy.sparse.csr_matrix(points).T
        B_sparse = get_bspline_mtx(num_control_points, dim, order=b_spline_order, u=u)
        B_dense = B_sparse.toarray()
        B = B_sparse
        B_shape = B_dense.shape
        rows = B_shape[0]
        cols = B_shape[1]
        if alpha < 0:
            raise ValueError("Regularization parameters alpha has to be a greater than 0")
        elif alpha == 0:
            A = B.T @ B
            b = B.T @ sparse_points
            cpts = scipy.sparse.linalg.spsolve(A, b)
            # B_star = np.dot(scipy.sparse.linalg.inv(np.dot(B.T,B)), B.T)
        else:
            # B_star = np.dot(scipy.sparse.linalg.inv(np.matmul(B.T, B) + alpha * scipy.sparse.csr_matrix(np.eye(cols))), B.T)
            A = B.T @ B + alpha * scipy.sparse.csr_matrix(np.eye(cols))
            b = B.T @ sparse_points
            cpts = scipy.sparse.linalg.spsolve(A, b)
            # control_points = np.dot(B_star, points)

        return  cpts, B

def get_bspline_mtx(num_cp, num_pt, order=4, u=None):
    order = min(order, num_cp)

    knots = np.zeros(num_cp + order)
    knots[order-1:num_cp+1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0

    if u is not None:
        if len(u) != num_pt:
            print(len(u))
            print(num_pt)
            raise ValueError('u and num_pt are not the same')
        else:
            t_vec = u
    else:
        t_vec = np.linspace(0, 1, num_pt)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order
        
        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.
        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return scipy.sparse.csr_matrix(
        (data, (rows, cols)), 
        shape=(num_pt, num_cp),
    )



import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
if __name__ == "__main__":
    x = np.linspace(0+1e-3, np.pi-1e-3, 20)
    
    data = x**3 #np.sin(x)


    c, B = fit_b_spline(data, num_control_points=5, b_spline_order=4, alpha=1e-11)
    print(type(c))
    print(c)
    exit()
    # control_points, x = fit_b_spline(data, 100, 3, 0)
    # print(x.shape)
    # print(B.shape)
    from caddee.core.primitives.bsplines.bspline_curve import BSplineCurve
    test_spline = BSplineCurve(
        name='test',
        control_points=c,
        order_u=3,
    )

    u = test_spline.project(data, grid_search_n=200,  return_parametric_coordinates=True, max_iter=100)


    p = B @ c

    print(u)
    # import matplotlib.pyplot as plt
    error = abs((p - data) / data) * 100
    error2 = abs((u.value - data) / data) * 100
    # print(max(error))
    print(error)
    print(error2)
    exit()
    print('\n')
    print(u)
    # plt.plot(x, error)
    # plt.ylim([0, 5])
    plt.plot(x, data)
    plt.plot(x, u.value)
    plt.show()