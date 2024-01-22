import numpy as np
import scipy


def gaussian_elimination(augmented_matrix):
    # Convert the augmented matrix to a numpy array for easier manipulation
    augmented_matrix = np.array(augmented_matrix, dtype=float)
    
    num_rows, num_cols = augmented_matrix.shape
    for pivot_row in range(num_rows):
        # Find the pivot element in the current column
        pivot_col = pivot_row
        while pivot_col < num_cols - 1 and augmented_matrix[pivot_row][pivot_col] == 0:
            pivot_col += 1

        if pivot_col >= num_cols - 1:
            # If the pivot element is zero, continue to the next row
            continue

        # Swap the current row with the row containing the pivot element
        augmented_matrix[[pivot_row, pivot_col]] = augmented_matrix[[pivot_col, pivot_row]]

        # Scale the current row to make the pivot element 1
        pivot_value = augmented_matrix[pivot_row][pivot_col]
        augmented_matrix[pivot_row] /= pivot_value

        # Eliminate other rows
        for row in range(num_rows):
            if row != pivot_row:
                factor = augmented_matrix[row][pivot_col]
                augmented_matrix[row] -= factor * augmented_matrix[pivot_row]

    return augmented_matrix


A = np.array([
    [2, -4, -2, 0],
    [1, -1, 0, 2],
    [-2, 4, 2, -1],
    [-4, -4, 1, -3],
])

# p, l, u = scipy.linalg.lu(A, permute_l=True, overwrite_a=True)

# print(p)
# print(l)
# print(u)

# A = [    2, -4, -2, 0,    1, -1, 0, 2,    -2, 4, 2, -1,    -4, -4, 1, -3]

# # Step 1: Partial Pivoting (Swap rows 2 and 4)
# P2 = [    1, 0, 0, 0,    0, 0, 0, 1,    0, 0, 1, 0,    0, 1, 0, 0]

# PA2 = P2 * A = [    2, -4, -2, 0,    -4, -4, 1, -3,    -2, 4, 2, -1,    1, -1, 0, 2]

# # Step 2: LU Decomposition
# U2 = [    2, -4, -2, 0,    0, -3, 1.5, -3,    0, 0, 2.5, 0.5,    0, 0, 0, 2]

# L2 = [    1, 0, 0, 0,    -2, 1, 0, 0,    -1, -0.5, 1, 0,    0.5, -0.5, 0.2, 1]

# # So, the second PA = LU factorization is:
# P2 * A = L2 * U2

L = np.array([
    [1, 0, 0],
    [0.5, 1, 0],
    [0.2, 0.1, 1]
])

U = np.array([
    [3, 0.5, 0.2],
    [0, 5, 0.1],
    [0, 0, 2],
])

A = L @ U

print(A)

D = np.diag([3, 5, 2])

V = np.array([
    [1, 0.5/3, 0.2/3],
    [0, 1, 0.1/5],
    [0, 0, 1],
])

A2 = L @ D @ V

print(A2)

# # Example usage:
# augmented_matrix = [
#     [1, 2, 0,  1],
#     [1, -1, 1,  4],
#     [0, 3, -1, 0]
# ]

# A = np.array([
#     [1, 2, 0],
#     [1, -1, 1],
#     [0, 3, -1]
# ])


# # Assemble element wise global stiffness matrix
# def element_global(phi):
#     phi_rad = np.deg2rad(phi)
#     c = np.cos(phi_rad)
#     s = np.sin(phi_rad)

#     K_element = np.array([
#         [c**2, c*s, -c**2, -c*s],
#         [c*s, s**2, -c*s, -s**2],
#         [-c**2, -c*s, c**2, c*s],
#         [-c*s, -s**2, c*s, s**2],
#     ])
#     return K_element

# print(element_global(90))

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Define the coefficients of the plane equation
# A = 1
# B = 1
# C = -1
# D = 3

# # Create a grid of points in the x1, x2, and x3 dimensions
# x1 = np.linspace(-10, 10, 100)
# x2 = np.linspace(-10, 10, 100)
# x1, x2 = np.meshgrid(x1, x2)

# # Calculate x3 based on the plane equation
# x3 = A * x1 + B * x2 + D

# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Plot the plane
# ax.plot_surface(x1, x2, x3, alpha=0.5)

# # Label the axes
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('X3')

# plt.title('Plot of the Plane: x1 + x2 = x3 - 3')
# plt.show()

