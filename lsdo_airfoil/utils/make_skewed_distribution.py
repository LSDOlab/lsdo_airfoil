import numpy as np

def make_skewed_normalized_distribution(num_points, half_cos=False, power=False):
    if half_cos:
        # Half cosine distribution
        N = num_points 
        i_vec = np.arange(0, N)
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