#
# Auxiliary functions for the project
#

import warnings
import numpy as np


def get_curing_time(solution, threshold=0.95):
    alpha = solution["Degree of polimerisation"].entries
    t = solution["Time [s]"].entries
    N = np.shape(alpha)[0]
    t_c = np.zeros(N)
    for i in range(N):
        a = alpha[i, :]
        if np.max(a) < threshold:
            warnings.warn(
                "The maximum degree of polimerisation in the solution is smaller than the threshold",
                UserWarning,
            )
            t_c[i] = np.nan
        else:
            t_c[i] = np.interp(threshold, a, t)
    return t_c
