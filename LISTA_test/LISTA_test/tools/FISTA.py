import numpy as np
import numpy.linalg as la
import tools.shrinkage as shrinkage

def fista(A, b, l, maxit):
    x = np.zeros((A.shape[1],b.shape[1]))
    t = 1.
    z = x
    L = la.norm(A, 2) ** 2
    for _ in range(maxit):
        xold = x
        z = z + A.T.dot(b - A.dot(z)) / L
        x = shrinkage.simple_soft_threshold_np(z, l / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)

    return x