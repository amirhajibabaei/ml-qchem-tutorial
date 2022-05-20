# +
"""
Implemets the dot-product kernel
    kern(x, y) = (x.y)**eta
where eta is a hyper-parameter.

"""
import numpy as np


def get_design_matrix(inducing, descriptor_data, eta):
    """
    Args:
        inducing           an inducing descriptor vector
        descriptor_data    descriptor data (neighbors, descriptors,
                           and jacobians) for an atomic configuration
        eta                exponent of the kernel (hyper-parameter)

    Returns:
        p    sum of kernel evaluations between "inducing" descriptor
             and descriptors of the atomic configuration
        q    gradienf of p wrt atomic positions

    """
    kern_sum = 0.0
    N = len(descriptor_data)
    grad_kern_sum = np.zeros((N, 3))
    for index, (neighbors, descriptor, jacobian) in enumerate(descriptor_data):
        kern, grad_kern = dot_product_kernel(inducing, descriptor, jacobian, eta)
        kern_sum += kern
        grad_kern_sum[index] -= grad_kern.sum(axis=0)
        grad_kern_sum[neighbors] += grad_kern
    return kern_sum, grad_kern_sum


def dot_product_kernel(inducing, descriptor, jacobian, eta):
    """
    inducing     an inducing descriptor vector
    descriptor   a genral descriptor vector
    jacobian     Jacobian of the descriptor vector
    eta          exponent of the kernel function


    Returns: p, q
        p = (sum_i x[i]*y[i]) ** eta
        q = grad of p

    """
    prod = (inducing * descriptor).sum()
    grad_prod = (inducing[..., None, None] * jacobian).sum(axis=0)
    kern = prod**eta
    grad_kern = eta * prod ** (eta - 1) * grad_prod
    return kern, grad_kern
