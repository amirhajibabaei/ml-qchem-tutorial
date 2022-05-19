# +
import numpy as np


def dot_product_design_matrix(inducing, descriptor_data, eta):
    """ """
    kern_sum = 0.0
    N = len(descriptor_data)
    jac_kern_sum = np.zeros((N, 3))
    for index, (neighbors, descriptor, jacobian) in enumerate(descriptor_data):
        kern, jac_kern = dot_product_kernel(inducing, descriptor, jacobian, eta)
        kern_sum += kern
        jac_kern_sum[index] -= jac_kern.sum(axis=0)
        jac_kern_sum[neighbors] += jac_kern
    return kern_sum, jac_kern_sum


def dot_product_kernel(inducing, descriptor, jacobian, eta):
    """ """
    prod = (inducing * descriptor).sum()
    jac_prod = (inducing[..., None, None] * jacobian).sum(axis=0)
    kern = prod**eta
    jac_kern = eta * prod ** (eta - 1) * jac_prod
    return kern, jac_kern
