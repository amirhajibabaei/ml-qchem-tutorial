# +
"""
Introduction:
    In kernel regression, the potential energy of
    an atomic configuration is defined by
        energy = sum_i sum_j { kern(x[i], y[j]) * w[j] } + mean_energy
    where
        {x[i]} are the local descriptors for the atomic configuration
        {y[j]} are a set of inducing descriptors
        {w[j]} are the weights for the inducing descriptors
    Therefore a model is defined by {y[j]}, {w[j]}, and mean_energy.

Inducing descriptors:
    The inducing descriptors are sampled from the atomic configurations.
    Once the descriptors for all the training configurations are calculated,
    a subset of them are selected as inducing descriptors.
    Here, for simplicity, we choose such a subset randomly.


Regression:
    The weights W={w[j]} should be calculated such that the potential energy
    and forces for the training configurations are reproduced.
    Here, again for simplicity, we convert the problem into a
    least-squares problem:
        W = argmin || K @ W -Y ||**2
    where K is the design matrix and Y collects potential energy and forces
    of all training configurations.

"""

import numpy as np
from descriptor import get_descriptor_data
from kernel import get_design_matrix


def regression(training, cutoff, alpha, beta, eta, max_inducing):
    """
    Args:
        training      a set of ase.Atoms objects
        cutoff        cutoff radius
        alpha, beta   descriptor hyper-parameters (see descriptor.py)
        eta           kernel hyper-parameters (see kernel.py)
        max_inducing  maximum number of inducing points

    Returns:
        inducing      inducing descriptors for the model
        weights       weights of the inducing descriptors
        mean_energy   mean energy which should be manually added in predictions
        energy_mae    training mean absolute error for energies
        forces_mae    training mean absolute error for forces

    """

    # Parse training data:
    targets = []
    descriptor_data = []
    inducing = []
    for atoms in training:
        data = get_descriptor_data(atoms, cutoff, alpha, beta)
        descriptor_data.append(data)
        for _, descriptor, _ in data:
            inducing.append(descriptor)
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        targets.append((e, f.reshape(-1)))
    energies, forces = zip(*targets)
    a = np.array(energies)
    mean_energy = a.mean()
    a -= mean_energy
    b = np.concatenate(forces)
    Y = np.concatenate([a, b])

    # Select inducing descriptors
    if len(inducing) > max_inducing:
        perm = np.random.permutation(len(inducing))
        random_selection = perm[:max_inducing]
        inducing = [inducing[k] for k in random_selection]

    # Calculate the design matrix:
    columns = []
    for x in inducing:
        column = []
        for data in descriptor_data:
            kern, grad = get_design_matrix(x, data, eta)
            column.append((kern, grad.reshape(-1)))
        kern, grad = zip(*column)
        a = np.array(kern)
        b = np.concatenate(grad)
        c = np.concatenate([a, -b])
        columns.append(c)

    # Solve A w = Y:
    A = np.stack(columns, axis=1)
    weights, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    # Training errors:
    delta_Y = A @ weights - Y
    n = len(training)
    energy_mae = abs(delta_Y[:n]).mean()
    forces_mae = abs(delta_Y[n:]).mean()

    return inducing, weights, mean_energy, energy_mae, forces_mae


def prediction(atoms, cutoff, alpha, beta, eta, inducing, weights, mean_energy):
    descriptor_data = get_descriptor_data(atoms, cutoff, alpha, beta)
    energy = mean_energy
    grad = np.zeros_like(atoms.positions)
    for x, w in zip(inducing, weights):
        kern, grad_kern = get_design_matrix(x, descriptor_data, eta)
        energy += w * kern
        grad += w * grad_kern
    return energy, -grad
