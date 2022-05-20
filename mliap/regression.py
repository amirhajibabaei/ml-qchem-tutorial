# +
import numpy as np
from descriptor import get_rbf_descriptors
from kernel import dot_product_design_matrix


def regression(training, cutoff, alpha, beta, eta, max_inducing):

    # Parse training data:
    targets = []
    descriptor_data = []
    inducing = []
    for atoms in training:
        data = get_rbf_descriptors(atoms, cutoff, alpha, beta)
        descriptor_data.append(data)
        for neighbors, descriptor, jacobian in data:
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
            kern, grad = dot_product_design_matrix(x, data, eta)
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
    descriptor_data = get_rbf_descriptors(atoms, cutoff, alpha, beta)
    energy = mean_energy
    grad = np.zeros_like(atoms.positions)
    for x, w in zip(inducing, weights):
        kern, grad_kern = dot_product_design_matrix(x, descriptor_data, eta)
        energy += w * kern
        grad += w * grad_kern
    return energy, -grad
