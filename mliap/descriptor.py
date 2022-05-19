# +
import numpy as np
from neighborlist import get_neighbor_list


def get_rbf_descriptors(atoms, cutoff, alpha, beta):
    """
    Calculates RBF descriptor for local environment
    of each atom within a cutoff.

    Args:
        atoms         ase.Atoms object
        cutoff        cutoff radius
        alpha, beta   see "rbf_descriptor" function below

    Returns:
        [(n_0, d_0, j_0), (n_1, d_1, j_1), ...]
        where
        n_i  -->  indices of the neighbors of atom i
        d_i  -->  rbf descriptor for atom i
        j_i  -->  jacobian of d_i wrt rij

        Note: rij is coordinates of n_i relative to i

    """
    nl = get_neighbor_list(atoms, cutoff)
    descriptor_data = []
    for a in atoms:
        i = a.index
        j, offsets = nl.get_neighbors(i)
        shifts = (offsets[..., None] * atoms.cell).sum(axis=1)
        rij = atoms.positions[j] - atoms.positions[i] + shifts
        d, jac = rbf_descriptor(rij, cutoff, alpha, beta)
        descriptor_data.append((j, d, jac))
    return descriptor_data


def rbf_descriptor(r, cutoff, alpha, beta, normalized=True):
    """
    Radial Basis Functions (RBF) descriptor.

    Args:
        r          coordinates of neighbors within cutoff
        cutoff     cutoff radius
        alpha      spacing for a 1d grid in the range (0, cutoff)
        beta       bandwidth of Gaussian density functions

    Returns:
        p     descriptor vector
        q     jacobian of p wrt r

    """

    # Descriptor: p
    d = np.linalg.norm(r, axis=1)
    assert all(d < cutoff)
    c = smooth_cutoff(d, cutoff)
    grid = radial_grid(cutoff, alpha)
    x = grid[:, None] - d[None, :]  # x[i, j] = grid[i] - d[j]
    g = gaussian(x, beta)
    p = (g * c).sum(axis=1)

    # Jacobian: q
    # We use D_y_x to indicate derivative of y wrt x (dy/dx).
    D_d_r = r / d[:, None]
    D_c_d = smooth_cutoff_derivative(d, cutoff)
    D_c_r = D_c_d[:, None] * D_d_r
    D_g_x = gaussian_derivative(x, beta)
    D_g_r = -D_g_x[:, :, None] * D_d_r
    q = D_g_r * c[:, None] + g[:, :, None] * D_c_r

    # Normalize:
    if normalized:
        norm = np.linalg.norm(p)
        p = p / norm
        q = q / norm
        q = q - p[..., None, None] * (p[..., None, None] * q).sum(axis=0)

    return p, q


def smooth_cutoff(d, cutoff):
    return (1 - d / cutoff) ** 2


def smooth_cutoff_derivative(d, cutoff):
    return 2 * (-1 / cutoff) * (1 - d / cutoff)


def gaussian(x, beta):
    g = np.exp(-0.5 * (x / beta) ** 2)
    return g


def gaussian_derivative(x, beta):
    return -(x / beta**2) * gaussian(x, beta)


def radial_grid(cutoff, alpha):
    return np.arange(0.0, cutoff, alpha)


def _generate_test_r(n, cutoff):
    rand = np.random.uniform(size=(10 * n, 3))
    r = 2 * (rand - 0.5) * cutoff
    d = np.linalg.norm(r, axis=1)
    m = np.logical_and(d > 0.1 * cutoff, d < cutoff)
    return r[m][:n]


def test_rbf_jacobian():
    cutoff = 3.0
    alpha = 0.5
    beta = 0.5
    delta = 1e-6
    for _ in range(1000):
        r = _generate_test_r(10, cutoff)
        # Analytical
        d1, jac = rbf_descriptor(r, cutoff, alpha, beta)
        jac_ana = jac[:, 0, 0]
        # Finite-difference
        r[0, 0] += delta
        d2, _ = rbf_descriptor(r, cutoff, alpha, beta)
        jac_fd = (d2 - d1) / delta
        # Compare:
        # print(abs(jac_fd - jac_ana).max())
        assert np.allclose(jac_ana, jac_fd, atol=2 * delta)


if __name__ == "__main__":
    test_rbf_jacobian()
