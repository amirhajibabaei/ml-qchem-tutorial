# +
import numpy as np
from regression import prediction


def test_model(test_data, cutoff, alpha, beta, eta, inducing, weights, mean_energy):
    e_exact = []
    f_exact = []
    e_pred = []
    f_pred = []
    for atoms in test_data:
        e1 = atoms.get_potential_energy()
        f1 = atoms.get_forces()
        e2, f2 = prediction(
            atoms, cutoff, alpha, beta, eta, inducing, weights, mean_energy
        )
        e_exact.append(e1)
        e_pred.append(e2)
        f_exact.append(f1)
        f_pred.append(f2)
    e_exact = np.array(e_exact)
    e_pred = np.array(e_pred)
    f_exact = np.concatenate(f_exact)
    f_pred = np.concatenate(f_pred)

    test_energy_mae = abs(e_exact - e_pred).mean()
    test_forces_mae = abs(f_exact - f_pred).mean()
    return test_energy_mae, test_forces_mae
