# +
import numpy as np
import pylab as plt
from regression import prediction


def test_model(
    test_data,
    cutoff,
    alpha,
    beta,
    eta,
    inducing,
    weights,
    mean_energy,
    figure="test.pdf",
):
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

    if figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_aspect("equal")
        ax1.scatter(e_exact, e_pred)
        ax1.set_xlabel("exact energies")
        ax1.set_ylabel("predicted energies")
        line = [e_exact.min(), e_exact.max()]
        ax1.plot(line, line, color="r")

        ax2.set_aspect("equal")
        ax2.scatter(f_exact, f_pred)
        ax2.set_xlabel("exact forces")
        ax2.set_ylabel("predicted forces")
        line = [f_exact.min(), f_exact.max()]
        ax2.plot(line, line, color="r")
        fig.savefig(figure)
        print(f"{figure} is generated.")

    test_energy_mae = abs(e_exact - e_pred).mean()
    test_forces_mae = abs(f_exact - f_pred).mean()
    return test_energy_mae, test_forces_mae
