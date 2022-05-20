# +
import os
from test import test_model

import md
from ase.io import read
from regression import regression


def main():

    # Generate md data (only if not already available)
    if not os.path.isfile("md.traj"):
        print("\nRunning MD (only needed the 1st time) ... ")
        print("See md.log for progress ...")
        md.main(trajectory="md.traj")

    # Read and split data into training and testing
    data = read("md.traj", "::50")
    training_data = data[0::2]
    testing_data = data[1::2]
    print(f"Num. training data: {len(training_data)}")
    print(f"Num. testing data: {len(testing_data)}")

    # Generate a model
    print("\nBuiling the model ...")
    cutoff = 5.0
    alpha = 0.5
    beta = 0.5
    eta = 4
    max_inducing = 100
    inducing, weights, mean_energy, energy_mae, forces_mae = regression(
        training_data, cutoff, alpha, beta, eta, max_inducing
    )

    print("\nTraining (mean absolute) errors:")
    print(f"Energy: {energy_mae}")
    print(f"Forces: {forces_mae}")

    # Test the model
    print("\nTesting the model ...")
    test_energy_mae, test_forces_mae = test_model(
        testing_data, cutoff, alpha, beta, eta, inducing, weights, mean_energy
    )

    print("\nTesting (mean absolute) errors:")
    print(f"Energy: {test_energy_mae}")
    print(f"Forces: {test_forces_mae}")


if __name__ == "__main__":
    main()
