# +
import os
from test import test_model

import md
import numpy as np
from ase.io import read
from regression import regression


def main(
    cutoff=5.0,
    alpha=0.5,
    beta=0.5,
    eta=4,
    max_inducing=100,
    num_md_data=10,  # use 100
    data_spacing=10,  # use 100
    seed=574687867,
):

    """
    Args:
        cutoff        neighborlist/interactions cutoff
        alpha, beta   descriptor hyper-parameters
        eta           kernel hyper-parameter
        max_inducing  maximum number of inducing points (random selection)
        num_md_data   total number data points generated from md
                      this data will be split into training/testing equally
        data_spacing  a data point will sapmed after this many md steps
        seed          seed for the random number generator
                      setting this seed guarantees reproducibility

    Notes:
        For statistical certainty num_md_data and data_spacing can be
        increased.
        "alpha", "beta", and "gamma" are the main hyper-parameters which
        can be optimized. "cutoff" and "max_inducing" can be considered
        as secondary hyper-parameters.

    """

    # Set a random seed for reproducible simulations
    if seed:
        np.random.seed(seed)

    # Generate md data (only if not already available)
    if not os.path.isfile("md.traj"):
        steps = num_md_data * data_spacing
        print(f"\nRunning MD for {steps} steps (only needed the 1st time) ... ")
        print("See md.log for progress ...")
        md.main(trajectory="md.traj", steps=steps)
        print("md.traj is generated.")
    else:
        print("md.traj exists! skipping md.")

    # Read and split data into training and testing
    data = read("md.traj", f"::{data_spacing}")
    training_data = data[0::2]
    testing_data = data[1::2]
    print(f"Num. training data: {len(training_data)}")
    print(f"Num. testing data: {len(testing_data)}")

    # Generate a model
    print("\nBuiling the model ...")
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
