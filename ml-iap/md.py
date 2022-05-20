# +
"""
The objective of this module is to generate training
and testing data for machine learning potential.
For this a 3x3x3 cubic supercell of Au is used.
We perform a molecular dynamics (MD) simulation
and save the trajectory md.traj.

For fast demonstration, we use classical EMT
force-field instead of ab initio DFT potential.
But the calculation can be switched to ab initio
molecular dynamics (MLMD) simply by setting an
ab initio calculater in "set_calculator" function.


run:
    python data.py

output:
    md.traj

"""
import ase.units as units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def create_atoms(element, supercell):
    unit_cell = bulk(element, cubic=True)
    atoms = unit_cell.repeat(supercell)
    return atoms


def set_calculator(atoms):
    atoms.calc = EMT()


def initiate_velocities(atoms, temperature):
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)


def molecular_dynamics(atoms, temperature, steps, trajectory):
    """
    Note that despite using NPT, the simulation is still
    NVT since parameters needed for NPT are not specified.
    """
    dynamics = NPT(
        atoms,
        timestep=5 * units.fs,  # use 2 fs!
        temperature_K=temperature,
        ttime=100 * units.fs,
        externalstress=0,  # useless parameter!
        trajectory=trajectory,
        logfile="md.log",
    )
    dynamics.run(steps)


def main(
    element="Au", supercell=3, temperature=1000.0, steps=1000, trajectory="md.traj"
):
    atoms = create_atoms(element, supercell)
    atoms.rattle(0.1)
    set_calculator(atoms)
    initiate_velocities(atoms, temperature)
    molecular_dynamics(atoms, temperature, steps, trajectory)


if __name__ == "__main__":
    main()
