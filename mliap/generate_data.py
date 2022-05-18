# +
import ase.units as units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def create_atoms():
    unit_cell = bulk("Au", cubic=True)
    atoms = unit_cell.repeat(2)
    return atoms


def set_calculator(atoms):
    atoms.calc = EMT()


def initiate_velocities(atoms, temperature):
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)


def molecular_dynamics(atoms, temperature, steps, trajfile):
    dynamics = Langevin(
        atoms,
        timestep=2 * units.fs,
        temperature_K=temperature,
        friction=1e-3,
        trajectory=trajfile,
        logfile="-",
    )
    dynamics.run(steps)


def main():
    temperature = 1000.0  # kelvin
    steps = 5000
    atoms = create_atoms()
    set_calculator(atoms)
    initiate_velocities(atoms, temperature)
    molecular_dynamics(atoms, temperature, steps, "Langevin_Au.traj")


if __name__ == "__main__":
    main()
