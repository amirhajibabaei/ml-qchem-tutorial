# +
from ase.neighborlist import NeighborList


def get_neighbor_list(atoms, cutoff):
    nl = NeighborList(
        len(atoms) * [cutoff / 2],
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)
    return nl
