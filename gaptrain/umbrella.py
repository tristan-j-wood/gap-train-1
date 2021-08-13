from ase.calculators.calculator import Calculator
from gaptrain.calculators import DFTB
import numpy as np


def _get_distance_derivative(atoms, indexes, reference):

    derivitive_vector = np.zeros((len(atoms), 3))

    atom_1, atom_2 = atoms[indexes[0]], atoms[indexes[1]]
    x_dist, y_dist, z_dist = [atom_1.position[i] - atom_2.position[i]
                              for i in range(3)]

    euclidean_distance = atoms.get_distance(indexes[0], indexes[1],
                                            mic=True)
    assert euclidean_distance > 0

    norm = 2 * (euclidean_distance - reference) / euclidean_distance
    f_x, f_y, f_z = norm * x_dist, norm * y_dist, norm * z_dist

    derivitive_vector[indexes[0]][:] = [f_x, f_y, f_z]
    derivitive_vector[indexes[1]][:] = - derivitive_vector[indexes[0]][:]

    return derivitive_vector


def _get_torsion_derivative(atoms, indexes, reference):

    return NotImplementedError


def _get_rmsd_derivative(atoms, indexes, reference):

    return NotImplementedError


class DFTBUmbrellaCalculator(DFTB):

    implemented_properties = ["energy", "forces"]

    def __init__(self, configuration=None, atoms=None, kpts=None,
                 Hamiltonian_Charge=None,
                 **kwargs):
        super().__init__(restart=None,
                         label='dftb', atoms=None, kpts=(1, 1, 1),
                         slako_dir=None,
                         **kwargs)

        self.configuration = configuration
        self.atoms = atoms
        self.kpts = kpts
        self.Hamiltonian_Charge = Hamiltonian_Charge


class GAPUmbrellaCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def _calculate_bias(self, atoms):

        if self.coord_type == 'distance':
            coord_derivative = _get_distance_derivative(atoms, self.coordinate,
                                                        self.reference)

        if self.coord_type == 'rmsd':
            return NotImplementedError

        if self.coord_type == 'torsion':
            return NotImplementedError

        bias = -0.5 * self.bias_strength * coord_derivative

        return bias

    def get_potential_energy(self, atoms=None, force_consistent=False,
                             apply_constraint=True):

        gap_atoms = atoms.copy()
        gap_atoms.set_calculator(self.calculator)
        energy = gap_atoms.get_potential_energy()

        return energy

    def get_forces(self, atoms=None, force_consistent=False,
                   apply_constraint=True, **kwargs):

        gap_atoms = atoms.copy()
        gap_atoms.set_calculator(self.calculator)

        bias = self._calculate_bias(gap_atoms)

        forces = gap_atoms.get_forces() + bias

        return forces

    def __init__(self, gap_calc=None, coord_type=None, coordinate=None,
                 bias_strength=None, reference=None, **kwargs):
        Calculator.__init__(self, restart=None,
                            label=None, atoms=None, **kwargs)

        assert coord_type in ['distance, rmsd', 'torsion']
        assert coordinate is not None
        assert bias_strength is not None
        assert reference is not None

        self.calculator = gap_calc
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.bias_strength = bias_strength
        self.reference = reference
