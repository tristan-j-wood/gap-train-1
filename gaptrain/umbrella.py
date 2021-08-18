from ase.calculators.calculator import Calculator
from gaptrain.calculators import DFTB
from gaptrain.md import run_umbrella_gapmd
from gaptrain.data import Data
from ase.atoms import Atoms
import logging
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


class CustomAtoms(Atoms):

    def get_rxn_coords(self, indexes):

        euclidean_distance = self.atoms.get_distance(indexes[0], indexes[1],
                                                     mic=True)

        return euclidean_distance

    def __init__(self, atoms=None):

        self.atoms = atoms


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

        logging.info(f'Reference: {self.reference}')

        return forces

    def __init__(self, gap_calc=None, coord_type=None, coordinate=None,
                 bias_strength=None, reference=None, **kwargs):
        Calculator.__init__(self, restart=None,
                            label=None, atoms=None, **kwargs)

        self.calculator = gap_calc
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.bias_strength = bias_strength
        self.reference = reference

        assert coord_type in ['distance', 'rmsd', 'torsion']
        assert coordinate is not None
        assert bias_strength is not None
        assert reference is not None


class UmbrellaSampling:

    def generate_pulling_configs(self):

        traj = run_umbrella_gapmd(configuration=self.init_config,
                                  gap=self.gap,
                                  temp=self.temp,
                                  dt=self.dt,
                                  interval=self.interval,
                                  coord_type=self.coord_type,
                                  coordinate=self.coordinate,
                                  bias_strength=self.bias_strength,
                                  reference=self.reference,
                                  distance=self.distance,
                                  pulling_rate=self.pulling_rate,
                                  **self.kwargs)

        traj.save('traj_test_energy.xyz')

        umbrella_frames = Data()
        # Need to modify splicing such that it takes e.g., n % of frames
        [umbrella_frames.add(frame) for frame in traj[::10]]

        return umbrella_frames

    def run_umbrella_sampling(self, frames, gap, temp, dt, interval,
                              coord_type, coordinate, bias_strength, reference,
                              **kwargs):

        for i, frame in enumerate(frames):
            traj = run_umbrella_gapmd(frame,
                                       gap=gap,
                                    temp=temp,
                                    dt=dt,
                                    interval=interval,
                                    coord_type=coord_type,
                                    coordinate=coordinate,
                                    bias_strength=bias_strength,
                                    reference=reference)

            # decorator to have input files made elsewhere?
            with open('test_input.txt', 'w') as outfile:
                for configuration in traj:
                    print(f'{configuration.energy}',
                          f'{configuration.rxn_coord}', file=outfile)

        return NotImplementedError

    def run_wham_analysis(self):

        # Function to generate/make input files for wham and then run it

        return NotImplementedError

    def __init__(self, init_config=None, gap=None, temp=None,
                 dt=None, interval=None, coord_type=None, coordinate=None,
                 bias_strength=None, pulling_rate=None, reference=None,
                 init_ref=None, final_ref=None, distance=None, **kwargs):
        """
        :param init_config: (gaptrain.configurations.Configuration)

        :param gap: (gaptrain.gap.GAP)

        :param temp: (float) Temperature in K to run pulling simulation and
                     umbrella sampling

        :param dt: (float) Timestep in fs

        :param interval: (int) Interval between printing the geometry

        :param coord_type: (str | None) Type of coordinate to perform bias
                           along. Must be in the list ['distance', 'rmsd',
                            'torsion']

        :param coordinate: (list | None) Indices of the atoms which define the
                           reaction coordinate

        :param bias_strength: (float | None) Value of the bias strength, K,
                              used in umbrella sampling

        :param reference: (float | None) Value of the reference value, ξ_i,
                          used in umbrella sampling
        :param kwargs: {fs, ps, ns} Simulation time in some units
        """

        self.init_config = init_config
        self.gap = gap
        self.temp = temp
        self.dt = dt
        self.interval = interval
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.bias_strength = bias_strength
        self.reference = reference
        self.pulling_rate = pulling_rate
        self.distance = distance
        self.kwargs = kwargs

        if init_ref is not None:
            self.init_ref = init_ref

        if final_ref is not None:
            self.final_ref = final_ref
