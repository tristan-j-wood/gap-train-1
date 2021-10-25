from ase.calculators.calculator import Calculator
import gaptrain
from gaptrain.gap import UmbrellaGAP
from gaptrain.calculators import DFTB
from gaptrain.md import run_umbrella_gapmd, run_umbrella_dftbmd
from gaptrain.data import Data
from gaptrain.log import logger
from ase.atoms import Atoms
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
import os
from copy import deepcopy
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt


def _get_mpair_distance_derivative(atoms, indx, ref):
    """Calculates the vector of the derivative for an m-pair harmonic bias"""

    num_pairs = len(indx)
    derivitive_vector = np.zeros((len(atoms), 3))

    euclidean_dists = [atoms.get_distance(indx[i][0], indx[i][1], mic=True)
                       for i in range(num_pairs)]

    sum_dists = np.sum(euclidean_dists)
    normalisation = (2 / (num_pairs ** 2)) * (sum_dists - (num_pairs * ref))

    for i, pair in enumerate(indx):
        x_dist, y_dist, z_dist = [atoms[pair[0]].position[j] -
                                  atoms[pair[1]].position[j] for j in range(3)]

        x_i = x_dist * normalisation / euclidean_dists[i]
        y_i = y_dist * normalisation / euclidean_dists[i]
        z_i = z_dist * normalisation / euclidean_dists[i]

        derivitive_vector[pair[0]][:] = [x_i, y_i, z_i]
        derivitive_vector[pair[1]][:] = [-x_i, -y_i, -z_i]

    return derivitive_vector


def _get_torsion_derivative(atoms, indexes, reference):

    return NotImplementedError


def _get_rmsd_derivative(atoms, indexes, reference):

    return NotImplementedError


class RxnCoordinateAtoms(Atoms):
    """
    Extends the ASE Atoms class to include a function which returns
    the Euclidean distance reaction coordinate
    """

    def get_rxn_coords(self, indexes):

        num_pairs = len(indexes)
        euclidean_dists = [self.get_distance(indexes[i][0],
                                             indexes[i][1],
                                             mic=True)
                           for i in range(num_pairs)]

        avg_distance = (1 / num_pairs) * np.sum(euclidean_dists)
        assert avg_distance > 0

        return avg_distance

    def get_nd_coords(self, indexes):

        num_pairs = len(indexes)
        euclidean_dists = [self.get_distance(indexes[i][0],
                                             indexes[i][1],
                                             mic=True)
                           for i in range(num_pairs)]

        return euclidean_dists


class DFTBUmbrellaCalculator(DFTB):
    """Incomplete implementation of DFTB umbrella sampling calculator"""

    implemented_properties = ["energy", "forces"]

    def _calculate_force_bias(self, atoms):

        if self.coord_type == 'pairs':
            coord_derivative = _get_mpair_distance_derivative(atoms,
                                                              self.coordinate,
                                                              self.reference)

        if self.coord_type == 'rmsd':
            return NotImplementedError

        if self.coord_type == 'torsion':
            return NotImplementedError

        bias = -0.5 * self.spring_const * coord_derivative

        return bias

    def _calculate_energy_bias(self, atoms):

        indexes = self.coordinate

        if self.coord_type == 'pairs':
            num_pairs = len(indexes)
            euclidean_dists = [atoms.get_distance(indexes[i][0], indexes[i][1],
                                                  mic=True)
                               for i in range(num_pairs)]

            average_dists = (1 / num_pairs) * np.sum(euclidean_dists)

            coord = (average_dists - self.reference) ** 2

        bias = 0.5 * self.spring_const * coord

        return bias

    def get_potential_energy(self, atoms=None, force_consistent=False,
                             apply_constraint=True):

        dftb_atoms = atoms.copy()

        dftb_calc = DFTB(kpts=self.kpts,
                         Hamiltonian_Charge=self.hamiltonian_charge)
        dftb_atoms.set_calculator(dftb_calc)

        bias = self._calculate_energy_bias(dftb_atoms)
        energy = dftb_atoms.get_potential_energy() + bias

        return energy

    def get_forces(self, atoms=None, force_consistent=False,
                   apply_constraint=True, **kwargs):

        dftb_atoms = atoms.copy()

        dftb_calc = DFTB(kpts=self.kpts,
                         Hamiltonian_Charge=self.hamiltonian_charge)
        dftb_atoms.set_calculator(dftb_calc)

        bias = self._calculate_force_bias(dftb_atoms)

        if self.save_forces:

            force_mags = []
            for i, _ in enumerate(self.coordinate):
                force_vec = bias[self.coordinate[i][0]]
                force_mags.append(np.linalg.norm(force_vec))

            self.force_mag = (1 / len(self.coordinate) * np.sum(force_mags))

            euclidean_dists = [dftb_atoms.get_distance(self.coordinate[i][0],
                                                       self.coordinate[i][1],
                                                       mic=True)
                               for i in range(len(self.coordinate))]

            self.euclid_distance = (1 / len(self.coordinate) *
                                    np.sum(euclidean_dists))

        forces = dftb_atoms.get_forces() + bias

        return forces

    def __init__(self, configuration=None, kpts=(1, 1, 1), coord_type=None,
                 coordinate=None, spring_const=None, reference=None,
                 save_forces=False, **kwargs):
        super().__init__(restart=None, label='dftb', atoms=None,
                         kpts=kpts, slako_dir=None, **kwargs)

        self.configuration = configuration
        self.kpts = kpts
        self.hamiltonian_charge = kwargs["Hamiltonian_Charge"]
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.spring_const = spring_const
        self.reference = reference
        self.save_forces = save_forces
        self.force_mag = None
        self.euclid_distance = None


class GAPUmbrellaCalculator(Calculator):
    """
    Custom Calculator which modifies the energy and force according to
    a harmonic bias along a specified reaction coordinate
    """

    implemented_properties = ["energy", "forces"]

    def _calculate_force_bias(self, atoms):

        if self.coord_type == 'pairs':
            coord_derivative = _get_mpair_distance_derivative(atoms,
                                                              self.coordinate,
                                                              self.reference)

        if self.coord_type == 'rmsd':
            return NotImplementedError

        if self.coord_type == 'torsion':
            return NotImplementedError

        bias = -0.5 * self.spring_const * coord_derivative

        return bias

    def _calculate_energy_bias(self, atoms):

        indexes = self.coordinate

        if self.coord_type == 'pairs':
            num_pairs = len(indexes)
            euclidean_dists = [atoms.get_distance(indexes[i][0], indexes[i][1],
                                                  mic=True)
                               for i in range(num_pairs)]

            average_dists = (1 / num_pairs) * np.sum(euclidean_dists)

            coord = (average_dists - self.reference) ** 2

        bias = 0.5 * self.spring_const * coord

        return bias

    def get_potential_energy(self, atoms=None, force_consistent=False,
                             apply_constraint=True):

        gap_atoms = atoms.copy()
        gap_atoms.set_calculator(self.calculator)

        bias = self._calculate_energy_bias(gap_atoms)

        energy = gap_atoms.get_potential_energy() + bias

        return energy

    def get_forces(self, atoms=None, force_consistent=False,
                   apply_constraint=True, **kwargs):

        gap_atoms = atoms.copy()
        gap_atoms.set_calculator(self.calculator)

        bias = self._calculate_force_bias(gap_atoms)

        if self.save_forces:

            force_mags = []
            for i, _ in enumerate(self.coordinate):
                force_vec = bias[self.coordinate[i][0]]
                force_mags.append(np.linalg.norm(force_vec))

            self.force_mag = (1 / len(self.coordinate) * np.sum(force_mags))

            euclidean_dists = [gap_atoms.get_distance(self.coordinate[i][0],
                                                      self.coordinate[i][1],
                                                      mic=True)
                               for i in range(len(self.coordinate))]

            self.euclid_distance = (1 / len(self.coordinate) *
                                    np.sum(euclidean_dists))

        forces = gap_atoms.get_forces() + bias

        return forces

    def __init__(self, gap_calc=None, coord_type=None, coordinate=None,
                 spring_const=None, reference=None, save_forces=False,
                 **kwargs):
        Calculator.__init__(self, restart=None,
                            label=None, atoms=None, **kwargs)

        self.calculator = gap_calc
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.spring_const = spring_const
        self.reference = reference
        self.save_forces = save_forces

        self.force_mag = None
        self.euclid_distance = None

        assert coord_type in ['rmsd', 'torsion', 'pairs']
        assert coordinate is not None
        assert spring_const is not None
        assert reference is not None


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling in windows and running WHAM analysis using PyWham
    """

    def generate_pulling_configs(self, temp=None, dt=None, interval=None,
                                 pulling_rate=None, final_value=None,
                                 **kwargs):
        """Generates an MD trajectory along a reaction coordinate

        :param temp: (float) Temperature in K to run pulling simulation and
                     umbrella sampling

        :param dt: (float) Timestep in fs

        :param interval: (int) Interval between printing the geometry

        :param pulling_rate: (float) Rate of pulling in Å / fs

        :param final_value: (float) Final separation of system in Å

        :param kwargs: {fs, ps, ns} Simulation time in some units
        """

        self.final_value = final_value

        assert pulling_rate is not None
        if pulling_rate < 0:
            self.pulling_simulation = False

        self.pulling_rate = pulling_rate

        distance = final_value - self.reference
        self.simulation_time = distance / self.pulling_rate

        if self.simulation_time < 100:
            logger.warning(f'Simulation time = {self.simulation_time:.1f}'
                           f' < 100 fs!'
                           f' Decrease pulling rate or increase distance'
                           f' to increase simulation time')

        logger.info(f'Running pulling simulation for'
                    f' {self.simulation_time:.0f} fs')

        if self.method == 'gap':
            traj = run_umbrella_gapmd(configuration=self.init_config,
                                      umbrella_gap=self.umbrella_gap,
                                      temp=temp,
                                      dt=dt,
                                      interval=interval,
                                      distance=distance,
                                      pulling_rate=pulling_rate,
                                      save_forces=True,
                                      **kwargs)

        elif self.method == 'dftb':
            traj = run_umbrella_dftbmd(configuration=self.init_config,
                                       ase_atoms=self.ase_atoms,
                                       temp=temp,
                                       dt=dt,
                                       interval=interval,
                                       distance=distance,
                                       pulling_rate=pulling_rate,
                                       save_forces=True,
                                       **kwargs)

        traj.save('pulling_traj.xyz')

        return traj

    def run_umbrella_sampling(self, traj, temp, dt, interval,
                                  num_windows=10, pulling_rate=None, **kwargs):
        """Development function for variable K and reference for US"""

        if ('ps' and 'ns' and 'fs') not in kwargs:
            raise ValueError("Must specify time in umbrella sampling windows")

        if pulling_rate is not None:
            logger.error("Pulling rate must be None for umbrella "
                         "sampling simulations!")

        init_value = self.reference
        distance = self.final_value - init_value

        self.num_windows = num_windows
        distance_intervals = distance / (self.num_windows - 1)

        # Get a dictonary of reaction coordinate distances for each frame
        traj_dists = {}
        for index, frame in enumerate(traj):
            frame_atoms = frame.ase_atoms()
            num_pairs = len(self.coordinate)
            euclidean_dists = [frame_atoms.get_distance(self.coordinate[i][0],
                                                        self.coordinate[i][1],
                                                        mic=True)
                               for i in range(num_pairs)]

            avg_distance = (1 / num_pairs) * np.sum(euclidean_dists)
            traj_dists[index] = avg_distance

        # Get the initial configurations used in the umbrella sampling windows
        umbrella_frames = Data()
        for _ in range(self.num_windows):
            window_dists = deepcopy(traj_dists)

            for frame_key, dist_value in window_dists.items():
                window_dists[frame_key] = abs(dist_value - init_value)

            traj_index = min(window_dists.keys(), key=window_dists.get)
            init_value += distance_intervals

            umbrella_frames.add(traj[traj_index])

        combined_traj = Data()
        combined_coords = []

        def _run_individual_umbrella(frame_config):

            window_atoms = frame_config.ase_atoms()
            logger.info(f'Running umbrella sampling')

            num_pairs = len(self.coordinate)
            euclidean_dists = [window_atoms.get_distance(self.coordinate[i][0],
                                                         self.coordinate[i][1],
                                                         mic=True)
                               for i in range(num_pairs)]

            window_reference = np.mean(euclidean_dists)

            self.umbrella_gap.reference = window_reference
            self.umbrella_gap.spring_const = self.spring_const

            # Add window back in as argument
            # logger.info(f'Window {window} with reference '
            #             f'{self.umbrella_gap.reference:.2f} Å')

            traj = run_umbrella_gapmd(configuration=frame_config,
                                      umbrella_gap=self.umbrella_gap,
                                      temp=temp,
                                      dt=dt,
                                      interval=interval,
                                      save_forces=False,
                                      **kwargs)
            logger.info(f'Spring constant: {self.spring_const}')

            # return traj and call self.umbrella_gap.reference
            return traj, window_reference

        convergence_testing = False

        # Need to make this code a bit more user-flexible
        if convergence_testing:
            conv_parms_list = []
            init_time = kwargs["fs"]
            kwargs["fs"] = 1000
            for _ in range(10):
                conv_traj, conv_ref = _run_individual_umbrella(umbrella_frames[0])
                conv_data = [coord.rxn_coord for coord in conv_traj]
                conv_parms = self._fit_gaussian(conv_data)
                logger.info(conv_parms)
                conv_parms_list.append(conv_parms)
                kwargs["fs"] += 300
                logger.info(f'time: {kwargs}')

            kwargs["fs"] = init_time
            # Need to check Gaussians are normalised

            with open('convergence_parms.txt', 'w') as outfile:
                for coeffients in conv_parms_list:
                    print(f'{coeffients[0]}',
                          f'{coeffients[1]}',
                          f'{coeffients[2]}', file=outfile)

        # Gaussian overlap calculations
        gaussian_pair_parms = [None, None]

        overlaps_lower, overlaps_upper = [], []
        ref_discrepancies = []
        standard_deviation = []
        inital_spring = self.spring_const

        for window, frame in enumerate(umbrella_frames):

            win_traj, win_ref = _run_individual_umbrella(frame)
            combined_traj += win_traj
            with open(f'window_{self.window_count}.txt', 'w') as outfile:
                print(f'# {win_ref} {self.spring_const} {self.window_count}',
                      file=outfile)
                for i, configuration in enumerate(win_traj):
                    print(f'{i}',
                          f'{configuration.rxn_coord}',
                          f'{configuration.energy}', file=outfile)
            self.window_count += 1

            window_data = [coord.rxn_coord for coord in win_traj]
            combined_coords.append(window_data)

            if window == 0:
                gaussian_pair_parms[0] = self._fit_gaussian(window_data)

                ref_discrepancies.append(
                    gaussian_pair_parms[0][1] - win_ref)
                standard_deviation.append(gaussian_pair_parms[0][2])

            else:
                max_k_iters = 3
                threshold = 0.05

                for interation in range(max_k_iters):
                    gaussian_pair_parms[1] = self._fit_gaussian(window_data)
                    overlaps = self._get_overlap(gaussian_pair_parms[0],
                                                 gaussian_pair_parms[1])
                    if min(overlaps) < threshold:
                        logger.info(f'Overlap too small ({min(overlaps)}),'
                                    f' decreasing K')
                        self.spring_const = self.spring_const * 0.5

                        win_traj, win_ref = _run_individual_umbrella(frame)
                        combined_traj += win_traj
                        with open(f'window_{self.window_count}.txt',
                                  'w') as outfile:
                            print(
                                f'# {win_ref} {self.spring_const} {self.window_count}',
                                file=outfile)
                            for i, configuration in enumerate(win_traj):
                                print(f'{i}',
                                      f'{configuration.rxn_coord}',
                                      f'{configuration.energy}', file=outfile)
                        self.window_count += 1

                        window_data = [coord.rxn_coord for coord in win_traj]
                        combined_coords.append(window_data)
                    else:
                        logger.info(f'Overlap > {threshold} ({overlaps})')
                        break

                overlaps_lower.append(overlaps[0])
                overlaps_upper.append(overlaps[1])

                ref_discrepancies.append(
                    gaussian_pair_parms[1][1] - win_ref)
                standard_deviation.append(gaussian_pair_parms[1][2])

                gaussian_pair_parms[0] = gaussian_pair_parms[1]

            logger.info(f'Window count: {self.window_count}')

            with open(f'nd_coord_{self.window_count}.txt', 'w') as outfile:
                if self.method == 'gap':
                    print(f'# {win_ref} {self.window_count}', file=outfile)
                    for i, configuration in enumerate(win_traj):
                        print(f'{configuration.nd_coord}', file=outfile)
                else:
                    logger.error("N-d coordinate printing only implemented"
                                 "for gap currently")

            self.spring_const = inital_spring

        with open('overlap_data.txt', 'w') as outfile:
            print(f'{overlaps_lower}',
                  f'{overlaps_upper}',
                  f'{ref_discrepancies}',
                  f'{standard_deviation}', file=outfile, sep='\n')

        combined_traj.save(filename='combined_windows.xyz')

        return None

    def run_wham_analysis(self, temp, num_bins=30, tol=0.00001, numpad=0,
                          wham_path=None, num_MC_trials=0, randSeed=1,
                          correlation=10):
        """Calculates the Gibbs free energy using the WHAM method"""

        assert wham_path is not None

        file_list = [f'window_{i}.txt' for i in range(self.window_count)]

        if self.wham_method == 'grossman':

            self.correlation = correlation

            if self.pulling_simulation:
                hist_min = self.reference
                hist_max = self.final_value

            else:
                hist_max = self.reference
                hist_min = self.final_value

            metadatafile = 'metadata.txt'
            freefile = 'free_energy.txt'

            self._write_metafile(file_list)

            logger.warning('Ensure that the units in the Grossman WHAM '
                           'implementation are in eV')

            os.system(f'{wham_path} {hist_min} {hist_max} {num_bins} {tol} '
                      f'{temp} {numpad} {metadatafile} {freefile} '
                      f'{num_MC_trials} {randSeed}')

        else:
            logger.error("Only grossman WHAM implemented now")

        return None

    def _write_metafile(self, file_names):
        with open('metadata.txt', 'w') as outfile:

            for file in file_names:
                with open(file, 'r') as infile:
                    line = infile.readline().split()
                    ref = line[1]
                    spring_cont = line[3]

                    print(f'{file} '
                          f'{ref} ' 
                          f'{spring_cont} '
                          f'{self.correlation} ', file=outfile)

    def _fit_gaussian(self, window_data):

        def gauss(x, *p):
            a, b, c = p
            return a * np.exp(-(x - b)**2 / (2. * c**2))

        start = min(self.reference, self.final_value)
        end = max(self.reference, self.final_value)

        x_range = np.linspace(start - 0.1 * start, end + 0.1 * end, 500)

        hist, bin_edges = np.histogram(window_data, density=True, bins=500)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        initial_guess = [1.0, 1.0, 1.0]
        parms, _ = curve_fit(gauss, bin_centres, hist, p0=initial_guess,
                             maxfev=10000)
        parms[2] = np.abs(parms[2])
        gaussian_parms = parms.tolist()

        hist_fit = gauss(x_range, *parms)

        plt.plot(bin_centres, hist, label='Test data', alpha=0.1)
        plt.plot(x_range, hist_fit, label='Fitted data')

        plt.ylabel('Density')
        plt.xlabel('Reaction coordinate / Å')

        plt.savefig('overlap_gaussians.pdf', dpi=300)

        return gaussian_parms

    def _get_overlap(self, parms_a, parms_b):

        start = min(self.reference, self.final_value)
        end = max(self.reference, self.final_value)

        overlap = Overlap(
            x_range=np.linspace(start - 0.1 * start, end + 0.1 * end, 500)
        )

        integral_overlap_1 = overlap.calculate_overlap(parms_a,
                                                       norm_func=parms_b)
        integral_overlap_2 = overlap.calculate_overlap(parms_b,
                                                       norm_func=parms_a)

        logger.info(f'Overlaps: {integral_overlap_1}, {integral_overlap_2}')

        return integral_overlap_1, integral_overlap_2

    def __init__(self, init_config=None, method=None, gap=None,
                 coordinate=None, spring_const=None, wham_method='grossman'):
        """
        :param init_config: (gaptrain.configurations.Configuration | None)

        :param method: (str | None) Method to calculate energy and forces. Must
                       be in ['dftb', 'gap]

        :param gap: (gaptrain.gap.GAP | None)

        :param coordinate: (list | None) Indices of the atoms which define the
                           reaction coordinate

        :param spring_const: (float | None) Value of the spring constant, K,
                              used in umbrella sampling

        :param wham_method: (str) Method to use when performing WHAM. PyWham
                    and Grossman Wham currently implemented.
        """

        if any(isinstance(el, list) for el in coordinate):

            self.coord_type = 'pairs'
            logger.info(f'Coordinate type detected as pairs')

            flat_list = [item for sublist in coordinate for item in sublist]
            if len(set(flat_list)) != len(flat_list):
                raise ValueError("All atom indexes in pairs must be unique")

            _atoms = init_config.ase_atoms()

            num_pairs = len(coordinate)
            euclidean_dists = [_atoms.get_distance(coordinate[i][0],
                                                   coordinate[i][1],
                                                   mic=True)
                               for i in range(num_pairs)]

            self.reference = (1 / num_pairs) * np.sum(euclidean_dists)
            logger.info(f'Initial value of reference: {self.reference:.2f}')

        elif len(coordinate) == 4:
            self.coord_type = 'torsion'
        elif isinstance(coordinate, gaptrain.configurations.Configuration):
            self.coord_type = 'rmsd'
        else:
            raise ValueError("Coordinate type could not be inferred from "
                             "coordinates")

        self.spring_const = spring_const
        self.method = method
        self.init_config = init_config

        if method == 'gap':
            self.gap = gap

            self.umbrella_gap = UmbrellaGAP(name=gap.name,
                                            system=gap.system,
                                            coord_type=self.coord_type,
                                            coordinate=coordinate,
                                            spring_const=self.spring_const,
                                            reference=self.reference)

        elif method == 'dftb':
            self.ase_atoms = init_config.ase_atoms()

            self.umbrella_dftb = DFTBUmbrellaCalculator(
                                 configuration=self.init_config,
                                 kpts=(1, 1, 1),
                                 coord_type=self.coord_type,
                                 coordinate=coordinate,
                                 spring_const=self.spring_const,
                                 reference=self.reference,
                                 Hamiltonian_Charge=self.init_config.charge)

            self.ase_atoms.set_calculator(self.umbrella_dftb)

        else:
            raise ValueError("Method must be in ['dftb', 'gap]")

        assert wham_method in ['grossman', 'pywham']
        self.wham_method = wham_method

        self.coordinate = coordinate
        self.final_value = None
        self.pulling_rate = None
        self.pulling_simulation = True
        self.num_windows = None
        self.correlation = None
        self.simulation_time = None
        self.window_count = 0


class Overlap:
    """Docstrings NotImplementedError"""

    def gaussian(self, x, a, b, c):

        return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

    def calculate_area(self, low_limit, upp_limit, parms):

        a, b, c = parms
        integral = quad(self.gaussian, low_limit, upp_limit, args=(a, b, c))

        return integral

    def calc_intercepts(self, a, b, c, p, q, r):

        assert a != 0 and p != 0

        if c != r:

            a_val = (2 * c ** 2) - (2 * r ** 2)
            b_val = (4 * r ** 2 * b) - (4 * c ** 2 * q)
            c_val = (2 * c ** 2 * q ** 2) - (2 * r ** 2 * b ** 2) - (
                        4 * c ** 2 * r ** 2 * np.log(p / a))

            if b_val ** 2 - 4 * a_val * c_val <= 0:
                return False

            intercept_1 = (-b_val + np.sqrt(
                b_val ** 2 - 4 * a_val * c_val)) / (2 * a_val)
            intercept_2 = (-b_val - np.sqrt(
                b_val ** 2 - 4 * a_val * c_val)) / (2 * a_val)

            min_ab = min(intercept_1, intercept_2)
            max_ab = max(intercept_1, intercept_2)
            assert max_ab > min_ab

            intercepts = [min_ab, max_ab]

            return intercepts

        else:

            if b != q:
                intercept = ((q ** 2 - b ** 2) - 2 * c ** 2 * np.log(
                    p / a)) / (2 * (q - b))
                intercept = [intercept]

                return intercept

            elif b == q:

                return False

    def calculate_overlap(self, parms_a, norm_func):

        intercepts = self.calc_intercepts(parms_a[0], parms_a[1], parms_a[2],
                                          norm_func[0], norm_func[1],
                                          norm_func[2])
        norm_area = norm_func[0] * abs(norm_func[2]) * np.sqrt(2 * np.pi)

        if intercepts is False:

            area_a = parms_a[0] * parms_a[2] * np.sqrt(2 * np.pi)

            if area_a / norm_area <= 1:
                overlap = area_a / norm_area
                assert overlap <= 1

            else:
                overlap = 1

            return overlap

        elif len(intercepts) == 1:

            if parms_a[1] > norm_func[1]:
                lower_func = parms_a
                upper_func = norm_func
            else:
                lower_func = norm_func
                upper_func = parms_a

            int_lower = self.calculate_area(-np.inf, intercepts[0], lower_func)
            int_upper = self.calculate_area(intercepts[0], np.inf, upper_func)

            overlap = (int_lower[0] + int_upper[0]) / norm_area

            return overlap

        else:

            a, b, c = parms_a
            p, q, r = norm_func

            if (intercepts[0] < b < intercepts[1]) and (intercepts[0]
                                                        < q < intercepts[1]):
                if a > p:
                    func_1_parms = parms_a
                    func_2_parms = norm_func

                elif a < p:
                    func_1_parms = norm_func
                    func_2_parms = parms_a

                else:
                    logger.error(
                        "Could not assign functions to calculate overlap")

            elif intercepts[0] < b < intercepts[1]:
                func_1_parms = parms_a
                func_2_parms = norm_func

            elif intercepts[0] < q < intercepts[1]:
                func_1_parms = norm_func
                func_2_parms = parms_a

            else:
                logger.error(
                    "Could not assign functions to calculate overlap")

            int_lower = self.calculate_area(-np.inf, intercepts[0],
                                            func_1_parms)
            int_middle = self.calculate_area(intercepts[0], intercepts[1],
                                             func_2_parms)
            int_upper = self.calculate_area(intercepts[1], np.inf,
                                            func_1_parms)

            overlap = (int_lower[0] + int_middle[0] + int_upper[0]) / norm_area

            return overlap

    def __init__(self, x_range):

        self.x_range = x_range
