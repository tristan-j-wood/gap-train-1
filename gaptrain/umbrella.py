from ase.calculators.calculator import Calculator
from gaptrain.calculators import DFTB
from gaptrain.md import run_umbrella_gapmd
from gaptrain.data import Data
from gaptrain.log import logger
from ase.atoms import Atoms
from copy import copy
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
import os


def _write_xml_file(energy_interval, coord_interval, temp, temp_interval,
                   file_names, energy_function):
    """Writes the xml input file required for PyWham"""

    root = ET.Element("WhamSpec")
    general = ET.SubElement(root, "General")

    # Coordinate block
    coordinate = ET.SubElement(general, 'Coordinates')
    ET.SubElement(coordinate, "Coordinate", name=f"{energy_function}")
    ET.SubElement(coordinate, "Coordinate", name="Reaction coordinate")

    # Default Coordinate File Reader Block
    default_coord = ET.SubElement(general, 'DefaultCoordinateFileReader',
                                  returnsTime='false')
    ET.SubElement(default_coord, "ReturnList", name=f"{energy_function}")
    ET.SubElement(default_coord, "ReturnList", name="Reaction coordinate")

    # Binnings block
    binnings = ET.SubElement(general, 'Binnings')
    binning_1 = ET.SubElement(binnings, "Binning", name=f"{energy_function}")
    ET.SubElement(binning_1, "Interval").text = f'{energy_interval}'
    binning_2 = ET.SubElement(binnings, "Binning", name="Reaction coordinate")
    # Add begin/end if necessary
    # ET.SubElement(binning_2, "Begin").text = "interval"
    # ET.SubElement(binning_2, "End").text = "interval"
    ET.SubElement(binning_2, "Interval").text = f'{coord_interval}'

    # Trajectories block
    trajectories = ET.SubElement(root, "Trajectories")

    for i, file_name in enumerate(file_names):
        trajectory_1 = ET.SubElement(trajectories, 'Trajectory', T=f'{temp[i]}')
        ET.SubElement(trajectory_1, 'EnergyFunction').text = f'{energy_function}'
        coord_file = ET.SubElement(trajectory_1, 'CoordinateFiles')
        ET.SubElement(coord_file, 'CoordinateFile').text = f'{file_name}'

    # Jobs block
    jobs = ET.SubElement(root, "Jobs")

    free_energy = ET.SubElement(jobs, 'FreeEnergy', outFilePrefix='out/fe')
    coordinates = ET.SubElement(free_energy, 'Coordinates')
    ET.SubElement(coordinates, 'Coodinate', name='Reaction coodinate')
    ET.SubElement(free_energy, 'EnergyFunction').text = f'{energy_function}'
    ET.SubElement(free_energy,
                  'Temperatures').text = f'{temp[0]}:{temp_interval}:{temp[-1]}'
    parameters = ET.SubElement(free_energy, 'Parameters')
    ET.SubElement(parameters, 'Parameter', name="in_kT").text = "true"

    heat_capacity = ET.SubElement(jobs, 'HeatCapacity', outFile='out/cv')
    ET.SubElement(heat_capacity, 'EnergyFunction').text = f'{energy_function}'
    ET.SubElement(heat_capacity,
                  'Temperatures').text = f'{temp[0]}:{temp_interval}:{temp[-1]}'

    ET.SubElement(jobs, 'DensityOfStates', outFile='out/dos')

    # Write file with indentation
    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open("wham.spec.xml", "w") as f:
        f.write(xml_string)


def _get_distance_derivative(atoms, indexes, reference):
    """Calculates the vector of the derivative of the harmonic bias"""

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


class RxnCoordinateAtoms(Atoms):
    """
    Extends the ASE Atoms class to include a function which returns
    the Euclidean distance reaction coordinate
    """

    def get_rxn_coords(self, indexes):

        euclidean_distance = self.atoms.get_distance(indexes[0], indexes[1],
                                                     mic=True)

        assert euclidean_distance > 0

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
                                  num_windows=self.num_windows,
                                  **self.kwargs)

        traj.save('traj_test_energy.xyz')

        umbrella_frames = Data()
        frame_num = len(traj) // 10
        [umbrella_frames.add(frame) for frame in traj[::frame_num]]

        self.simulation_time = self.distance / self.pulling_rate

        return umbrella_frames

    def run_umbrella_sampling(self, frames, gap, temp, dt, interval,
                              coord_type, coordinate, bias_strength, **kwargs):

        logger.info(f'kwargs in umbrella.py: {kwargs}')

        # Reference needs to be different for each window!
        for i, frame in enumerate(frames):

            assert self.simulation_time is not None

            window_ref = self.init_ref + (i * self.simulation_time * self.pulling_rate / self.num_windows)
            logger.info(f'Window reference = {window_ref}')

            traj = run_umbrella_gapmd(frame,
                                      gap=gap,
                                      temp=temp,
                                      dt=dt,
                                      interval=interval,
                                      coord_type=coord_type,
                                      coordinate=coordinate,
                                      bias_strength=bias_strength,
                                      reference=window_ref,
                                      **kwargs)

            # decorator to have input files made elsewhere?

            # if it doesn't make enough files it may just print default energy
            # and reference 0.04914365296360979 for the energy

            # Also for some reason it keeps printing

            # Also need to check how I calculate the US potential energy above
            with open(f'window_{i}.txt', 'w') as outfile:
                for configuration in traj:
                    print(f'{configuration.energy}',
                          f'{configuration.rxn_coord}', file=outfile)

        return None

    def run_wham_analysis(self):

        file_list = [f'window_{i}.txt' for i in range(self.num_windows)]
        temps = [300 for _ in range(self.num_windows)]

        _write_xml_file(energy_interval=0.01, coord_interval=0.1, temp=temps,
                        temp_interval=0.1, file_names=file_list,
                        energy_function='V')

        os.system("python2 wham.py wham.spec.xml")

        return None

    def __init__(self, init_config=None, gap=None, temp=None,
                 dt=None, interval=None, coord_type=None, coordinate=None,
                 bias_strength=None, pulling_rate=None, reference=None,
                 distance=None,
                 num_windows=None, **kwargs):
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
        self.num_windows = num_windows
        self.kwargs = kwargs
        self.simulation_time = None
        # Unsure if we need to copy reference (worried it is being overwritten
        # in the pulling method
        self.init_ref = copy(reference)
