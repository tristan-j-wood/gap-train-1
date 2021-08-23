from ase.calculators.calculator import Calculator
import gaptrain
from gaptrain.gap import UmbrellaGAP
from gaptrain.calculators import DFTB
from gaptrain.md import run_umbrella_gapmd
from gaptrain.data import Data
from gaptrain.log import logger
from ase.atoms import Atoms
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
import os


def _write_xml_file(energy_interval, coord_interval, temp, temp_interval,
                    file_names, energy_function, energy_begin=None,
                    energy_end=None, coord_begin=None, coord_end=None):
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

    if (energy_begin and energy_end) is not None:
        ET.SubElement(binning_1, "Begin").text = f'{energy_begin}'
        ET.SubElement(binning_1, "End").text = f'{energy_end}'

    ET.SubElement(binning_1, "Interval").text = f'{energy_interval}'

    binning_2 = ET.SubElement(binnings, "Binning", name="Reaction coordinate")

    if (coord_begin and coord_end) is not None:
        ET.SubElement(binning_2, "Begin").text = f'{coord_begin}'
        ET.SubElement(binning_2, "End").text = f'{coord_end}'

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

    euclidean_distance = atoms.get_distance(indexes[0], indexes[1], mic=True)

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

        euclidean_distance = self.get_distance(indexes[0], indexes[1],
                                               mic=True)

        assert euclidean_distance > 0

        return euclidean_distance


class DFTBUmbrellaCalculator(DFTB):
    """Incomplete implementation of DFTB umbrella sampling calculator"""

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
    """
    Custom Calculator which modifies the energy and force according to
    a harmonic bias along a specified reaction coordinate
    """

    implemented_properties = ["energy", "forces"]

    def _calculate_bias(self, atoms):

        if self.coord_type == 'distance':
            coord_derivative = _get_distance_derivative(atoms, self.coordinate,
                                                        self.reference)

        if self.coord_type == 'rmsd':
            return NotImplementedError

        if self.coord_type == 'torsion':
            return NotImplementedError

        bias = -0.5 * self.spring_const * coord_derivative

        return bias

    # Do I need to modify the energy?
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
                 spring_const=None, reference=None, **kwargs):
        Calculator.__init__(self, restart=None,
                            label=None, atoms=None, **kwargs)

        self.calculator = gap_calc
        self.coord_type = coord_type
        self.coordinate = coordinate
        self.spring_const = spring_const
        self.reference = reference

        assert coord_type in ['distance', 'rmsd', 'torsion']
        assert coordinate is not None
        assert spring_const is not None
        assert reference is not None


class UmbrellaSampling:
    """
    Umbrella sampling class for generating pulling simulation, running
    umbrella sampling in windows and running WHAM analysis using PyWham
    """

    def generate_pulling_configs(self, temp=None, dt=None, interval=None,
                                 pulling_rate=None, final_value=None, **kwargs):
        """Generates an MD trajectory along a reaction coordinate

        :param temp: (float) Temperature in K to run pulling simulation and
                     umbrella sampling

        :param dt: (float) Timestep in fs

        :param interval: (int) Interval between printing the geometry

        :param pulling_rate: (float) Rate of pulling in Å / fs

        :param final_value: (float) Final separation of system in Å

        :param kwargs: {fs, ps, ns} Simulation time in some units
        """

        self.pulling_rate = pulling_rate
        distance = final_value - self.reference
        self.simulation_time = distance / self.pulling_rate

        if self.simulation_time < 100:
            logger.warning(f'Simulation time = {self.simulation_time} < 100 fs!'
                           f' Decrease pulling rate or increase distance'
                           f' to increase simulation time')

        logger.info(f'Running pulling simulation for'
                    f' {self.simulation_time:.0f} fs')

        traj = run_umbrella_gapmd(configuration=self.init_config,
                                  umbrella_gap=self.umbrella_gap,
                                  temp=temp,
                                  dt=dt,
                                  interval=interval,
                                  distance=distance,
                                  pulling_rate=pulling_rate,
                                  **kwargs)

        traj.save('pulling_traj.xyz')

        return traj

    def run_umbrella_sampling(self, traj, temp, dt, interval, num_windows,
                              **kwargs):
        """
        Performs umbrella sampling under a harmonic bias in windows
        generated from the pulling trajectory

        :param num_windows: (int) Number of umbrella sampling windows to run
        """

        self.num_windows = num_windows
        umbrella_frames = Data()
        frame_num = len(traj) // self.num_windows
        [umbrella_frames.add(frame) for frame in traj[::frame_num]]

        # paralellise this
        for window, frame in enumerate(umbrella_frames):

            # assert self.simulation_time is not None
            # window_ref = self.reference + (window * self.simulation_time *
            #                                self.pulling_rate / self.num_windows)

            # Think if this method is valid
            window_atoms = frame.ase_atoms()
            window_ref = window_atoms.get_distance(self.coordinate[0],
                                                   self.coordinate[1],
                                                   mic=True)

            logger.info(f'Running umbrella sampling')
            logger.info(f'Window {window} with reference {window_ref:.2f} Å')

            traj = run_umbrella_gapmd(configuration=frame,
                                      umbrella_gap=self.umbrella_gap,
                                      temp=temp,
                                      dt=dt,
                                      interval=interval,
                                      reference=window_ref,
                                      **kwargs)

            traj.save(f'window_{window}_traj.xyz')

            # decorator to have input files made elsewhere?
            # if it doesn't make enough files it may just print default energy
            # and reference 0.04914365296360979 for the energy
            # Also need to check how I calculate the US potential energy above
            with open(f'window_{window}.txt', 'w') as outfile:
                for configuration in traj:
                    print(f'{configuration.energy}',
                          f'{configuration.rxn_coord}', file=outfile)

        return None

    def run_wham_analysis(self, temp, energy_interval, coord_interval,
                          temp_interval, energy_function):
        """Calculates the Gibbs free energy using the WHAM method"""

        file_list = [f'window_{i}.txt' for i in range(self.num_windows)]
        temps = [temp for _ in range(self.num_windows)]

        # Might want to input being and end values
        _write_xml_file(energy_interval=energy_interval, energy_begin=None,
                        energy_end=None, coord_begin=None, coord_end=None,
                        coord_interval=coord_interval, temp=temps,
                        temp_interval=temp_interval, file_names=file_list,
                        energy_function=energy_function)

        os.system("python2 wham.py wham.spec.xml")

        return None

    def __init__(self, init_config=None, gap=None, coordinate=None,
                 spring_const=None):
        """
        :param init_config: (gaptrain.configurations.Configuration)

        :param gap: (gaptrain.gap.GAP)

        :param coordinate: (list | None) Indices of the atoms which define the
                           reaction coordinate

        :param spring_const: (float | None) Value of the spring constant, K,
                              used in umbrella sampling
        """

        if len(coordinate) == 2:
            self.coord_type = 'distance'
            ase_atoms = init_config.ase_atoms()
            self.reference = ase_atoms.get_distance(coordinate[0],
                                                    coordinate[1], mic=True)
        elif len(coordinate) == 4:
            self.coord_type = 'torsion'
        elif isinstance(coordinate, gaptrain.configurations.Configuration):
            self.coord_type = 'rmsd'
        else:
            raise ValueError("Coordinate type could not be inferred from "
                             "coordinates")

        self.umbrella_gap = UmbrellaGAP(name=gap.name,
                                        system=gap.system,
                                        coord_type=self.coord_type,
                                        coordinate=coordinate,
                                        spring_const=spring_const,
                                        reference=self.reference)

        self.init_config = init_config
        self.gap = gap
        self.coordinate = coordinate
        self.final_value = None
        self.pulling_rate = None
        self.num_windows = None
        self.simulation_time = None
