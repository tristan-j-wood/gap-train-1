import gaptrain as gt
from gaptrain.umbrella import GAPUmbrellaCalculator, DFTBUmbrellaCalculator
from gaptrain.md import run_umbrella_gapmd, run_umbrella_dftbmd
from gaptrain.gap import UmbrellaGAP
from gaptrain import GTConfig
import os
gt.GTConfig.n_cores = 1

here = os.path.abspath(os.path.dirname(__file__))
h2o = gt.Molecule(os.path.join(here, 'data', 'h2o_2.xyz'))

water = gt.System(box_size=[10, 10, 10])
water.add_molecules(h2o)
config = gt.Configuration(system=water)

potential = os.path.join(here, 'data', 'potential_h2o.xml')
gap = gt.gap.GAP(name=potential, system=water)


def test_get_distance_coordinate_bias():
    """ Tests for calculating the derivative of harmonic Euclidean distance."""

    # Choose O atoms as the coordinate, with reference 1 and set new positions
    indexes = [0, 3]
    ref = 1
    new_pos = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [5, 1, 1],
               [1, 1, 1],
               [1, 1, 1]
               ]

    ase_atoms = config.ase_atoms()
    ase_atoms.set_positions(new_pos, apply_constraint=False)

    derivative = gt.umbrella._get_distance_derivative(ase_atoms, indexes, ref)

    # Forces should be equal an opposite (6 in this case) for the two atoms
    assert derivative[indexes[0]][0] == -6 and derivative[indexes[1]][0] == 6
    assert derivative[0][0] == -derivative[indexes[1]][0]

    bias_strength = 1
    bias = -0.5 * bias_strength * derivative

    # shape of forces object should be three for x,y,z
    assert bias.shape[1] == 3

    return None


def test_gapumbrellacalculator():

    gap_umbrella_calc = GAPUmbrellaCalculator()

    assert hasattr(gap_umbrella_calc, 'calculator')
    assert hasattr(gap_umbrella_calc, 'coordinate')
    assert hasattr(gap_umbrella_calc, 'spring_const')
    assert hasattr(gap_umbrella_calc, 'reference')

    return None


def test_dftbumbrellacalculator():

    return NotImplementedError


def test_run_umbrella_gapmd():

    return NotImplementedError


def test_run_umbrella_dftbmd():

    return NotImplementedError
