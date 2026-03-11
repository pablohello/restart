# The ASI part of the code is modified from ASI package's examples.
# For details please refer to the documentation: https://pvst.gitlab.io/asi/asi_8h.html
import sys, os
import numpy as np
from asi4py.asecalc import ASI_ASE_calculator
from ase.io import read
from ase.calculators.aims import Aims
sys.path.append(os.environ["DEEPH_INTERFACE_PATH"])
from aims2DeepH import aims_get_data

deeph_output_dir = "preprocessed" # Output directory of DeepH's preprocessed files
atoms = read("geometry.in") # The structure to be computed
work_dir = "asi.temp" # Default work directory of ASI_ASE_calculator (no need to change)
logfile = "asi.log" # Default output file of ASI_ASE_calculator (no need to change)

def init_via_ase(asi):
  calc = Aims(xc='pbe',
    relativistic="atomic_zora scalar",
    occupation_type="gaussian 0.010",
    density_update_method="density_matrix",
    species_dir = os.environ['AIMS_SPECIES_PATH'],
    output = ['h_s_matrices', "k_point_list"], # Mandatory for the DeepH interface
    k_grid = (7, 7, 1)
  )
  calc.write_input(asi.atoms)

# read path to ASI-implementing shared library from environment variable
AIMS_LIB_PATH = os.environ["AIMS_LIB_PATH"]

# initialize ASI library via ASE calculators
atoms.calc = ASI_ASE_calculator(AIMS_LIB_PATH, init_via_ase, None, atoms, work_dir=work_dir, logfile=logfile)

# Ask to save Hamiltonian and overlap matrices
atoms.calc.asi.keep_hamiltonian = True
atoms.calc.asi.keep_overlap = True
print(f'basis size = {atoms.calc.asi.n_basis}')
print(f'E = {atoms.get_potential_energy():.6f} eV') # actual calculation

aims_get_data(atoms, work_dir, logfile, deeph_output_dir) # The DeepH interface