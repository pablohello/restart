# DeepH@FHI-aims interface

The README file describes the interface of [DeepH-pack](https://github.com/mzjb/DeepH-pack) with FHI-aims.

## Introduction 
DeepH is a deep-learning method for the neural-network modeling of DFT electronic Hamiltonians. Starting with this version, DeepH can be interfaced with FHI-aims through the [Atomic Simulation Interface](https://fhi-aims.org/uploads/documents/du-2023/Stishenko-pres2.pdf) (ASI) package, which faciliates the reading of Hamiltonian and overlap matrices.  The interface is now compatible with both clusters and periodic systems.

According to our primary tests, [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3),a variant of DeepH series of methods, achieves meV-level accuracy accross multiple test datasets computed with FHI-aims. 

## Requirements
In addition to the dependencies required for [DeepH-pack](https://github.com/mzjb/DeepH-pack), using the DeepH@FHI-aims interface neccessitates some additional setups:

### Compiling FHI-aims as a shared library
To comply with the ASI package requirements, a shared library file of FHI-aims (`libaims.*.so`) is essential. You can achieve this by adding the line `set(BUILD_SHARED_LIBS ON CACHE STRING "")` to the cmake configuration file of FHI-aims, followed by recompiling the FHI-aims software.

### Installation of the ASI module
You can install the asi4py easily with the command `pip install asi4py`. For further instructions, please refer to the documentation for the [ASI package](https://pvst.gitlab.io/asi/testing.html).

## Usage
To use the interface, the FHI-aims must be executed with the ASI package's calculator (`asi4py.asecalc.ASI_ASE_calculator`) to access the Hamiltonian and overlap matrices. Utilizing the ASI calculator, the DeepH interface can preprocess the data to extract all relevant information for DeepH training. The preprocessed data can then be integrated with [DeepH](https://github.com/mzjb/DeepH-pack) or [DeepH-E3](https://github.com/Xiaoxun-Gong/DeepH-E3) for further training.

Two examples (located in `Examples/Cluster` and `Examples/Periodic`) illustrate how the interface operates for cluster and periodic systems, respectively. Please refer to the Examples for additional information. 

## Examples

### Setting up the environment
Before running any examples, you need to specify some environment variables listed in `Examples/setup_environ.sh`, which includes:
- `AIMS_LIB_PATH`: The path to the shared library file of FHI-aims.
- `AIMS_SPECIES_PATH`: The path to the FHI-aims species file you prefer to include. Please note that g-orbitals are not yet supported and are generally not favored for DeepH due to efficiency concerns.
- `DEEPH_INTERFACE_PATH`: The path to the `Src` subdirectory of the deeph_interface's.

Additionally, please ensure that your MPI environments are loaded to run FHI-aims.

### Clusters
An example of using the interface for clusters (molecules) is provided in `Examples/Cluster/run.py`. The script reads the structure to be computed with `ase.io.read` (line 12) and sets up an ASI_ASE calulator with the configurations specified in lines 16 to 23. The DFT computation is invoked at line 36, and a call to the DeepH interface (line 38) dumps the computation results to `deeph_output_dir`.

Note that for clusters, specifying `output = ['h_s_matrices']` (in line 22) is mandatory for the interface.

### Periodic systems
An example of using the interface for periodic systems is provided in `Examples/Periodic/run.py`. The script is quite similar to that for cluster systems, with two key differences:

1. An additional output of `"k_point_list"` is also required (in line 22).
2. The k-mesh must be specified (in line 23)

## References
1. DeepH: [Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation](https://www.nature.com/articles/s43588-022-00265-6)
2. DeepH-E3: [General framework for E(3)-equivariant neural network representation of density functional theory Hamiltonian](https://www.nature.com/articles/s41467-023-38468-8)
3. ASI: [Atomic Simulation Interface (ASI): application programming interface for electronic structure codes](https://joss.theoj.org/papers/10.21105/joss.05186)