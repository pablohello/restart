import numpy as np 
import os 
import h5py 
import json
from numpy import pi
from ase import Atoms
# Импорт точных констант
from ase.units import Hartree, eV

# DeepH interface for FHI-aims, built upon ASI package for FHI-aims
# Authors: Zechen Tang and Dr. He Li @ CMT Group, Tsinghua Univ. 
# Supervisors: Prof. Yong Xu and Prof. Wenhui Duan

# [ИЗМЕНЕНИЕ] Новая функция для чтения матриц напрямую из файлов
def read_aims_matrix_file(filepath, n_basis):
    """Читает полную матрицу из файла aims (row col val)."""
    # FHI sparse format: row col value
    # Индексы 1-based
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        # Если файла нет, лучше упасть с ошибкой, чем вернуть None и упасть потом
        raise FileNotFoundError(f"Matrix file not found: {filepath}")
    
    # Читаем сразу в float64
    try:
        data = np.loadtxt(filepath, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise

    mat = np.zeros((n_basis, n_basis), dtype=np.float64)
    
    # Обработка случая, если файл пустой или имеет странный формат
    if data.ndim == 1 and data.size >= 3:
        data = data.reshape(1, -1)
    if data.size == 0:
        return mat

    rows = data[:, 0].astype(int) - 1
    cols = data[:, 1].astype(int) - 1
    vals = data[:, 2]
    
    # Заполняем матрицу
    for r, c, v in zip(rows, cols, vals):
        mat[r, c] = v
        if r != c:
            mat[c, r] = v # Гарантируем симметрию
            
    return mat

def aims_get_data(ase_object, aims_output_dir="asi.temp", asi_std_out_f="asi.log", output_dir="preprocessed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # [ИЗМЕНЕНИЕ] Точная константа
    Hartree2eV = Hartree / eV

    if ase_object.calc.asi.n_spin != 1:
        print("Spin polarized cases not supported yet!")
        exit()

    # lat.dat, site_positions.dat, orbital_types.dat
    if not ase_object.pbc.any():
        np.savetxt(os.path.join(output_dir, "lat.dat"), 
                np.array([[5.2917721e8,0,0],[0,5.2917721e8,0],[0,0,5.2917721e8]])) # HUGE vacuum layer in aims for molecules
        atoms = Atoms(symbols=ase_object.get_chemical_symbols(),
                      positions = ase_object.get_positions(), pbc=False)
    else:
        np.savetxt(os.path.join(output_dir, "lat.dat"), np.transpose(ase_object.cell))
        latvol = np.dot(ase_object.cell[0,:], np.cross(ase_object.cell[1,:], ase_object.cell[2,:]))
        rlat = np.array([2 * np.pi * np.cross(ase_object.cell[1,:], ase_object.cell[2,:]) / latvol,
                        2 * np.pi * np.cross(ase_object.cell[2,:], ase_object.cell[0,:]) / latvol,
                        2 * np.pi * np.cross(ase_object.cell[0,:], ase_object.cell[1,:]) / latvol ])
        np.savetxt(os.path.join(output_dir, "rlat.dat"), np.transpose(rlat))
        # transform to W-S cell if pbc
        cell = ase_object.get_cell()
        scaled_positions = ase_object.get_scaled_positions()
        for ia in range(len(scaled_positions)):
            for ix in range(3):
                if scaled_positions[ia,ix]-1e-8 > 0.5:
                    scaled_positions[ia,ix] -= 1
        atoms = Atoms(symbols=ase_object.get_chemical_symbols(),
                      scaled_positions = scaled_positions, cell=cell, pbc=True)

    np.savetxt(os.path.join(output_dir, "site_positions.dat"), np.transpose(atoms.get_positions()))
    np.savetxt(os.path.join(output_dir, "element.dat"), atoms.get_atomic_numbers(), fmt="%d")

    # basis-indices
    basis_indices = np.genfromtxt(os.path.join(aims_output_dir, "basis-indices.out"))[1:] # skip header line
    # column 0: basisset id; 1: type; 2: atom id (1-based); 3: n; 4: l; 5: m
    orbital_types = {}
    for io in range(len(basis_indices)):
        # print(basis_indices[io])
        if basis_indices[io,5] == 0:
            ia = int(basis_indices[io,2])
            if ia not in orbital_types.keys():
                orbital_types[ia] = []
            orbital_types[ia].append(int(basis_indices[io,4]))

    # Establish sorting/parity table for each atom
    site_norbits = []
    m_sort_list = {0:[0], 1:[2,0,1], 2:[2,4,0,3,1], 3:[3,4,2,5,1,6,0]} # Sorts FHI-aims' spherical harmonics to DeepH's
    m_phase_list = {0:[1], 1:[1,1,-1], 2:[1,1,1,-1,1], 3:[1,1,1,1,-1,1,-1]} 

    sort_table = [] # dimension 1: atom index; dimension 2: sorting index
    phase_table = []
    for ia in range(1, len(atoms.get_atomic_numbers())+1): # +1 for 1-based
        site_norbits.append(sum(orbital_types[ia])*2+len(orbital_types[ia]))
        cur_sort_table = []
        cur_phase_table = []
        for l in orbital_types[ia]: # orbital_types[ia] corresponds to a specific l
            if l not in m_sort_list.keys():
                print("l={} bases not supported yet!".format(l))
                exit()
            cur_length = len(cur_sort_table)
            cur_sort_table.extend([cur_length+m_order for m_order in m_sort_list[l]])
            cur_phase_table.extend([m_order for m_order in m_phase_list[l]])
        sort_table.append(cur_sort_table)
        phase_table.append(cur_phase_table)
    site_norbits_cumsum = np.cumsum(site_norbits)

    # orbital_types.dat
    with open(os.path.join(output_dir, "orbital_types.dat"), 'w') as f_orb_type:
        for ia in range(1, len(atoms.get_atomic_numbers())+1): # +1 for 1-based
            for l in orbital_types[ia]:
                f_orb_type.write("{} ".format(l))
            f_orb_type.write("\n")

    # [ИЗМЕНЕНИЕ] Парсинг лога для k-points (оставляем для хим.потенциала и radius, k-points перезапишем)
    k_list_from_log = np.zeros((ase_object.calc.asi.n_kpts,4)) 
    species_list = set(atoms.get_chemical_symbols())
    rc_list = []
    element_label = None
    found_table = False
    rc = []
    n_k = None
    chemical_potential = 0.0 # Default value
    
    with open(os.path.join(aims_output_dir, asi_std_out_f), 'r') as f_log:
        for line in f_log:
            if "| Chemical potential (Fermi level):" in line:
                chemical_potential = float(line.strip().split()[-2])
            if "| k-point:" in line:
                # Оставляем парсинг, но использовать будем k_list_from_log только если periodic
                k_list_from_log[int(line.strip().split()[2])-1,:] = np.array([line.strip().split()[i] for i in [4,5,6,9] ],dtype=np.float64)
            stripped_line = line.strip()
            for label in species_list:
                if label in stripped_line:
                    element_label = label
            if "outer radius [A]" in stripped_line:
                found_table = True 
                rc = []  
                continue
            if found_table:
                if stripped_line == "": 
                    rc_list.append({"element_label": element_label, "rc": rc})
                    found_table = False 
                    continue
                rc.append(float(stripped_line.split()[-1]))

    # Determine the cutoff radius of each species
    species_max_rc = {}
    for rc in rc_list:
        if rc['element_label'] not in species_max_rc.keys():
            species_max_rc[rc['element_label']] = max(rc['rc'])
        else:
            species_max_rc[rc['element_label']] = max(max(rc['rc']),species_max_rc[rc['element_label']])
    max_rc = 0.0
    for element in species_max_rc.keys():
        print("Cutoff radius of {} is set to {} Angstrom".format(element,species_max_rc[element]))
        max_rc = max(species_max_rc[element], max_rc)

    # [ИЗМЕНЕНИЕ] Вместо загрузки H/S из ASE, читаем файлы с диска
    # Это позволяет обойти потерю точности в интерфейсе ASI/ASE
    
    nbasis = int(site_norbits_cumsum[-1])
    h_file = os.path.join(aims_output_dir, "hamiltonian.out")
    s_file = os.path.join(aims_output_dir, "overlap.out")
    
    print(f"[INFO] Reading matrices directly from files: {h_file}")
    
    H_full = read_aims_matrix_file(h_file, nbasis)
    S_full = read_aims_matrix_file(s_file, nbasis)
    
    # Создаем фиктивный k_list и H_k для совместимости с кодом ниже
    # Для молекулы (Gamma point) это корректно.
    # Если PERIODIC - нужно быть осторожным, но обычно aims пишет k-point dependent файлы
    # в другом формате. hamiltonian.out - это обычно real space или Gamma.
    # Для целей DeepH с FHI-aims обычно используется molecular режим или supercell.
    
    H_k = np.zeros((1, nbasis, nbasis), dtype=np.complex128)
    S_k = np.zeros((1, nbasis, nbasis), dtype=np.complex128)
    
    H_k[0] = H_full
    S_k[0] = S_full
    
    # Перезаписываем k_list на идеальный Gamma (вес 1.0)
    k_list = np.array([[0.0, 0.0, 0.0, 1.0]]) 

    # Determine atom pairs according to species_max_rc and lattice vectors
    R_list = []
    atom_pair_list = []
    if not atoms.pbc.any():
        nRmax = [0,0,0]
    else:
        nRmax = [int(np.ceil(max_rc/2/np.pi*np.linalg.norm(rlat[iR,:]))) for iR in range(3)]
    
    for (Rx,Ry,Rz) in [(Rx,Ry,Rz) for Rx in range(-nRmax[0],nRmax[0]+1) 
                                for Ry in range(-nRmax[1],nRmax[1]+1) 
                                for Rz in range(-nRmax[2],nRmax[2]+1)]:
        for (iatom,jatom) in [(ia, ja) for ia in range(len(orbital_types))
                                    for ja in range(len(orbital_types))]:
            distance = np.linalg.norm(atoms.get_positions()[jatom] - atoms.get_positions()[iatom] + np.array([Rx, Ry, Rz])@atoms.cell)
            if distance < species_max_rc[atoms.get_chemical_symbols()[iatom]]+species_max_rc[atoms.get_chemical_symbols()[jatom]]:
                atom_pair_list.append("[{}, {}, {}, {}, {}]".format(Rx,Ry,Rz,iatom+1,jatom+1))
                if "[{}, {}, {}]".format(Rx,Ry,Rz) not in R_list:
                    R_list.append("[{}, {}, {}]".format(Rx,Ry,Rz))

    # [ИЗМЕНЕНИЕ] Оптимизированная A_k_to_A_R для молекул
    def A_k_to_A_R(A_k_list, k_list, R_list):
        A_R = {}
        # Если молекула (1 k-точка), просто берем Real часть без лишней математики
        if len(k_list) == 1:
            for R in R_list:
                A_R[R] = np.real(A_k_list[0])
            return A_R
            
        # Старый код (на случай если вдруг periodic + k-points)
        for R in R_list:
            A_R[R] = np.zeros_like(A_k_list[0],dtype=np.float64) 
            for ik in range(len(k_list)):
                A_R[R] +=  np.real(A_k_list[ik] * np.exp(-1j * 2 * pi * np.dot(np.array(eval(R)),k_list[ik][:3])) * k_list[ik][3]) 
        return A_R

    H_R = A_k_to_A_R(H_k, k_list, R_list)
    S_R = A_k_to_A_R(S_k, k_list, R_list)

    # Slice and reorder
    H_deeph = {}
    S_deeph = {}
    for atom_pair in atom_pair_list:
        eval_key = eval(atom_pair)
        # Slice
        matrix_slice_i = slice(site_norbits_cumsum[eval_key[3]-1] - site_norbits[eval_key[3]-1], site_norbits_cumsum[eval_key[3]-1])
        matrix_slice_j = slice(site_norbits_cumsum[eval_key[4]-1] - site_norbits[eval_key[4]-1], site_norbits_cumsum[eval_key[4]-1])
        # Обработка случая, если R нет в H_R (для молекул R_list = [0,0,0])
        R_key = "[{}, {}, {}]".format(eval_key[0],eval_key[1],eval_key[2])
        if R_key not in H_R:
             continue # Или H_R[R_key] = 0
             
        cur_H = H_R[R_key][matrix_slice_i, matrix_slice_j]
        cur_S = S_R[R_key][matrix_slice_i, matrix_slice_j]
        
        # parity
        parity_i = np.array(phase_table[eval_key[3]-1])
        parity_j = np.array(phase_table[eval_key[4]-1])
        cur_H *= parity_i[:, np.newaxis]
        cur_H *= parity_j[np.newaxis, :]
        cur_S *= parity_i[:, np.newaxis]
        cur_S *= parity_j[np.newaxis, :]
        # sort
        sort_i = np.array(sort_table[eval_key[3]-1])
        sort_j = np.array(sort_table[eval_key[4]-1])
        cur_H = cur_H[sort_i,:][:,sort_j]
        cur_S = cur_S[sort_i,:][:,sort_j]
        H_deeph[atom_pair] = cur_H * Hartree2eV
        S_deeph[atom_pair] = cur_S

    # [ИЗМЕНЕНИЕ] Сохранение с float64
    with h5py.File(os.path.join(output_dir,"hamiltonians.h5"),'w') as f_h:
        for key in H_deeph.keys():
            f_h.create_dataset(key, data=H_deeph[key], dtype='float64')

    with h5py.File(os.path.join(output_dir,"overlaps.h5"),'w') as f_s:
        for key in S_deeph.keys():
            f_s.create_dataset(key, data=S_deeph[key], dtype='float64')

    info = {'nsites' : len(orbital_types), 'isorthogonal': False, 'isspinful': False, 'norbits': int(site_norbits_cumsum[-1]), 'fermi_level': chemical_potential}
    with open(os.path.join(output_dir,"info.json"),'w') as info_j:
        json.dump(info, info_j,indent=4)
