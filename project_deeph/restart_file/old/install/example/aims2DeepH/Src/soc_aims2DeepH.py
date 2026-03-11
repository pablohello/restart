import numpy as np
from numpy import pi
import os
import h5py
import json
from ase import Atoms

def modified_aims_get_data(ase_object,
                           aims_output_dir="asi.temp",
                           asi_std_out_f="asi.log",
                           output_dir="preprocessed_nc",
                           n_spin=2,
                           n_kpts=1):
    os.makedirs(output_dir, exist_ok=True)
    Hartree2eV = 27.2113845

    # --- Подготовка кристалла (лат, позиции, инициализация ASE Atoms) ---
    if not ase_object.pbc.any():
        np.savetxt(os.path.join(output_dir, "lat.dat"),
                   np.array([[5.2917721e8,0,0],[0,5.2917721e8,0],[0,0,5.2917721e8]]))
        atoms = Atoms(symbols=ase_object.get_chemical_symbols(),
                      positions=ase_object.get_positions(), pbc=False)
    else:
        np.savetxt(os.path.join(output_dir, "lat.dat"), np.transpose(ase_object.cell))
        cell = ase_object.get_cell()
        scaled_positions = ase_object.get_scaled_positions()
        for ia in range(len(scaled_positions)):
            for ix in range(3):
                if scaled_positions[ia,ix] > 0.5:
                    scaled_positions[ia,ix] -= 1
        atoms = Atoms(symbols=ase_object.get_chemical_symbols(),
                      scaled_positions=scaled_positions, cell=cell, pbc=True)

    np.savetxt(os.path.join(output_dir, "site_positions.dat"), np.transpose(atoms.get_positions()))
    np.savetxt(os.path.join(output_dir, "element.dat"), atoms.get_atomic_numbers(), fmt="%d")

    # --- Определение орбиталей, сортировки и фаз ---
    basis_indices = np.genfromtxt(os.path.join(aims_output_dir, "basis-indices.out"))[1:]
    orbital_types = {}
    for io in range(len(basis_indices)):
        if basis_indices[io,5] == 0:
            ia = int(basis_indices[io,2])
            if ia not in orbital_types.keys():
                orbital_types[ia] = []
            orbital_types[ia].append(int(basis_indices[io,4]))

    site_norbits = []
    m_sort_list = {0:[0], 1:[2,0,1], 2:[2,4,0,3,1], 3:[3,4,2,5,1,6,0]}
    m_phase_list = {0:[1], 1:[1,1,-1], 2:[1,1,1,-1,1], 3:[1,1,1,1,-1,1,-1]}
    sort_table = []
    phase_table = []
    for ia in range(1, len(atoms.get_atomic_numbers())+1):
        site_norbits.append(sum(orbital_types[ia])*2+len(orbital_types[ia]))
        cur_sort_table = []
        cur_phase_table = []
        for l in orbital_types[ia]:
            if l not in m_sort_list.keys():
                print("l={} bases not supported yet!".format(l))
                exit()
            cur_length = len(cur_sort_table)
            cur_sort_table.extend([cur_length+m_order for m_order in m_sort_list[l]])
            cur_phase_table.extend([m_order for m_order in m_phase_list[l]])
        sort_table.append(cur_sort_table)
        phase_table.append(cur_phase_table)
    site_norbits_cumsum = np.cumsum(site_norbits)

    with open(os.path.join(output_dir, "orbital_types.dat"), 'w') as f_orb_type:
        for ia in range(1, len(atoms.get_atomic_numbers())+1):
            for l in orbital_types[ia]:
                f_orb_type.write("{} ".format(l))
            f_orb_type.write("\n")

    # --- Fermi, k-points, и радиусы из лога ---
    k_list = np.zeros((n_kpts, 4))
    species_list = set(atoms.get_chemical_symbols())
    rc_list = []
    element_label = None
    found_table = False
    rc = []
    chemical_potential = 0.0
    with open(os.path.join(aims_output_dir, asi_std_out_f), 'r') as f_log:
        for line in f_log:
            if "| Chemical potential (Fermi level):" in line:
                chemical_potential = float(line.strip().split()[-2])
            if "| k-point:" in line:
                k_idx = int(line.strip().split()[2]) - 1
                k_list[k_idx, :] = np.array([line.strip().split()[i] for i in [4, 5, 6, 9]], dtype=np.float64)
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

    species_max_rc = {}
    for rc in rc_list:
        if rc['element_label'] not in species_max_rc.keys():
            species_max_rc[rc['element_label']] = max(rc['rc'])
        else:
            species_max_rc[rc['element_label']] = max(max(rc['rc']), species_max_rc[rc['element_label']])
    max_rc = 0.0
    for element in species_max_rc.keys():
        print("Cutoff radius of {} is set to {} Angstrom".format(element, species_max_rc[element]))
        max_rc = max(species_max_rc[element], max_rc)

    if not atoms.pbc.any():
        k_list = np.array([[0.0, 0.0, 0.0, 1.0]])

    # --- Инициализация H_k и S_k для спиноров ---
    N = int(site_norbits_cumsum[-1])
    H_k = np.zeros((n_kpts, N, N, 2, 2), dtype=np.complex128)
    S_k = np.zeros((n_kpts, N, N), dtype=np.complex128)
    f_up   = os.path.join(aims_output_dir, "hamiltonian_up.out")
    f_dn   = os.path.join(aims_output_dir, "hamiltonian_dn.out")
    f_updn = os.path.join(aims_output_dir, "hamiltonian_up_down.out")
    f_dnup = os.path.join(aims_output_dir, "hamiltonian_dn_up.out")
    f_s    = os.path.join(aims_output_dir, "overlap-matrix.out")

    for f in (f_up, f_dn, f_s):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file {f}")
    exist_updn = os.path.exists(f_updn)
    exist_dnup = os.path.exists(f_dnup)

    for ik in range(n_kpts):
        S_k[ik] = 0
        H_k[ik, :, :, :, :] = 0
        def read_block(path, block_idx):
            data = np.zeros((N,N), dtype=np.complex128)
            with open(path) as f:
                for ln in f:
                    if ln.strip():
                        i, j, v = ln.split()
                        i,j = int(i)-1, int(j)-1
                        data[i,j] = float(v)
                        data[j,i] = float(v)
            return data

        H_up   = read_block(f_up,   None)
        H_dn   = read_block(f_dn,   None)
        H_updn = read_block(f_updn, None) if exist_updn else np.zeros((N,N), dtype=np.complex128)
        H_dnup = read_block(f_dnup, None) if exist_dnup else np.zeros((N,N), dtype=np.complex128)
        H_k[ik,:,:,0,0] = H_up
        H_k[ik,:,:,1,1] = H_dn
        H_k[ik,:,:,0,1] = H_updn
        H_k[ik,:,:,1,0] = H_dnup

        with open(f_s) as f:
            for ln in f:
                if ln.strip():
                    i,j,v = ln.split()
                    i,j = int(i)-1, int(j)-1
                    S_k[ik,i,j] = float(v)
                    S_k[ik,j,i] = float(v)

    # Определение пар атомов
    R_list = []
    atom_pair_list = []
    if not atoms.pbc.any():
        nRmax = [0, 0, 0]
    else:
        nRmax = [1, 1, 1]
    for (Rx, Ry, Rz) in [(Rx, Ry, Rz) for Rx in range(-nRmax[0], nRmax[0]+1)
                         for Ry in range(-nRmax[1], nRmax[1]+1)
                         for Rz in range(-nRmax[2], nRmax[2]+1)]:
        for (iatom, jatom) in [(ia, ja) for ia in range(len(orbital_types))
                               for ja in range(len(orbital_types))]:
            distance = np.linalg.norm(atoms.get_positions()[jatom] - atoms.get_positions()[iatom] + np.array([Rx, Ry, Rz])@atoms.cell)
            if distance < species_max_rc[atoms.get_chemical_symbols()[iatom]] + species_max_rc[atoms.get_chemical_symbols()[jatom]]:
                atom_pair_list.append("[{}, {}, {}, {}, {}]".format(Rx, Ry, Rz, iatom+1, jatom+1))
                if "[{}, {}, {}]".format(Rx, Ry, Rz) not in R_list:
                    R_list.append("[{}, {}, {}]".format(Rx, Ry, Rz))

    # Обратное Фурье-преобразование
    def A_k_to_A_R_spinor(A_k, k_list, R_list):
        # A_k: [n_kpts, N, N, 2, 2]
        A_R = {R: np.zeros((N,N,2,2), dtype=np.float64) for R in R_list}
        for R in R_list:
            vecR = np.array(eval(R))
            for ik in range(len(k_list)):
                phase = np.exp(-1j * 2*pi * np.dot(vecR, k_list[ik,:3])) * k_list[ik,3]
                A_R[R] += np.real(A_k[ik] * phase)
        return A_R

    def A_k_to_A_R(A_k_list, k_list, R_list):
        A_R = {}
        for R in R_list:
            A_R[R] = np.zeros_like(A_k_list[0], dtype=np.float64)
            for ik in range(len(k_list)):
                A_R[R] += np.real(A_k_list[ik] * np.exp(-1j * 2 * pi * np.dot(np.array(eval(R)), k_list[ik][:3])) * k_list[ik][3])
        return A_R

    H_R_spinor = A_k_to_A_R_spinor(H_k, k_list, R_list)
    S_R = A_k_to_A_R(S_k, k_list, R_list)

    # Срезка и перестановка (неколлинеарный случай!)
    H_deeph = {'up_up':{}, 'up_dn':{}, 'dn_up':{}, 'dn_dn':{}}
    S_deeph = {}
    for atom_pair in atom_pair_list:
        eval_key = eval(atom_pair)
        matrix_slice_i = slice(site_norbits_cumsum[eval_key[3]-1] - site_norbits[eval_key[3]-1], site_norbits_cumsum[eval_key[3]-1])
        matrix_slice_j = slice(site_norbits_cumsum[eval_key[4]-1] - site_norbits[eval_key[4]-1], site_norbits_cumsum[eval_key[4]-1])
        parity_i = np.array(phase_table[eval_key[3]-1])
        parity_j = np.array(phase_table[eval_key[4]-1])
        sort_i = np.array(sort_table[eval_key[3]-1])
        sort_j = np.array(sort_table[eval_key[4]-1])

        cur_H = H_R_spinor["[{}, {}, {}]".format(eval_key[0], eval_key[1], eval_key[2])][matrix_slice_i, matrix_slice_j, :, :]  # (ni, nj, 2, 2)
        for s1, s2, spin_lbl in [(0,0,'up_up'), (0,1,'up_dn'), (1,0,'dn_up'), (1,1,'dn_dn')]:
            sub = cur_H[:,:,s1,s2].copy()
            sub *= parity_i[:, None]
            sub *= parity_j[None, :]
            sub = sub[sort_i, :][:, sort_j]
            H_deeph[spin_lbl][atom_pair] = sub * Hartree2eV

        cur_S = S_R["[{}, {}, {}]".format(eval_key[0], eval_key[1], eval_key[2])][matrix_slice_i, matrix_slice_j]
        cur_S *= parity_i[:, np.newaxis]
        cur_S *= parity_j[np.newaxis, :]
        cur_S = cur_S[sort_i, :][:, sort_j]
        S_deeph[atom_pair] = cur_S

    # Вывод в .h5
    with h5py.File(os.path.join(output_dir, "hamiltonians.h5"), 'w') as f_h:
        for spin_lbl in H_deeph:
            for key in H_deeph[spin_lbl].keys():
                f_h[f"{key}_{spin_lbl}"] = H_deeph[spin_lbl][key]

    with h5py.File(os.path.join(output_dir, "overlaps.h5"), 'w') as f_s:
        for key in S_deeph.keys():
            f_s[key] = S_deeph[key]

    info = {
        'nsites': len(orbital_types),
        'isorthogonal': False,
        'isspinful': True,
        'norbits': int(site_norbits_cumsum[-1]),
        'fermi_level': chemical_potential
    }
    with open(os.path.join(output_dir, "info.json"), 'w') as info_j:
        json.dump(info, info_j, indent=4)
    with h5py.File(hamil_file, 'r') as f_h:
    for k in f_h.keys():
        print(f"Processing key: {k}")
        k_clean = k.split('_')[0]
        try:
            key = json.loads(k_clean)
        except json.decoder.JSONDecodeError as e:
            print(f"Error parsing key: {k_clean}, original: {k}, error: {e}")
            raise
    print("Noncollinear data prepared and saved in '{}'.".format(output_dir))

