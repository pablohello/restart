#!/usr/bin/env python3
"""DeepH (new sparse format) -> FHI-aims restart_file (non-spin case).

Input format expected in `hamiltonian.h5` / `overlap.h5`:
  - atom_pairs       : (n_blocks, 2) or (n_blocks, 5)
      * (n,2): (i_atom, j_atom)
      * (n,5): (Rx, Ry, Rz, i_atom, j_atom)
  - chunk_shapes     : (n_blocks, 2)
  - chunk_boundaries : (n_blocks + 1,)
  - entries          : (sum(prod(chunk_shapes)),)

For (n,5) atom_pairs, blocks are filtered by selected `--R` (default 0,0,0).
Only non-spin workflow is implemented.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

K_B_EV_PER_K = 8.617333262e-5
EV_PER_HA = 27.211386245988


@dataclass
class NewInfo:
    atoms_quantity: int
    orbits_quantity: int
    orthogonal_basis: bool
    spinful: bool
    fermi_energy_eV: float
    elements_orbital_map: dict[str, list[int]]


def read_info(info_path: Path) -> NewInfo:
    data = json.loads(info_path.read_text(encoding="utf-8"))
    return NewInfo(
        atoms_quantity=int(data["atoms_quantity"]),
        orbits_quantity=int(data["orbits_quantity"]),
        orthogonal_basis=bool(data["orthogonal_basis"]),
        spinful=bool(data["spinful"]),
        fermi_energy_eV=float(data["fermi_energy_eV"]),
        elements_orbital_map={k: [int(x) for x in v] for k, v in data["elements_orbital_map"].items()},
    )


def parse_poscar_species(poscar_path: Path, expected_atoms: int) -> list[str]:
    lines = [x.strip() for x in poscar_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(lines) < 7:
        raise ValueError("POSCAR is too short")
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError("POSCAR species/count mismatch")

    per_atom: list[str] = []
    for sp, n in zip(species, counts):
        per_atom.extend([sp] * n)

    if len(per_atom) != expected_atoms:
        raise ValueError(
            f"POSCAR atom count mismatch: {len(per_atom)} vs info atoms_quantity={expected_atoms}"
        )
    return per_atom


def orbital_count_from_l_list(l_list: list[int]) -> int:
    return int(sum(2 * int(l) + 1 for l in l_list))


def build_offsets(
    per_atom_species: list[str],
    elements_orbital_map: dict[str, list[int]],
) -> tuple[dict[int, tuple[int, int]], int]:
    offsets: dict[int, tuple[int, int]] = {}
    cursor = 0
    for i, sp in enumerate(per_atom_species, start=1):
        if sp not in elements_orbital_map:
            raise KeyError(f"Element {sp!r} from POSCAR not found in elements_orbital_map")
        n_orb = orbital_count_from_l_list(elements_orbital_map[sp])
        offsets[i] = (cursor, cursor + n_orb)
        cursor += n_orb
    return offsets, cursor


def truncate_states(
    c: np.ndarray,
    evals_eV: np.ndarray,
    occ: np.ndarray,
    n_states_target: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_states_target is None:
        return c, evals_eV, occ
    if n_states_target <= 0 or n_states_target > c.shape[1]:
        raise ValueError(f"--n-states must be in range 1..{c.shape[1]}")
    return c[:, :n_states_target], evals_eV[:n_states_target], occ[:n_states_target]


def build_sort_phase_tables(types_per_site: list[list[int]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build local orbital reordering/parity tables from shell-l lists."""
    m_sort_list = {0: [0], 1: [2, 0, 1], 2: [2, 4, 0, 3, 1], 3: [3, 4, 2, 5, 1, 6, 0]}
    m_phase_list = {0: [1], 1: [1, 1, -1], 2: [1, 1, 1, -1, 1], 3: [1, 1, 1, 1, -1, 1, -1]}

    sort_table: list[np.ndarray] = []
    phase_table: list[np.ndarray] = []
    for shells in types_per_site:
        cur_sort: list[int] = []
        cur_phase: list[int] = []
        for l in shells:
            if l not in m_sort_list:
                raise ValueError(f"l={l} not supported in undo transform")
            cur_len = len(cur_sort)
            cur_sort.extend([cur_len + m for m in m_sort_list[l]])
            cur_phase.extend(m_phase_list[l])
        sort_table.append(np.array(cur_sort, dtype=int))
        phase_table.append(np.array(cur_phase, dtype=np.float64))
    return sort_table, phase_table


def undo_deeph_basis_transform(c: np.ndarray, per_atom_species: list[str], elements_orbital_map: dict[str, list[int]]) -> np.ndarray:
    """Undo local sort/parity transform used in aims->DeepH pipeline."""
    types_per_site = [elements_orbital_map[sp] for sp in per_atom_species]
    sort_table, phase_table = build_sort_phase_tables(types_per_site)

    out = c.copy()
    start = 0
    for sort_idx, phase in zip(sort_table, phase_table):
        n = len(sort_idx)
        stop = start + n
        block_d = out[start:stop, :]

        block_unsorted = np.zeros_like(block_d)
        block_unsorted[sort_idx, :] = block_d
        block_orig = phase[:, None] * block_unsorted
        out[start:stop, :] = block_orig
        start = stop

    if start != c.shape[0]:
        raise ValueError(
            f"orbital types total orbitals ({start}) do not match C rows ({c.shape[0]})"
        )
    return out


def normalize_and_phase_fix(c: np.ndarray, s: np.ndarray | None, phase_fix: bool) -> np.ndarray:
    c_out = c.astype(np.float64, copy=True)
    n_states = c_out.shape[1]

    for ist in range(n_states):
        v = c_out[:, ist]
        nrm2 = float(v @ v) if s is None else float(v @ (s @ v))
        if nrm2 <= 0:
            raise ValueError(f"Invalid norm for eigenvector[{ist}] = {nrm2}")
        c_out[:, ist] = v / np.sqrt(nrm2)

        if phase_fix:
            idx = int(np.argmax(np.abs(c_out[:, ist])))
            if c_out[idx, ist] < 0.0:
                c_out[:, ist] *= -1.0

    return c_out


def build_occupancy(evals_eV: np.ndarray, occ_in: np.ndarray, mode: str, spin_deg: int = 2) -> np.ndarray:
    occ = np.clip(occ_in.astype(np.float64), 0.0, float(spin_deg))
    if mode == "input":
        return occ
    if mode != "aufbau":
        raise ValueError("--occupancy-mode must be input or aufbau")

    n_elec = int(round(float(np.sum(occ))))
    n_elec = min(max(n_elec, 0), spin_deg * len(evals_eV))
    occ_out = np.zeros_like(occ, dtype=np.float64)
    full = n_elec // spin_deg
    rem = n_elec % spin_deg
    if full > 0:
        occ_out[:full] = float(spin_deg)
    if full < len(occ_out) and rem > 0:
        occ_out[full] = float(rem)
    return occ_out


def _read_record(fh, endian: str) -> bytes:
    marker = fh.read(4)
    if len(marker) != 4:
        raise EOFError("Unexpected EOF while reading record marker")
    n = struct.unpack(f"{endian}i", marker)[0]
    payload = fh.read(n)
    end_marker = fh.read(4)
    if len(payload) != n or len(end_marker) != 4:
        raise EOFError("Unexpected EOF while reading record payload")
    n2 = struct.unpack(f"{endian}i", end_marker)[0]
    if n2 != n:
        raise ValueError("Mismatched record markers")
    return payload


def _detect_restart_endian(path: Path) -> str:
    data = path.read_bytes()[:4]
    if len(data) < 4:
        raise ValueError("restart file is too short")
    for endian in ("<", ">"):
        n = struct.unpack(f"{endian}i", data)[0]
        if n == 4:
            return endian
    raise ValueError("Cannot detect restart endian")


def load_restart_binary(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    endian = _detect_restart_endian(path)
    with path.open("rb") as fh:
        n_basis = struct.unpack(f"{endian}i", _read_record(fh, endian))[0]
        n_states = struct.unpack(f"{endian}i", _read_record(fh, endian))[0]
        n_spin = struct.unpack(f"{endian}i", _read_record(fh, endian))[0]

        c = np.zeros((n_basis, n_states), dtype=np.float64)
        for ib in range(n_basis):
            for ist in range(n_states):
                for isp in range(n_spin):
                    val = struct.unpack(f"{endian}d", _read_record(fh, endian))[0]
                    if isp == 0:
                        c[ib, ist] = val

        evals = np.zeros(n_states, dtype=np.float64)
        occ = np.zeros(n_states, dtype=np.float64)
        for ist in range(n_states):
            for isp in range(n_spin):
                ev, oc = struct.unpack(f"{endian}dd", _read_record(fh, endian))
                if isp == 0:
                    evals[ist] = ev
                    occ[ist] = oc
    return c, evals, occ, n_spin


def align_to_reference(c: np.ndarray, c_ref: np.ndarray, s: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    if c.shape != c_ref.shape:
        raise ValueError(f"reference restart shape mismatch: {c_ref.shape} vs {c.shape}")

    ov = c_ref.T @ c if s is None else c_ref.T @ s @ c

    n = c.shape[1]
    perm = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)
    abs_ov = np.abs(ov)
    for i in range(n):
        row = abs_ov[i].copy()
        row[used] = -1.0
        j = int(np.argmax(row))
        perm[i] = j
        used[j] = True

    c2 = c[:, perm]
    diag_ov = np.einsum("ij,ij->j", c_ref, c2) if s is None else np.einsum("ij,ij->j", c_ref, s @ c2)
    signs = np.where(diag_ov < 0.0, -1.0, 1.0)
    c2 = c2 * signs[np.newaxis, :]

    final_ov = np.abs(np.einsum("ij,ij->j", c_ref, c2)) if s is None else np.abs(np.einsum("ij,ij->j", c_ref, s @ c2))
    return c2, perm, float(np.min(final_ov))


def _normalize_pair_indexing(atom_pairs_ij: np.ndarray, n_atoms: int) -> np.ndarray:
    amin = int(atom_pairs_ij.min())
    amax = int(atom_pairs_ij.max())
    if amin >= 1 and amax <= n_atoms:
        return atom_pairs_ij.astype(np.int64)
    if amin >= 0 and amax < n_atoms:
        return atom_pairs_ij.astype(np.int64) + 1
    raise ValueError(f"atom_pairs indexing is not understood: min={amin}, max={amax}, n_atoms={n_atoms}")


def _extract_blocks_for_R(
    atom_pairs: np.ndarray,
    chunk_shapes: np.ndarray,
    chunk_boundaries: np.ndarray,
    entries: np.ndarray,
    r_tuple: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (atom_pairs_ij, chunk_shapes_sel, entries_sel_flat).

    Supports atom_pairs with 2 columns (no R) and 5 columns (Rx,Ry,Rz,i,j).
    """
    if atom_pairs.ndim != 2:
        raise ValueError(f"atom_pairs must be 2D, got {atom_pairs.shape}")

    if atom_pairs.shape[1] == 2:
        if chunk_shapes.shape[0] != atom_pairs.shape[0] or chunk_boundaries.shape[0] != atom_pairs.shape[0] + 1:
            raise ValueError("chunk arrays are inconsistent with atom_pairs (n,2)")
        return atom_pairs.astype(np.int64), chunk_shapes, entries

    if atom_pairs.shape[1] != 5:
        raise ValueError(f"atom_pairs must have shape (n,2) or (n,5), got {atom_pairs.shape}")

    if chunk_shapes.shape[0] != atom_pairs.shape[0] or chunk_boundaries.shape[0] != atom_pairs.shape[0] + 1:
        raise ValueError("chunk arrays are inconsistent with atom_pairs (n,5)")

    mask = np.all(atom_pairs[:, :3] == np.array(r_tuple, dtype=atom_pairs.dtype), axis=1)
    selected = np.where(mask)[0]
    if selected.size == 0:
        raise ValueError(
            f"No blocks found for R={r_tuple}. Try another --R or inspect atom_pairs[:,0:3]."
        )

    atom_pairs_ij = atom_pairs[selected, 3:5].astype(np.int64)
    chunk_shapes_sel = chunk_shapes[selected]

    flat_parts: list[np.ndarray] = []
    for ib in selected:
        left = int(chunk_boundaries[ib])
        right = int(chunk_boundaries[ib + 1])
        flat_parts.append(entries[left:right])
    entries_sel = np.concatenate(flat_parts, axis=0) if flat_parts else np.array([], dtype=entries.dtype)

    return atom_pairs_ij, chunk_shapes_sel, entries_sel


def assemble_from_sparse_chunks(
    h5_path: Path,
    n_orb_total: int,
    n_atoms: int,
    offsets: dict[int, tuple[int, int]],
    hermitize: bool,
    r_tuple: tuple[int, int, int],
    entry_order: str,
) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        for required in ("atom_pairs", "chunk_shapes", "chunk_boundaries", "entries"):
            if required not in f:
                raise KeyError(f"{h5_path}: required dataset '{required}' not found")

        atom_pairs = np.array(f["atom_pairs"][...], dtype=np.int64)
        chunk_shapes = np.array(f["chunk_shapes"][...], dtype=np.int64)
        chunk_boundaries = np.array(f["chunk_boundaries"][...], dtype=np.int64)
        entries = np.array(f["entries"][...], dtype=np.float64)

    if chunk_shapes.ndim != 2 or chunk_shapes.shape[1] != 2:
        raise ValueError(f"chunk_shapes must have shape (n,2), got {chunk_shapes.shape}")

    atom_pairs_ij, chunk_shapes, entries = _extract_blocks_for_R(
        atom_pairs=atom_pairs,
        chunk_shapes=chunk_shapes,
        chunk_boundaries=chunk_boundaries,
        entries=entries,
        r_tuple=r_tuple,
    )

    atom_pairs_ij = _normalize_pair_indexing(atom_pairs_ij, n_atoms)

    m = np.zeros((n_orb_total, n_orb_total), dtype=np.float64)
    entry_cursor = 0
    n_blocks = atom_pairs_ij.shape[0]
    for ib in range(n_blocks):
        i_atom = int(atom_pairs_ij[ib, 0])
        j_atom = int(atom_pairs_ij[ib, 1])

        a0, a1 = offsets[i_atom]
        b0, b1 = offsets[j_atom]

        shape_r, shape_c = int(chunk_shapes[ib, 0]), int(chunk_shapes[ib, 1])
        expected_ij = (a1 - a0, b1 - b0)
        expected_ji = (b1 - b0, a1 - a0)

        swap_ij = False
        if (shape_r, shape_c) == expected_ij:
            swap_ij = False
        elif (shape_r, shape_c) == expected_ji:
            swap_ij = True
        else:
            raise ValueError(
                f"Block {ib} shape mismatch for pair ({i_atom},{j_atom}): "
                f"from file {(shape_r, shape_c)} vs expected {expected_ij} or swapped {expected_ji}"
            )

        block_size = shape_r * shape_c
        block_flat = entries[entry_cursor : entry_cursor + block_size]
        entry_cursor += block_size
        if block_flat.size != block_size:
            raise ValueError(f"Block {ib} entries mismatch: {block_flat.size} vs {block_size}")
        order = "C" if entry_order == "C" else "F"
        block = block_flat.reshape((shape_r, shape_c), order=order)
        if swap_ij:
            block = block.T
        m[a0:a1, b0:b1] = block

    if hermitize:
        m = 0.5 * (m + m.T)
    return m


def choose_entry_order_for_s(
    s_path: Path,
    n_orb_total: int,
    n_atoms: int,
    offsets: dict[int, tuple[int, int]],
    hermitize: bool,
    r_tuple: tuple[int, int, int],
) -> str:
    s_c = assemble_from_sparse_chunks(s_path, n_orb_total, n_atoms, offsets, hermitize, r_tuple, entry_order="C")
    s_f = assemble_from_sparse_chunks(s_path, n_orb_total, n_atoms, offsets, hermitize, r_tuple, entry_order="F")

    def score(m: np.ndarray) -> tuple[float, float]:
        asym = float(np.linalg.norm(m - m.T) / (np.linalg.norm(m) + 1e-30))
        diag_bad = float(np.sum(np.clip(-np.diag(m), 0.0, None)))
        return asym, diag_bad

    sc = score(s_c)
    sf = score(s_f)
    return "C" if sc <= sf else "F"


def solve_generalized(h: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    l = np.linalg.cholesky(s)
    linv = np.linalg.inv(l)
    h_ort = linv @ h @ linv.T
    evals, u = np.linalg.eigh(h_ort)
    c = linv.T @ u
    return evals, c


def fermi_dirac(evals: np.ndarray, ef: float, temp_k: float, spin_deg: int) -> np.ndarray:
    if temp_k <= 0:
        return spin_deg * (evals <= ef).astype(float)
    beta = 1.0 / (K_B_EV_PER_K * temp_k)
    x = (evals - ef) * beta
    occ = np.empty_like(evals, dtype=np.float64)
    occ[x > 50] = 0.0
    occ[x < -50] = float(spin_deg)
    mid = (x >= -50) & (x <= 50)
    occ[mid] = float(spin_deg) / (1.0 + np.exp(x[mid]))
    return occ


def write_record(fh, marker_fmt: str, payload: bytes) -> None:
    n = len(payload)
    fh.write(struct.pack(marker_fmt, n))
    fh.write(payload)
    fh.write(struct.pack(marker_fmt, n))


def write_restart_binary(
    output_bin: Path,
    c: np.ndarray,
    evals_ha: np.ndarray,
    occ: np.ndarray,
    endian_word: str,
) -> None:
    n_basis, n_states = c.shape
    n_spin = 1
    endian = "<" if endian_word == "little" else ">"
    marker_fmt = f"{endian}i"

    with output_bin.open("wb") as fh:
        for x in (n_basis, n_states, n_spin):
            write_record(fh, marker_fmt, struct.pack(f"{endian}i", int(x)))

        for ib in range(n_basis):
            for ist in range(n_states):
                write_record(fh, marker_fmt, struct.pack(f"{endian}d", float(c[ib, ist])))

        for ist in range(n_states):
            write_record(
                fh,
                marker_fmt,
                struct.pack(f"{endian}dd", float(evals_ha[ist]), float(occ[ist])),
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepH new-format -> FHI-aims restart_file (non-spin)")
    ap.add_argument("--H", type=Path, default=Path("hamiltonian.h5"))
    ap.add_argument("--S", type=Path, default=Path("overlap.h5"))
    ap.add_argument("--info", type=Path, default=Path("info.json"))
    ap.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    ap.add_argument("--binary-out", type=Path, default=Path("restart_file"))
    ap.add_argument("--n-states", type=int, default=None, help="truncate number of states written to restart")
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--endian", choices=["little", "big"], default="little")
    ap.add_argument("--hermitize", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--R", default="0,0,0", help="lattice vector for (n,5) atom_pairs: Rx,Ry,Rz")
    ap.add_argument("--undo-deeph-transform", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vector-mode", choices=["preserve", "canonical"], default="preserve")
    ap.add_argument("--occupancy-mode", choices=["input", "aufbau"], default="aufbau")
    ap.add_argument("--reference-restart", type=Path, default=None)
    ap.add_argument("--entry-order", choices=["auto", "C", "F"], default="auto")
    args = ap.parse_args()

    r_tuple = tuple(int(x) for x in args.R.split(","))
    if len(r_tuple) != 3:
        raise ValueError("--R must be formatted as '0,0,0'")

    for p in (args.H, args.S, args.info, args.poscar):
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    info = read_info(args.info)
    if info.spinful:
        raise ValueError("Only non-spin case is supported now (spinful=false required).")

    per_atom_species = parse_poscar_species(args.poscar, expected_atoms=info.atoms_quantity)
    offsets, norb_from_map = build_offsets(per_atom_species, info.elements_orbital_map)
    if norb_from_map != info.orbits_quantity:
        raise ValueError(
            f"orbits_quantity mismatch: from map/POSCAR={norb_from_map}, from info={info.orbits_quantity}"
        )

    if args.entry_order == "auto":
        entry_order = "C" if info.orthogonal_basis else choose_entry_order_for_s(
            args.S,
            info.orbits_quantity,
            info.atoms_quantity,
            offsets,
            args.hermitize,
            r_tuple,
        )
    else:
        entry_order = args.entry_order

    h = assemble_from_sparse_chunks(
        args.H,
        n_orb_total=info.orbits_quantity,
        n_atoms=info.atoms_quantity,
        offsets=offsets,
        hermitize=args.hermitize,
        r_tuple=r_tuple,
        entry_order=entry_order,
    )

    s_matrix: np.ndarray | None = None
    if info.orthogonal_basis:
        evals_eV, c = np.linalg.eigh(h)
    else:
        s_matrix = assemble_from_sparse_chunks(
            args.S,
            n_orb_total=info.orbits_quantity,
            n_atoms=info.atoms_quantity,
            offsets=offsets,
            hermitize=args.hermitize,
            r_tuple=r_tuple,
            entry_order=entry_order,
        )
        evals_eV, c = solve_generalized(h, s_matrix)

    if args.undo_deeph_transform:
        c = undo_deeph_basis_transform(c, per_atom_species, info.elements_orbital_map)

    if args.vector_mode == "canonical":
        c = normalize_and_phase_fix(c, s_matrix, phase_fix=True)

    align_info = None
    if args.reference_restart is not None:
        cref, _ev_ref, _oc_ref, nspin_ref = load_restart_binary(args.reference_restart)
        if nspin_ref != 1:
            raise ValueError("reference restart must be non-spin (n_spin=1)")
        c, _perm, minov = align_to_reference(c, cref[:, :c.shape[1]], s_matrix)
        align_info = minov

    occ = fermi_dirac(evals_eV, info.fermi_energy_eV, args.temp_k, spin_deg=2)
    c, evals_eV, occ = truncate_states(c, evals_eV, occ, args.n_states)
    occ = build_occupancy(evals_eV, occ, mode=args.occupancy_mode, spin_deg=2)
    evals_ha = evals_eV / EV_PER_HA

    write_restart_binary(args.binary_out, c, evals_ha, occ, args.endian)

    print("[OK] restart_file written:", args.binary_out)
    print(f"[INFO] n_basis={c.shape[0]}, n_states={c.shape[1]}, n_spin=1")
    print(f"[INFO] R-filter={r_tuple}")
    print(f"[INFO] undo_deeph_transform={args.undo_deeph_transform}, vector_mode={args.vector_mode}")
    print(f"[INFO] occupancy_mode={args.occupancy_mode}")
    print(f"[INFO] hermitize={args.hermitize}")
    print(f"[INFO] entry_order={entry_order}")
    print(f"[INFO] eigenvalue range (eV): [{evals_eV.min():.6f}, {evals_eV.max():.6f}]")
    print(f"[INFO] total electrons from occupations: {occ.sum():.6f}")
    if align_info is not None:
        print(f"[INFO] aligned to reference restart: min|overlap|={align_info:.6e}")


if __name__ == "__main__":
    main()

