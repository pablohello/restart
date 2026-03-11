#!/usr/bin/env python3
"""DeepH -> FHI-aims restart_file.

Один скрипт для двух сценариев:
1) из результатов диагонализации (`out_diag/*`), как раньше;
2) напрямую из `hamiltonians.h5 + overlaps.h5 + info.json` (без отдельного t.py).

По умолчанию пытается AUTO:
- если есть `eigenvectors_C.npy/eigenvalues_eV.txt/occupancies.txt` -> берет их;
- иначе диагонализует H/S напрямую.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

K_B_EV_PER_K = 8.617333262e-5
EV_PER_HA = 27.211386245988


# ---------- DeepH diagonalization helpers ----------
def parse_key(key: str) -> Tuple[int, int, int, int, int]:
    nums = re.findall(r"-?\d+", key)
    if len(nums) != 5:
        raise ValueError(f"Неожиданный ключ dataset: {key!r}")
    return tuple(int(x) for x in nums)  # type: ignore


def infer_orbits_per_site(h5_path: Path, nsites: int, R=(0, 0, 0)) -> Dict[int, int]:
    n_orb_site: Dict[int, int] = {}
    with h5py.File(h5_path, "r") as f:
        for i in range(1, nsites + 1):
            key = f"[{R[0]}, {R[1]}, {R[2]}, {i}, {i}]"
            if key not in f:
                raise KeyError(f"Не найден диагональный блок {key} в {h5_path}")
            ds = f[key]
            if ds.shape[0] != ds.shape[1]:
                raise ValueError(f"Диагональный блок {key} должен быть квадратным")
            n_orb_site[i] = int(ds.shape[0])
    return n_orb_site


def build_offsets(n_orb_site: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    offsets: Dict[int, Tuple[int, int]] = {}
    cur = 0
    for i in sorted(n_orb_site):
        n = n_orb_site[i]
        offsets[i] = (cur, cur + n)
        cur += n
    return offsets


def assemble_matrix(
    h5_path: Path,
    norbits: int,
    offsets: Dict[int, Tuple[int, int]],
    R=(0, 0, 0),
) -> np.ndarray:
    m = np.zeros((norbits, norbits), dtype=np.float64)
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            rx, ry, rz, i, j = parse_key(key)
            if (rx, ry, rz) != tuple(R):
                continue
            block = f[key][...]
            a0, a1 = offsets[i]
            b0, b1 = offsets[j]
            if block.shape != (a1 - a0, b1 - b0):
                raise ValueError(f"Несоответствие размеров блока {key}: {block.shape}")
            m[a0:a1, b0:b1] = block
    return m




def read_orbital_types(path: Path) -> list[list[int]]:
    types: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            types.append([int(x) for x in line.split()])
    return types


def build_sort_phase_tables(types_per_site: list[list[int]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
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


def undo_deeph_basis_transform(c: np.ndarray, orbital_types_path: Path) -> np.ndarray:
    """Undo nonespin_aims2DeepH local basis transform (sort+phase) per atom.

    DeepH basis was built from FHI basis by: parity then sort.
    For eigenvectors to restart(FHI), apply inverse: unsort then parity.
    """
    types = read_orbital_types(orbital_types_path)
    sort_table, phase_table = build_sort_phase_tables(types)

    out = c.copy()
    start = 0
    for sort_idx, phase in zip(sort_table, phase_table):
        n = len(sort_idx)
        stop = start + n
        block_d = out[start:stop, :]

        # unsort: v_orig[sort_idx[k]] = v_sorted[k]
        block_unsorted = np.zeros_like(block_d)
        block_unsorted[sort_idx, :] = block_d

        # undo/apply parity in original order (same sign matrix, P^-1=P)
        block_orig = phase[:, None] * block_unsorted
        out[start:stop, :] = block_orig
        start = stop

    if start != c.shape[0]:
        raise ValueError(
            f"orbital_types total orbitals ({start}) do not match C rows ({c.shape[0]})"
        )
    return out


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


def diagonalize_from_h5(
    h_path: Path,
    s_path: Path,
    info_path: Path,
    r_tuple: tuple[int, int, int],
    temp_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    nsites = int(info["nsites"])
    norbits = int(info["norbits"])
    isorth = bool(info["isorthogonal"])
    isspinful = bool(info["isspinful"])
    ef = float(info["fermi_level"])

    n_orb_site = infer_orbits_per_site(h_path, nsites, R=r_tuple)
    offsets = build_offsets(n_orb_site)
    h = assemble_matrix(h_path, norbits, offsets, R=r_tuple)

    if isorth:
        evals, c = np.linalg.eigh(h)
    else:
        s = assemble_matrix(s_path, norbits, offsets, R=r_tuple)
        evals, c = solve_generalized(h, s)

    spin_deg = 1 if isspinful else 2
    occ = fermi_dirac(evals, ef, temp_k, spin_deg=spin_deg)
    n_spin = 2 if isspinful else 1
    return c.astype(np.float64), evals.astype(np.float64), occ.astype(np.float64), n_spin


# ---------- restart encode helpers ----------
def write_record(fh, marker_fmt: str, payload: bytes) -> None:
    n = len(payload)
    fh.write(struct.pack(marker_fmt, n))
    fh.write(payload)
    fh.write(struct.pack(marker_fmt, n))


def write_restart_binary(
    output_bin: Path,
    c: np.ndarray,
    evals: np.ndarray,
    occ: np.ndarray,
    n_spin: int,
    endian_word: str,
) -> None:
    n_basis, n_states = c.shape
    endian = "<" if endian_word == "little" else ">"
    marker_fmt = f"{endian}i"

    with output_bin.open("wb") as fh:
        for x in (n_basis, n_states, n_spin):
            write_record(fh, marker_fmt, struct.pack(f"{endian}i", int(x)))

        # IMPORTANT: порядок basis-major, затем state, затем spin.
        # Это соответствует тому, как FHI-aims читает restart vectors.
        for ib in range(n_basis):
            for ist in range(n_states):
                for _isp in range(n_spin):
                    val = float(c[ib, ist])
                    write_record(fh, marker_fmt, struct.pack(f"{endian}d", val))

        for ist in range(n_states):
            for _isp in range(n_spin):
                ev = float(evals[ist])
                oc = float(occ[ist])
                write_record(fh, marker_fmt, struct.pack(f"{endian}dd", ev, oc))


def save_diag_outputs(diag_dir: Path, c: np.ndarray, evals: np.ndarray, occ: np.ndarray) -> None:
    diag_dir.mkdir(parents=True, exist_ok=True)
    np.save(diag_dir / "eigenvectors_C.npy", c)
    np.savetxt(diag_dir / "eigenvalues_eV.txt", evals, fmt="%.12f")
    np.savetxt(diag_dir / "occupancies.txt", occ, fmt="%.8f")


def load_diag(diag_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_path = diag_dir / "eigenvectors_C.npy"
    e_path = diag_dir / "eigenvalues_eV.txt"
    o_path = diag_dir / "occupancies.txt"
    if not c_path.exists() or not e_path.exists() or not o_path.exists():
        raise FileNotFoundError("diag файлы не найдены")

    c = np.load(c_path)
    evals = np.loadtxt(e_path, dtype=np.float64)
    occ = np.loadtxt(o_path, dtype=np.float64)

    if c.ndim != 2:
        raise ValueError(f"eigenvectors_C.npy должен быть 2D, получено {c.shape}")
    if evals.ndim != 1 or occ.ndim != 1:
        raise ValueError("eigenvalues/occupancies должны быть 1D")

    n_basis, n_states = c.shape
    if evals.shape[0] != n_states or occ.shape[0] != n_states:
        raise ValueError("Размеры C/evals/occ не согласованы")
    return c.astype(np.float64), evals.astype(np.float64), occ.astype(np.float64)


def truncate_states(
    c: np.ndarray,
    evals: np.ndarray,
    occ: np.ndarray,
    n_states_target: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_states_target is None:
        return c, evals, occ
    if n_states_target <= 0 or n_states_target > c.shape[1]:
        raise ValueError(f"--n-states должен быть в диапазоне 1..{c.shape[1]}")
    return c[:, :n_states_target], evals[:n_states_target], occ[:n_states_target]


def to_hartree(evals: np.ndarray, eigenvalue_unit: str) -> np.ndarray:
    if eigenvalue_unit == "Ha":
        return evals.astype(np.float64)
    if eigenvalue_unit == "eV":
        return (evals / EV_PER_HA).astype(np.float64)
    raise ValueError("--eigenvalue-unit должен быть eV или Ha")


def sanitize_occupancies(occ: np.ndarray, spin_deg: int) -> np.ndarray:
    return np.clip(occ.astype(np.float64), 0.0, float(spin_deg))


def build_occupancy(evals: np.ndarray, occ_in: np.ndarray, mode: str, spin_deg: int = 2) -> np.ndarray:
    """Собирает occupations для restart.

    mode=input  : взять входные occupancies (после clip).
    mode=aufbau : заполнить по возрастанию eigenvalues с тем же числом электронов,
                  что и у входного occ (округление до ближайшего целого).
    """
    occ = sanitize_occupancies(occ_in, spin_deg=spin_deg)
    if mode == "input":
        return occ
    if mode != "aufbau":
        raise ValueError("--occupancy-mode должен быть input или aufbau")

    n_elec = int(round(float(np.sum(occ))))
    if n_elec < 0:
        n_elec = 0
    max_e = spin_deg * len(evals)
    n_elec = min(n_elec, max_e)

    occ_out = np.zeros_like(occ, dtype=np.float64)
    full = n_elec // spin_deg
    rem = n_elec % spin_deg
    if full > 0:
        occ_out[:full] = float(spin_deg)
    if full < len(occ_out) and rem > 0:
        occ_out[full] = float(rem)
    return occ_out


def normalize_and_phase_fix(c: np.ndarray, s: np.ndarray | None, phase_fix: bool) -> np.ndarray:
    """Нормировка собственных векторов и фиксация знака (gauge) для стабильного restart.

    - при наличии S: нормировка по C^T S C = I
    - иначе: евклидова нормировка
    - фиксация знака: максимальная по модулю компонента делается положительной
    """
    c_out = c.astype(np.float64, copy=True)
    n_states = c_out.shape[1]

    for ist in range(n_states):
        v = c_out[:, ist]
        if s is None:
            nrm2 = float(v @ v)
        else:
            nrm2 = float(v @ (s @ v))
        if nrm2 <= 0:
            raise ValueError(f"Некорректная норма eigenvector[{ist}] = {nrm2}")
        c_out[:, ist] = v / np.sqrt(nrm2)

        if phase_fix:
            idx = int(np.argmax(np.abs(c_out[:, ist])))
            if c_out[idx, ist] < 0.0:
                c_out[:, ist] *= -1.0

    return c_out


def orthogonality_report(c: np.ndarray, s: np.ndarray | None) -> tuple[float, float]:
    if s is None:
        gram = c.T @ c
    else:
        gram = c.T @ s @ c
    diag_err = float(np.max(np.abs(np.diag(gram) - 1.0)))
    off = gram - np.diag(np.diag(gram))
    off_err = float(np.max(np.abs(off))) if off.size else 0.0
    return diag_err, off_err



def _read_record(fh, endian: str) -> bytes:
    marker = fh.read(4)
    if len(marker) != 4:
        raise EOFError("Unexpected EOF while reading record marker")
    n = struct.unpack(f"{endian}i", marker)[0]
    if n < 0 or n > 10**8:
        raise ValueError("Invalid record size marker")
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
                for _isp in range(n_spin):
                    val = struct.unpack(f"{endian}d", _read_record(fh, endian))[0]
                    if _isp == 0:
                        c[ib, ist] = val

        evals = np.zeros(n_states, dtype=np.float64)
        occ = np.zeros(n_states, dtype=np.float64)
        for ist in range(n_states):
            for _isp in range(n_spin):
                ev, oc = struct.unpack(f"{endian}dd", _read_record(fh, endian))
                if _isp == 0:
                    evals[ist] = ev
                    occ[ist] = oc

    return c, evals, occ, n_spin


def align_to_reference(c: np.ndarray, c_ref: np.ndarray, s: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    if c.shape != c_ref.shape:
        raise ValueError(f"reference restart shape mismatch: {c_ref.shape} vs {c.shape}")

    if s is None:
        ov = c_ref.T @ c
    else:
        ov = c_ref.T @ s @ c

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
    if s is None:
        diag_ov = np.einsum("ij,ij->j", c_ref, c2)
    else:
        diag_ov = np.einsum("ij,ij->j", c_ref, s @ c2)

    signs = np.where(diag_ov < 0.0, -1.0, 1.0)
    c2 = c2 * signs[np.newaxis, :]

    if s is None:
        final_ov = np.abs(np.einsum("ij,ij->j", c_ref, c2))
    else:
        final_ov = np.abs(np.einsum("ij,ij->j", c_ref, s @ c2))

    return c2, perm, float(np.min(final_ov))



def check_control_in(control_in: Path) -> tuple[bool, str]:
    """Проверка, что control.in действительно читает restart_file."""
    text = control_in.read_text(encoding="utf-8", errors="ignore")
    low = text.lower()
    has_restart = "restart_read_only" in low or "restart" in low and "restart_file" in low
    if not has_restart:
        return False, (
            "В control.in не найдено restart_read_only restart_file. "
            "FHI-aims запустится с суперпозицией свободных атомов, "
            "и спектр/энергия будут отличаться от restart."
        )
    return True, "OK"

def main() -> None:
    ap = argparse.ArgumentParser(description="DeepH -> FHI-aims restart_file")
    ap.add_argument("--mode", choices=["auto", "diag", "h5"], default="auto")

    ap.add_argument("--diag-dir", type=Path, default=Path("out_diag"))

    ap.add_argument("--H", type=Path, default=Path("hamiltonians.h5"))
    ap.add_argument("--S", type=Path, default=Path("overlaps.h5"))
    ap.add_argument("--info", type=Path, default=Path("info.json"))
    ap.add_argument("--R", default="0,0,0")
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--save-diag", action="store_true", help="сохранить diag-артефакты в --diag-dir")

    ap.add_argument("--n-spin", type=int, default=None, help="принудительно n_spin (обычно 1 для spin none)")
    ap.add_argument(
        "--n-states",
        type=int,
        default=None,
        help="обрезать число состояний (важно, чтобы совпало с FHI-aims текущим n_states)",
    )
    ap.add_argument("--endian", choices=["little", "big"], default="little")
    ap.add_argument(
        "--eigenvalue-unit",
        choices=["eV", "Ha"],
        default="eV",
        help="Единицы входных eigenvalues из DeepH/diag. Для FHI-aims в restart записывается Ha.",
    )
    ap.add_argument("--binary-out", type=Path, default=Path("restart_file"))
    ap.add_argument("--orbital-types", type=Path, default=Path("orbital_types.dat"),
                    help="orbital_types.dat from nonespin_aims2DeepH")
    ap.add_argument("--undo-deeph-transform", action=argparse.BooleanOptionalAction, default=True,
                    help="Undo local sort/parity from nonespin_aims2DeepH before writing restart")
    ap.add_argument("--occupancy-mode", choices=["input", "aufbau"], default="aufbau",
                    help="Как писать occupations в restart (рекомендуется aufbau для стабильности)")
    ap.add_argument(
        "--vector-mode",
        choices=["preserve", "canonical"],
        default="preserve",
        help="preserve: писать eigenvectors как есть; canonical: S-нормировка + фиксация знака",
    )
    ap.add_argument("--control-in", type=Path, default=None, help="Опционально: проверить, что control.in читает restart_file")
    ap.add_argument("--reference-restart", type=Path, default=None,
                    help="Опционально: путь к эталонному restart_file для выравнивания знаков/порядка векторов")

    args = ap.parse_args()

    r_tuple = tuple(int(x) for x in args.R.split(","))
    if len(r_tuple) != 3:
        raise ValueError("--R должен быть вида 0,0,0")

    mode = args.mode
    if mode == "auto":
        diag_ok = all((args.diag_dir / n).exists() for n in ["eigenvectors_C.npy", "eigenvalues_eV.txt", "occupancies.txt"])
        mode = "diag" if diag_ok else "h5"

    s_matrix: np.ndarray | None = None
    if mode == "diag":
        c, evals, occ = load_diag(args.diag_dir)
        if args.S.exists():
            try:
                info_for_s = json.loads(args.info.read_text(encoding="utf-8")) if args.info.exists() else {}
                nsites_s = int(info_for_s.get("nsites")) if "nsites" in info_for_s else None
                norbits_s = int(info_for_s.get("norbits")) if "norbits" in info_for_s else c.shape[0]
                if nsites_s is not None:
                    n_orb_site_s = infer_orbits_per_site(args.S, nsites_s, R=r_tuple)
                    offsets_s = build_offsets(n_orb_site_s)
                    s_matrix = assemble_matrix(args.S, norbits_s, offsets_s, R=r_tuple)
            except Exception:
                s_matrix = None
        if args.n_spin is None:
            info_path = args.diag_dir / "info.json"
            if info_path.exists():
                info = json.loads(info_path.read_text(encoding="utf-8"))
                n_spin = 2 if bool(info.get("isspinful", False)) else 1
            else:
                n_spin = 1
        else:
            n_spin = args.n_spin
    else:
        missing = [p for p in (args.H, args.S, args.info) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Не найдены входные файлы: {missing}")
        c, evals, occ, n_spin_auto = diagonalize_from_h5(args.H, args.S, args.info, r_tuple, args.T)
        n_spin = n_spin_auto if args.n_spin is None else args.n_spin
        info_for_s = json.loads(args.info.read_text(encoding="utf-8"))
        if not bool(info_for_s.get("isorthogonal", False)):
            n_orb_site_s = infer_orbits_per_site(args.S, int(info_for_s["nsites"]), R=r_tuple)
            offsets_s = build_offsets(n_orb_site_s)
            s_matrix = assemble_matrix(args.S, int(info_for_s["norbits"]), offsets_s, R=r_tuple)
        if args.save_diag:
            save_diag_outputs(args.diag_dir, c, evals, occ)

    if n_spin != 1:
        raise ValueError("Пока поддерживается только spin none (n_spin=1).")

    c, evals, occ = truncate_states(c, evals, occ, args.n_states)

    if args.undo_deeph_transform:
        if args.orbital_types.exists():
            c = undo_deeph_basis_transform(c, args.orbital_types)
        else:
            print(f"[WARN] orbital_types not found: {args.orbital_types}; skip undo transform")

    if args.vector_mode == "canonical":
        c = normalize_and_phase_fix(c, s_matrix, phase_fix=True)

    align_info = None
    if args.reference_restart is not None:
        cref, _ev_ref, _oc_ref, nspin_ref = load_restart_binary(args.reference_restart)
        if nspin_ref != 1:
            raise ValueError("reference restart must be non-spin (n_spin=1)")
        c, perm, minov = align_to_reference(c, cref[:, :c.shape[1]], s_matrix)
        align_info = (perm, minov)

    evals = to_hartree(evals, args.eigenvalue_unit)
    occ = build_occupancy(evals, occ, mode=args.occupancy_mode, spin_deg=2)
    write_restart_binary(args.binary_out, c, evals, occ, n_spin=n_spin, endian_word=args.endian)

    print("[OK] restart_file written:", args.binary_out)
    print(f"[INFO] n_basis={c.shape[0]}, n_states={c.shape[1]}, n_spin={n_spin}, mode={mode}")
    print(f"[INFO] eigenvalue range in restart (Ha): [{evals.min():.6f}, {evals.max():.6f}]")
    print(f"[INFO] eigenvalue range in restart (eV): [{(evals.min()*EV_PER_HA):.6f}, {(evals.max()*EV_PER_HA):.6f}]")
    print(f"[INFO] occupancy mode={args.occupancy_mode}, total electrons from restart={np.sum(occ):.6f}")
    print(f"[INFO] undo_deeph_transform={args.undo_deeph_transform}, orbital_types={args.orbital_types}")
    print("[TIP] Если FHI-aims пишет 'restart file cannot belong to current system', проверьте --n-states.")
    print("[TIP] Если eigenvalues в restart завышены ~в 27.2 раза, проверьте --eigenvalue-unit (обычно eV).")
    print(f"[INFO] vector mode={args.vector_mode}")
    if args.vector_mode == "canonical":
        d_err, o_err = orthogonality_report(c, s_matrix)
        print(f"[INFO] orthogonality check: max|diag-1|={d_err:.3e}, max|offdiag|={o_err:.3e}")
    if align_info is not None:
        _perm, minov = align_info
        print(f"[INFO] aligned to reference restart: min|overlap|={minov:.6e}")

    if args.control_in is not None:
        ok, msg = check_control_in(args.control_in)
        if ok:
            print(f"[CHECK] control.in OK: {args.control_in}")
        else:
            print(f"[WARN] {msg}")


if __name__ == "__main__":
    main()

