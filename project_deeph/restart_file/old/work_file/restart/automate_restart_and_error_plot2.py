#!/usr/bin/env python3
"""Подготовка restart-расчетов и построение распределения ошибок энергий.

Скрипт выполняет 2 шага:
1) Для каждой подпапки (например 1,2,3,...):
   - копирует `deeph_to_restart.py` и `encode.py`;
   - запускает команду конвертации в restart_file;
   - создает `restart/` и копирует туда restart_file, control.in, run.sh, geometry.in;
   - при `--submit` отправляет `sbatch run.sh` из `restart/`.
2) Считывает энергии из `aims.out` (из preprocessed/.../restart/aims.out)
   и из `asi.log` (из batch_calc/.../asi.temp/asi.log), считает
   ошибку по паре папка-к-папке и строит histogram + KDE.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable



AIMS_PATTERNS = [
    re.compile(r"Total\s+energy\s+uncorrected\s*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"\|\s*Total\s+energy\s+uncorrected\s*:\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", re.IGNORECASE),
]

ASI_PATTERNS = [
    re.compile(r"Total\s+energy\s*(?:uncorrected)?\s*[:=]\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"energy[^\n]*?([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*eV", re.IGNORECASE),
]


@dataclass
class EnergyPair:
    folder_id: str
    pred_energy_ev: float
    ref_energy_ev: float

    @property
    def error_ev(self) -> float:
        return self.pred_energy_ev - self.ref_energy_ev


@dataclass
class FolderResult:
    folder_id: str
    status: str
    message: str = ""
    submitted: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Автоматизация restart + сбор ошибок энергий")

    p.add_argument("--preprocessed-root", type=Path, required=True, help="Корень папок 1,2,3... для стадии restart")
    p.add_argument("--batch-root", type=Path, required=True, help="Корень папок 1,2,3... для эталона (asi.temp/asi.log)")

    p.add_argument("--deeph-script", type=Path, required=True, help="Путь к исходному deeph_to_restart.py")
    p.add_argument("--encode-script", type=Path, required=True, help="Путь к исходному encode.py")
    p.add_argument("--control-template", type=Path, help="Если указан, копировать этот control.in в restart")
    p.add_argument("--run-template", type=Path, help="Если указан, копировать этот run.sh в restart")

    p.add_argument(
        "--folder-ids",
        nargs="*",
        default=None,
        help="Явный список ID папок (например: 1 2 3). Если не указан, берется пересечение подпапок в обоих корнях.",
    )

    p.add_argument("--asi-subdir", default="asi.temp", help="Подпапка внутри batch/<id>/..., где лежит asi.log")
    p.add_argument("--restart-subdir", default="restart", help="Имя подпапки для restart-расчета")

    p.add_argument("--n-states", type=int, default=148)
    p.add_argument("--vector-mode", default="preserve")
    p.add_argument("--mode", default="h5")
    p.add_argument("--undo-deeph-transform", action="store_true", default=True)

    p.add_argument("--submit", action="store_true", help="После подготовки отправить sbatch run.sh в restart/ каждой папки")
    p.add_argument("--dry-run", action="store_true", help="Только печатать действия, без выполнения")

    p.add_argument("--plot-out", type=Path, default=Path("energy_error_distribution.png"))
    p.add_argument("--pairs-out", type=Path, default=Path("energy_pairs.tsv"))
    p.add_argument("--status-out", type=Path, default=Path("folder_status.tsv"))

    p.add_argument("--continue-on-error", action="store_true", default=True, help="Продолжать обработку следующих папок при ошибке в текущей")
    p.add_argument("--wait-for-aims", action="store_true", help="Ждать появления aims.out во всех успешно подготовленных папках перед сбором энергий")
    p.add_argument("--wait-timeout-min", type=int, default=0, help="Таймаут ожидания aims.out в минутах (0 = без таймаута)")
    p.add_argument("--wait-poll-sec", type=int, default=60, help="Интервал опроса при ожидании aims.out")

    return p.parse_args()


def discover_folder_ids(pre_root: Path, batch_root: Path, explicit_ids: list[str] | None) -> list[str]:
    if explicit_ids:
        return explicit_ids

    pre_ids = {p.name for p in pre_root.iterdir() if p.is_dir()}
    batch_ids = {p.name for p in batch_root.iterdir() if p.is_dir()}
    common = sorted(pre_ids & batch_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))

    if not common:
        raise RuntimeError("Не найдены общие подпапки между preprocessed-root и batch-root")
    return common




def resolve_source_file(description: str, candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    looked = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"Не найден файл '{description}'. Проверены пути:\n{looked}"
    )

def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"[RUN] cwd={cwd} :: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    print(f"[COPY] {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_restart_for_folder(args: argparse.Namespace, folder_id: str) -> FolderResult:
    folder = args.preprocessed_root / folder_id
    if not folder.exists():
        raise FileNotFoundError(f"Папка не существует: {folder}")

    deeph_local = folder / "deeph_to_restart.py"
    encode_local = folder / "encode.py"
    copy_file(args.deeph_script, deeph_local, args.dry_run)
    copy_file(args.encode_script, encode_local, args.dry_run)

    cmd = [
        "python3",
        "deeph_to_restart.py",
        "--mode",
        args.mode,
        "--n-states",
        str(args.n_states),
        "--vector-mode",
        args.vector_mode,
        "--orbital-types",
        "orbital_types.dat",
        "--binary-out",
        "restart_file",
    ]
    if args.undo_deeph_transform:
        cmd.insert(8, "--undo-deeph-transform")

    run(cmd, cwd=folder, dry_run=args.dry_run)

    restart_dir = folder / args.restart_subdir
    if args.dry_run:
        print(f"[MKDIR] {restart_dir}")
    else:
        restart_dir.mkdir(parents=True, exist_ok=True)

    control_candidates = [args.control_template] if args.control_template else [folder / "control.in", args.preprocessed_root / "control.in"]
    run_candidates = [args.run_template] if args.run_template else [folder / "run.sh", args.preprocessed_root / "run.sh"]

    control_src = resolve_source_file("control.in", control_candidates)
    run_src = resolve_source_file("run.sh", run_candidates)
    geometry_src = resolve_source_file("geometry.in", [folder / "geometry.in"])
    restart_file_src = resolve_source_file("restart_file", [folder / "restart_file"])

    copy_file(restart_file_src, restart_dir / "restart_file", args.dry_run)
    copy_file(control_src, restart_dir / "control.in", args.dry_run)
    copy_file(run_src, restart_dir / "run.sh", args.dry_run)
    copy_file(geometry_src, restart_dir / "geometry.in", args.dry_run)

    submitted = False
    if args.submit:
        run(["sbatch", "run.sh"], cwd=restart_dir, dry_run=args.dry_run)
        submitted = True

    return FolderResult(folder_id=folder_id, status="prepared", submitted=submitted)




def save_status_tsv(statuses: list[FolderResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("folder_id	status	submitted	message\n")
        for st in statuses:
            msg = st.message.replace("\n", " ").replace("\t", " ")
            f.write(f"{st.folder_id}\t{st.status}\t{int(st.submitted)}\t{msg}\n")


def wait_for_aims_outputs(args: argparse.Namespace, folder_ids: list[str]) -> None:
    if not folder_ids:
        print("[WAIT] Нет папок для ожидания aims.out")
        return

    deadline = None
    if args.wait_timeout_min > 0:
        deadline = time.time() + args.wait_timeout_min * 60

    print(f"[WAIT] Ожидание aims.out для {len(folder_ids)} папок...")
    while True:
        missing = []
        for fid in folder_ids:
            aims_path = args.preprocessed_root / fid / args.restart_subdir / "aims.out"
            if not aims_path.exists():
                missing.append(fid)

        if not missing:
            print("[WAIT] Все aims.out найдены, продолжаем сбор энергий")
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[WAIT {now}] Пока нет aims.out в папках: {missing}")

        if deadline is not None and time.time() > deadline:
            print("[WAIT] Достигнут таймаут ожидания. Продолжаем с доступными файлами.")
            return

        time.sleep(max(1, args.wait_poll_sec))

def parse_last_energy(path: Path, patterns: Iterable[re.Pattern[str]]) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    matches: list[float] = []
    for pattern in patterns:
        for m in pattern.finditer(text):
            try:
                matches.append(float(m.group(1)))
            except ValueError:
                continue

    if not matches:
        tail = "\n".join(text.splitlines()[-30:])
        raise ValueError(f"Не удалось извлечь энергию из {path}. Хвост файла:\n{tail}")

    return matches[-1]


def collect_energy_pairs(args: argparse.Namespace, folder_ids: list[str]) -> tuple[list[EnergyPair], list[FolderResult]]:
    pairs: list[EnergyPair] = []
    statuses: list[FolderResult] = []
    for fid in folder_ids:
        aims_path = args.preprocessed_root / fid / args.restart_subdir / "aims.out"
        asi_path = args.batch_root / fid / args.asi_subdir / "asi.log"

        try:
            pred = parse_last_energy(aims_path, AIMS_PATTERNS)
            ref = parse_last_energy(asi_path, ASI_PATTERNS)
            pairs.append(EnergyPair(folder_id=fid, pred_energy_ev=pred, ref_energy_ev=ref))
            statuses.append(FolderResult(folder_id=fid, status="energy_ok"))
        except Exception as exc:
            statuses.append(FolderResult(folder_id=fid, status="energy_failed", message=str(exc)))
            print(f"[WARN] Пропуск папки {fid} при сборе энергий: {exc}")

    return pairs, statuses


def save_pairs_tsv(pairs: list[EnergyPair], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("folder_id\tpred_energy_ev\tref_energy_ev\terror_ev\n")
        for p in pairs:
            f.write(f"{p.folder_id}\t{p.pred_energy_ev:.16e}\t{p.ref_energy_ev:.16e}\t{p.error_ev:.16e}\n")


def plot_error_distribution(pairs: list[EnergyPair], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    errors = [p.error_ev for p in pairs]

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=min(30, max(5, len(errors))), alpha=0.75, edgecolor="black")
    plt.xlabel("Ошибка энергии (pred - ref), eV")
    plt.ylabel("Количество")
    plt.title("Распределение ошибок энергий по папкам")
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.submit and not args.wait_for_aims:
        print("[INFO] Вы используете --submit без --wait-for-aims. График может быть неполным, если aims.out еще не готов.")

    folder_ids = discover_folder_ids(args.preprocessed_root, args.batch_root, args.folder_ids)
    print(f"Найдено папок: {len(folder_ids)} -> {folder_ids}")

    prep_statuses: list[FolderResult] = []
    prepared_ids: list[str] = []

    for fid in folder_ids:
        print(f"\n=== Подготовка restart для папки {fid} ===")
        try:
            st = prepare_restart_for_folder(args, fid)
            prep_statuses.append(st)
            prepared_ids.append(fid)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] Папка {fid}: {msg}")
            prep_statuses.append(FolderResult(folder_id=fid, status="prepare_failed", message=msg))
            if not args.continue_on_error:
                raise

    save_status_tsv(prep_statuses, args.status_out)
    print(f"Сохранен статус подготовки: {args.status_out}")

    if args.wait_for_aims:
        wait_for_aims_outputs(args, prepared_ids)

    print("\n=== Сбор энергий и расчет ошибок ===")
    pairs, energy_statuses = collect_energy_pairs(args, prepared_ids)

    # дописываем результаты энергий в тот же статус-файл
    save_status_tsv(prep_statuses + energy_statuses, args.status_out)

    if not pairs:
        print("[WARN] Нет ни одной валидной пары энергий для построения графика.")
        return

    save_pairs_tsv(pairs, args.pairs_out)
    plot_error_distribution(pairs, args.plot_out)

    errors = [p.error_ev for p in pairs]
    mae = sum(abs(e) for e in errors) / len(errors)
    mean_err = sum(errors) / len(errors)
    print(f"Сохранены пары: {args.pairs_out}")
    print(f"Сохранен график: {args.plot_out}")
    print(f"Статистика: N={len(errors)}, mean={mean_err:.6e} eV, MAE={mae:.6e} eV")


if __name__ == "__main__":
    main()

