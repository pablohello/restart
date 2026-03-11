#!/usr/bin/env python3
"""Legacy helper.

Ранее `encode.py` кодировал текстовый restart.
Теперь основной рабочий путь: `deeph_to_restart.py` (без промежуточного текста).
Этот файл оставлен только как явное уведомление, чтобы не ломать старые инструкции.
"""

from __future__ import annotations

import sys


def main() -> None:
    msg = (
        "encode.py is deprecated in this repo.\n"
        "Use: python3 deeph_to_restart.py --mode h5 --binary-out restart_file\n"
        "or : python3 deeph_to_restart.py --mode diag --diag-dir out_diag --binary-out restart_file\n"
    )
    sys.stderr.write(msg)
    raise SystemExit(2)


if __name__ == "__main__":
    main()

