#!/usr/bin/env python3
"""Verify the offline wheelhouse (voice_processing/wheels) against the locks.

Every pin in scripts/wheelhouse/*.txt must have a matching wheel or sdist file
in the wheelhouse, otherwise an offline build (--network none) will fail.

Usage:
    python scripts/verify_wheelhouse.py            # check all locks
    python scripts/verify_wheelhouse.py lock_voice_clean.txt
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WHEELS = Path(__import__("os").environ.get("WHEELS_DIR") or ROOT / "voice_processing" / "wheels")
LOCK_DIR = ROOT / "scripts" / "wheelhouse"
DEFAULT_LOCKS = [
    "lock_base_full_clean.txt",
    "lock_voice_clean.txt",
    "lock_orch_clean.txt",
    "lock_jtalk_clean.txt",
    "req_sdists.txt",
]


def pin_pattern(pin: str) -> re.Pattern:
    spec = re.split(r"(==|>=|<=|~=|>|<)", pin, maxsplit=1)
    name = spec[0].strip()
    ver_part = re.escape(spec[2].strip()) if len(spec) >= 3 and spec[1] == "==" else r"[^-]+"
    name_re = "[-_.]".join(re.escape(part) for part in re.split(r"[-_.]+", name))
    return re.compile(
        "^" + name_re + "[-_.]" + ver_part + r"([-.].*)?\.(whl|tar\.gz|zip)$",
        re.IGNORECASE,
    )


def main() -> int:
    locks = sys.argv[1:] or DEFAULT_LOCKS
    patterns = []
    for lock in locks:
        path = Path(lock)
        if not path.is_file():
            path = LOCK_DIR / lock
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            patterns.append((path.name, line, pin_pattern(line)))

    files = list(WHEELS.iterdir()) if WHEELS.is_dir() else []
    missing = [
        f"{lock}: {pin}"
        for lock, pin, pat in patterns
        if not any(pat.match(f.name) for f in files)
    ]
    print(f"wheelhouse: {WHEELS} ({len(files)} files)")
    print(f"checked {len(patterns)} pins from {len(locks)} lock files -> {len(missing)} missing")
    for m in missing:
        print("  MISSING", m)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
