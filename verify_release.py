#!/usr/bin/env python3
"""Verify immutable KES artifacts and the 41 published benchmark pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    ROOT / "models" / "lns_rl_kes.pth":
        "ffcf8f83546a665c3cecd8d0b0a904a9d0ce3854a75dd12636fb3239b3efb664",
    ROOT / "results" / "Results_KES2025.xlsx":
        "309f6d6101b13bf5e7a7bebb870538c350fbd58ecf04b98338af50b48b3e2205",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"{status:8} {path.relative_to(ROOT)}  {actual}")
        if actual != expected:
            failures.append(str(path))

    instances = sorted((ROOT / "data" / "paper_instances").glob("*.vrp"))
    solutions = sorted((ROOT / "data" / "paper_solutions").glob("*.sol"))
    instance_keys = {path.stem for path in instances}
    solution_keys = {path.stem for path in solutions}
    print(f"{'OK' if len(instances) == 41 else 'MISMATCH':8} paper instances: {len(instances)}")
    print(f"{'OK' if len(solutions) == 41 else 'MISMATCH':8} paper solutions: {len(solutions)}")
    if len(instances) != 41 or len(solutions) != 41 or instance_keys != solution_keys:
        failures.append("paper benchmark pairs")

    if failures:
        raise SystemExit("Release verification failed: " + ", ".join(failures))
    print("Release verification passed.")


if __name__ == "__main__":
    main()
