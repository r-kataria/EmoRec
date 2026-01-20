from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional


def run_cmd(args: Iterable[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(list(args), check=True, cwd=str(cwd) if cwd else None)


def echo(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        echo(f"Created directory: {path}")
    else:
        echo(f"Exists, skipping dir: {path}")


def git_clone(url: str, dest: Path) -> None:
    if dest.exists():
        echo(f"Repo exists, skipping clone: {dest}")
    else:
        echo(f"Cloning {url} -> {dest}")
        run_cmd(["git", "clone", url, str(dest)])


def download(url: str, out: Path) -> None:
    if out.exists():
        echo(f"Exists, skipping download: {out}")
    else:
        echo(f"Downloading {url}")
        run_cmd(["wget", "-O", str(out), url])


def unzip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        echo(f"Unzipped content exists, skipping unzip: {out_dir}")
    else:
        echo(f"Unzipping {zip_path}")
        run_cmd(["unzip", "-o", str(zip_path), "-d", str(out_dir)])


def untar(tar_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        echo(f"Extracted content exists, skipping untar: {out_dir}")
    else:
        echo(f"Extracting {tar_path}")
        run_cmd(["tar", "-xvf", str(tar_path), "-C", str(out_dir)])

