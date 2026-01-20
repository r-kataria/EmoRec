#!/usr/bin/env python3
import subprocess
from pathlib import Path
from typing import Iterable, Optional


def run_cmd(args: Iterable[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(list(args), check=True, cwd=str(cwd) if cwd else None)


def echo(msg: str) -> None:
    print(msg, flush=True)


BASE = Path("cache/datasets")


# Download helpers
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

# Dataset downloaders

def download_redial() -> None:
    redial = BASE / "redial"
    git_clone("https://github.com/ReDialData/website", redial)

    run_cmd(["git", "checkout", "data"], cwd=redial)
    
    redial_zip = redial / "redial_dataset.zip"
    if not redial_zip.exists():
        raise FileNotFoundError(f"Expected ReDial zip not found: {redial_zip}")

    data_dir = redial / "data"
    if data_dir.exists():
        echo("ReDial dataset already extracted, skipping unzip")
    else:
        echo("Extracting ReDial dataset")
        run_cmd(["unzip", "-o", str(redial_zip), "-d", str(data_dir)])

def download_cosrec() -> None:
    cosrec = BASE / "cosrec"
    git_clone("https://github.com/CAMEO-22/CoSRec", cosrec)

def download_movielens() -> None:
    ml_dir = BASE / "movielens/ml-25m"
    ensure_dir(ml_dir)

    ml_zip = ml_dir / "ml-25m.zip"
    download(
        "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
        ml_zip,
    )
    unzip(ml_zip, ml_dir)

def download_amazon_2023() -> None:
    amazon = BASE / "amazon_2023"
    ensure_dir(amazon)

    asin2cat = amazon / "asin2category.json"
    download(
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json",
        asin2cat,
    )

    meta_dir = amazon / "raw/meta_categories"
    review_dir = amazon / "raw/review_categories"
    ensure_dir(meta_dir)
    ensure_dir(review_dir)

    if not any(meta_dir.iterdir()):
        echo("Downloading Amazon meta_categories")
        run_cmd(
            [
                "wget",
                "-r",
                "-np",
                "-nd",
                "-e",
                "robots=off",
                "-A",
                "meta_*.jsonl.gz",
                "-R",
                "index.html*",
                "-P",
                str(meta_dir),
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/",
            ]
        )
    else:
        echo("Amazon meta_categories exists, skipping")

    if not any(review_dir.iterdir()):
        echo("Downloading Amazon review_categories")
        run_cmd(
            [
                "wget",
                "-r",
                "-np",
                "-nd",
                "-e",
                "robots=off",
                "-A",
                "*.jsonl.gz",
                "-R",
                "index.html*",
                "-P",
                str(review_dir),
                "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/",
            ]
        )
    else:
        echo("Amazon review_categories exists, skipping")

def download_msmarco() -> None:
    msmarco = BASE / "msmarco"
    ensure_dir(msmarco)

    msmarco_tar = msmarco / "msmarco_v2.1_doc_segmented.tar"
    download(
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar",
        msmarco_tar,
    )
    untar(msmarco_tar, msmarco)

# Main

def download_all() -> None:
    ensure_dir(BASE)
    download_redial()
    download_cosrec()
    download_movielens()
    download_amazon_2023()
    # download_msmarco()
    echo("All data downloaded!!")

if __name__ == "__main__":
    download_all()
