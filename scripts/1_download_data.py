#!/usr/bin/env python3
from pathlib import Path

from utils.download_helpers import download, echo, ensure_dir, git_clone, run_cmd, untar, unzip


BASE = Path("cache/datasets")


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
    download("https://files.grouplens.org/datasets/movielens/ml-25m.zip", ml_zip)
    unzip(ml_zip, ml_dir)

def download_amazon_2023() -> None:
    amazon = BASE / "amazon_2023"
    ensure_dir(amazon)

    asin2cat = amazon / "asin2category.json"
    download("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json", asin2cat)

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
    download("https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar", msmarco_tar)
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
