#!/usr/bin/env bash
set -euo pipefail

# Always use ./cache/datasets
mkdir -p cache/datasets

# Git repos
git clone https://github.com/ReDialData/website cache/datasets/redial || true
git clone https://github.com/CAMEO-22/CoSRec cache/datasets/cosrec || true

# ReDial: checkout data branch + unzip
(cd cache/datasets/redial && git fetch --all && git checkout data)
mkdir -p cache/datasets/redial
unzip -o cache/datasets/redial/redial_dataset.zip -d cache/datasets/redial

# MovieLens 25M
mkdir -p cache/datasets/movielens/ml-25m
wget -c -O cache/datasets/movielens/ml-25m/ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip -o cache/datasets/movielens/ml-25m/ml-25m.zip -d cache/datasets/movielens/ml-25m

# Amazon Reviews 2023
mkdir -p cache/datasets/amazon_2023
wget -c -O cache/datasets/amazon_2023/asin2category.json \
  https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/asin2category.json

mkdir -p cache/datasets/amazon_2023/raw/meta_categories
wget -r -np -nd -e robots=off -A 'meta_*.jsonl.gz' -R 'index.html*' \
  -P cache/datasets/amazon_2023/raw/meta_categories \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/

mkdir -p cache/datasets/amazon_2023/raw/review_categories
wget -r -np -nd -e robots=off -A '*.jsonl.gz' -R 'index.html*' \
  -P cache/datasets/amazon_2023/raw/review_categories \
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/

mkdir -p cache/datasets/msmarco/
wget -c -O cache/datasets/msmarco/msmarco_v2.1_doc_segmented.tar \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar
tar -xvf cache/datasets/msmarco/msmarco_v2.1_doc_segmented.tar -C cache/datasets/msmarco/


echo "All data downloaded!!"